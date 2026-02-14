import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def hard_sigmoid(x: Tensor) -> Tensor:
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


@dataclass
class EPConfig:
    input_dim: int = 28 * 28
    hidden_dim: int = 256
    output_dim: int = 10
    beta: float = 0.5
    state_lr: float = 0.2
    weight_lr: float = 0.05
    free_steps: int = 25
    nudged_steps: int = 10


class EPNetwork:
    def __init__(self, config: EPConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        scale_1 = (2.0 / (config.input_dim + config.hidden_dim)) ** 0.5
        scale_2 = (2.0 / (config.hidden_dim + config.output_dim)) ** 0.5
        self.W1 = torch.randn(config.input_dim, config.hidden_dim, device=device) * scale_1
        self.W2 = torch.randn(config.hidden_dim, config.output_dim, device=device) * scale_2

    def energy(self, x: Tensor, h: Tensor, y: Tensor) -> Tensor:
        quadratic = 0.5 * (h.pow(2).sum(dim=1) + y.pow(2).sum(dim=1))
        interaction_1 = (x @ self.W1 * h).sum(dim=1)
        interaction_2 = (h @ self.W2 * y).sum(dim=1)
        return (quadratic - interaction_1 - interaction_2).mean()

    def total_energy(self, x: Tensor, h: Tensor, y: Tensor, target: Tensor | None, beta: float) -> Tensor:
        e = self.energy(x, h, y)
        if target is None:
            return e
        cost = 0.5 * (y - target).pow(2).sum(dim=1).mean()
        return e + beta * cost

    def relax(
        self,
        x: Tensor,
        h0: Tensor,
        y0: Tensor,
        target: Tensor | None,
        beta: float,
        steps: int,
    ) -> Tuple[Tensor, Tensor, List[float]]:
        h = h0.clone().detach()
        y = y0.clone().detach()
        trajectory: List[float] = []

        for _ in range(steps):
            h.requires_grad_(True)
            y.requires_grad_(True)
            total_e = self.total_energy(x, h, y, target, beta)
            grad_h, grad_y = torch.autograd.grad(total_e, (h, y))
            with torch.no_grad():
                h = hard_sigmoid(h - self.config.state_lr * grad_h)
                y = hard_sigmoid(y - self.config.state_lr * grad_y)
                trajectory.append(float(self.total_energy(x, h, y, target, beta).item()))

        return h.detach(), y.detach(), trajectory

    def dE_dW(self, x: Tensor, h: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        n = x.size(0)
        grad_W1 = -(x.t() @ h) / n
        grad_W2 = -(h.t() @ y) / n
        return grad_W1, grad_W2

    @torch.no_grad()
    def train_step(self, x: Tensor, labels: Tensor) -> Dict[str, object]:
        target = F.one_hot(labels, num_classes=self.config.output_dim).float()
        batch_size = x.size(0)

        h_init = torch.zeros(batch_size, self.config.hidden_dim, device=self.device)
        y_init = torch.zeros(batch_size, self.config.output_dim, device=self.device)

        h_free, y_free, free_energy = self.relax(
            x,
            h_init,
            y_init,
            target=None,
            beta=0.0,
            steps=self.config.free_steps,
        )

        h_nudged, y_nudged, nudged_energy = self.relax(
            x,
            h_free,
            y_free,
            target=target,
            beta=self.config.beta,
            steps=self.config.nudged_steps,
        )

        g1_free, g2_free = self.dE_dW(x, h_free, y_free)
        g1_nudged, g2_nudged = self.dE_dW(x, h_nudged, y_nudged)

        delta_w1 = (g1_nudged - g1_free) / self.config.beta
        delta_w2 = (g2_nudged - g2_free) / self.config.beta

        self.W1 -= self.config.weight_lr * delta_w1
        self.W2 -= self.config.weight_lr * delta_w2

        batch_loss = float(F.mse_loss(y_free, target).item())
        batch_acc = float((y_free.argmax(dim=1) == labels).float().mean().item())

        return {
            "loss": batch_loss,
            "acc": batch_acc,
            "free_energy": free_energy,
            "nudged_energy": nudged_energy,
        }

    @torch.no_grad()
    def predict(self, x: Tensor, steps: int = 25) -> Tensor:
        h = torch.zeros(x.size(0), self.config.hidden_dim, device=self.device)
        y = torch.zeros(x.size(0), self.config.output_dim, device=self.device)
        h, y, _ = self.relax(x, h, y, target=None, beta=0.0, steps=steps)
        return y


def build_loaders(batch_size: int, train_subset: int, test_subset: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    if train_subset > 0:
        train_ds = Subset(train_ds, list(range(train_subset)))
    if test_subset > 0:
        test_ds = Subset(test_ds, list(range(test_subset)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(model: EPNetwork, loader: DataLoader) -> Tuple[float, float]:
    losses, accs = [], []
    for x, labels in loader:
        x = x.to(model.device)
        labels = labels.to(model.device)
        pred = model.predict(x)
        target = F.one_hot(labels, num_classes=model.config.output_dim).float()
        losses.append(float(F.mse_loss(pred, target).item()))
        accs.append(float((pred.argmax(dim=1) == labels).float().mean().item()))
    return sum(losses) / len(losses), sum(accs) / len(accs)


def plot_curves(history: Dict[str, List[float]], free_energy: List[float], nudged_energy: List[float], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(free_energy, label="Free phase energy")
    axes[0].plot(range(len(free_energy), len(free_energy) + len(nudged_energy)), nudged_energy, label="Nudged phase energy")
    axes[0].set_title("Energy during state relaxation")
    axes[0].set_xlabel("Relaxation step")
    axes[0].set_ylabel("Energy")
    axes[0].legend()

    axes[1].plot(history["train_loss"], label="Train loss")
    axes[1].plot(history["test_loss"], label="Test loss")
    axes[1].set_title("Loss across epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE loss")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Equilibrium Propagation on MNIST with a 1-hidden-layer network")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--free-steps", type=int, default=25)
    parser.add_argument("--nudged-steps", type=int, default=10)
    parser.add_argument("--state-lr", type=float, default=0.2)
    parser.add_argument("--weight-lr", type=float, default=0.05)
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--test-subset", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot-path", type=Path, default=Path("artifacts/ep_training_curves.png"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_loaders(args.batch_size, args.train_subset, args.test_subset)
    config = EPConfig(
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        free_steps=args.free_steps,
        nudged_steps=args.nudged_steps,
        state_lr=args.state_lr,
        weight_lr=args.weight_lr,
    )
    model = EPNetwork(config, device)

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    captured_free_energy: List[float] = []
    captured_nudged_energy: List[float] = []

    for epoch in range(args.epochs):
        epoch_losses, epoch_accs = [], []
        for step, (x, labels) in enumerate(train_loader):
            x = x.to(device)
            labels = labels.to(device)
            stats = model.train_step(x, labels)
            epoch_losses.append(float(stats["loss"]))
            epoch_accs.append(float(stats["acc"]))
            if epoch == 0 and step == 0:
                captured_free_energy = list(stats["free_energy"])
                captured_nudged_energy = list(stats["nudged_energy"])

        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_acc = sum(epoch_accs) / len(epoch_accs)
        test_loss, test_acc = evaluate(model, test_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    plot_curves(history, captured_free_energy, captured_nudged_energy, args.plot_path)
    print(f"Saved training visualization to {args.plot_path}")


if __name__ == "__main__":
    main()
