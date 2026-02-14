# superb

Simple Equilibrium Propagation (EP) implementation for MNIST in PyTorch.

## What is included

- `ep_mnist.py`: 2-layer EP network (input → hidden → output) with hard-sigmoid state clamping.
- Free and nudged relaxation phases with the EP learning rule.
- Training/evaluation loop on MNIST.
- Plot generation for:
  - Energy decrease during free and nudged relaxation.
  - Train/test loss decrease over epochs.

## Run

```bash
python ep_mnist.py --epochs 8 --train-subset 10000 --test-subset 2000
```

Outputs a figure at `artifacts/ep_training_curves.png`.
