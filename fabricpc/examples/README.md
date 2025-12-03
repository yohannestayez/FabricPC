# Predictive Coding Examples

This directory contains example scripts demonstrating FabricPC.

## Quick Start

```bash
# Run MNIST demo
export PYTHONPATH=$PYTHONPATH:$(pwd)
python examples/mnist_demo.py
```

## Examples

### `mnist_demo.py`

**Description**: MNIST classification example demonstrating the basic PC workflow.

**Architecture**:
- Input layer: 784 units (28x28 flattened images)
- Hidden layer 1: 256 units (sigmoid activation)
- Hidden layer 2: 64 units (sigmoid activation)
- Output layer: 10 units (class logits)

**Configuration**:
- Optimizer: Adam (lr=1e-3)
- Inference: 20 steps @ eta=0.05
- Training: 20 epochs, batch size 200

**Expected Results**:
```
Epoch 1/20, Loss: 0.4529
Epoch 2/20, Loss: 0.1903
Epoch 3/20, Loss: 0.0757
Epoch 4/20, Loss: 0.0482
...
Epoch 18/20, Loss: 0.0052
Epoch 19/20, Loss: 0.0047
Epoch 20/20, Loss: 0.0043
Avg Training time: 2.05 seconds per epoch

Evaluating...
Test Accuracy: 97.93%
Test Loss: 0.0349
```

**Features Demonstrated**:
- Model creation with `create_pc_graph`
- Training with `train_pcn`
- Evaluation with `evaluate_pcn`
- One-hot encoding for classification
- Data loading with PyTorch DataLoader

## Coming Soon

- Deeper architectures
- Custom energy functional and non-Gaussian likelihoods

## Notes

### Data Loading Warning

You may see warnings about `os.fork()` when using PyTorch DataLoader:
```
RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code,
and JAX is multithreaded, so this will likely lead to a deadlock.
```

This is harmless for now but will be addressed in future versions by switching to JAX-native data pipelines.

### Performance

First batch will be slow due to JIT compilation (~5-10 seconds). Subsequent batches are fast.

## Troubleshooting
**Import Error**: Make sure PYTHONPATH is set:

