"""
Multi-GPU MNIST Example (JAX)
==============================

This example demonstrates data-parallel training across multiple GPUs using pmap.

Key features:
- Automatic device detection
- Batch sharding across GPUs
- Gradient averaging with pmean
- Linear scaling with number of devices

Note: This will work with 1 GPU (falls back to single-GPU) but the real benefits
come with 2+ GPUs.
"""

import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

from fabricpc_jax.models import create_pc_graph
from fabricpc_jax.training import train_pcn_multi_gpu, evaluate_pcn_multi_gpu

# Set random seed
key = jax.random.PRNGKey(42)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# fmt: off

config = {
    "node_list": [
        {"name": "pixels", "dim": 784, "activation": {"type": "identity"}, "type": "linear"},
        {"name": "h1",     "dim": 256, "activation": {"type": "relu"}, "type": "linear"},
        {"name": "h2",     "dim": 128, "activation": {"type": "relu"}, "type": "linear"},
        {"name": "class",  "dim": 10,  "activation": {"type": "identity"}, "type": "linear"},
    ],

    "edge_list": [
        {"source_name": "pixels", "target_name": "h1",    "slot": "in"},
        {"source_name": "h1",     "target_name": "h2",    "slot": "in"},
        {"source_name": "h2",     "target_name": "class", "slot": "in"},
    ],

    "task_map": {"x": "pixels", "y": "class"},
}

train_config = {
    "num_epochs": 10,
    "T_infer": 20,
    "eta_infer": 0.05,
    "optimizer": {
        "type": "adam",
        "lr": 0.001,
    },
}

# fmt: on
# ==============================================================================
# DEVICE INFORMATION
# ==============================================================================

print("="*70)
print("JAX Multi-GPU Predictive Coding - MNIST")
print("="*70)

n_devices = jax.device_count()
devices = jax.devices()

print(f"\n[Device Information]")
print(f"  Total devices: {n_devices}")
print(f"  Device types: {[d.device_kind for d in devices]}")
print(f"  Device IDs: {[d.id for d in devices]}")

if n_devices == 1:
    print(f"\n  ⚠ Only 1 device available - will fall back to single-GPU training")
    print(f"     To see multi-GPU benefits, run on a machine with multiple GPUs")
else:
    print(f"\n  ✓ Multi-GPU training enabled!")
    print(f"     Expected speedup: ~{n_devices}x (linear scaling)")

# ==============================================================================
# CREATE MODEL
# ==============================================================================

print(f"\n[Model Architecture]")
params, structure = create_pc_graph(config, key, init_std=0.05)
num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

print(f"  Nodes: {len(config['node_list'])}")
print(f"  Edges: {len(config['edge_list'])}")
print(f"  Parameters: {num_params:,}")

# ==============================================================================
# LOAD DATA
# ==============================================================================

print(f"\n[Data Loading]")

def one_hot(labels, num_classes=10):
    return jnp.eye(num_classes)[labels]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Important: Batch size should be divisible by number of devices!
batch_size = 128 * n_devices  # Scale batch size with number of devices
print(f"  Batch size: {batch_size} ({batch_size // n_devices} per device)")

train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
)
test_loader = DataLoader(
    test_data, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
)

# Wrap loaders for one-hot encoding
class OneHotWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset

    def __iter__(self):
        for x, y in self.loader:
            y_onehot = one_hot(y.numpy(), num_classes=10)
            yield x, y_onehot

    def __len__(self):
        return len(self.loader)

train_loader = OneHotWrapper(train_loader)
test_loader = OneHotWrapper(test_loader)

print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ==============================================================================
# TRAIN (Multi-GPU)
# ==============================================================================

print(f"\n[Training Configuration]")
print(f"  Epochs: {train_config['num_epochs']}")
print(f"  Optimizer: {train_config['optimizer']['type']}")
print(f"  Learning rate: {train_config['optimizer']['lr']}")
print(f"  Inference steps: {train_config['T_infer']}")

print(f"\n[Training on {n_devices} device(s)]")
print("  (First batch will be slow due to pmap compilation)\n")

start_time = time.time()
trained_params = train_pcn_multi_gpu(
    params=params,
    structure=structure,
    train_loader=train_loader,
    config=train_config,
    verbose=True,
)
training_time = time.time() - start_time

print(f"\n  Total training time: {training_time:.1f}s")
print(f"  Average time per epoch: {training_time / train_config['num_epochs']:.1f}s")
print(f"  Throughput: {len(train_loader.dataset) * train_config['num_epochs'] / training_time:.0f} samples/sec")

# ==============================================================================
# EVALUATE
# ==============================================================================

print(f"\n[Evaluation]")
metrics = evaluate_pcn_multi_gpu(trained_params, structure, test_loader, train_config)
print(f"  Test Accuracy: {metrics['accuracy'] * 100:.2f}%")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("Multi-GPU Training Complete!")
print("="*70)

if n_devices > 1:
    print(f"\n✓ Successfully trained on {n_devices} GPUs using pmap")
    print(f"✓ Automatic batch sharding across devices")
    print(f"✓ Gradient averaging with pmean")
    print(f"✓ Throughput: {len(train_loader.dataset) * train_config['num_epochs'] / training_time:.0f} samples/sec")

    print(f"\nScaling Efficiency:")
    print(f"  - Linear scaling expected: {n_devices}x speedup")
    print(f"  - Actual throughput scaled by batch size increase")
else:
    print(f"\n✓ Trained on single GPU (multi-GPU code is ready)")
    print(f"  Run on multi-GPU machine to see parallelization benefits")

print("\nKey Takeaways:")
print("  • pmap provides data parallelism with minimal code changes")
print("  • Batch size scales with number of devices")
print("  • Gradients are automatically averaged across devices")
print("  • Nearly linear speedup with multiple GPUs")
print("="*70)
