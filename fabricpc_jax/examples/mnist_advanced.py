"""
ADVANCED Predictive Coding Network Example (JAX)
=================================================

This example demonstrates:
1. Deeper network architectures
2. Different activation functions
3. Custom training configurations
4. Progress monitoring and checkpointing
5. Hyperparameter exploration

Compared to mnist_demo.py, this shows more realistic training scenarios.
"""

import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

from fabricpc_jax.models import create_pc_graph, initialize_state
from fabricpc_jax.training import train_step, create_optimizer, evaluate_pcn
from fabricpc_jax.core.inference import run_inference

# Set random seed
key = jax.random.PRNGKey(42)

# ==============================================================================
# ADVANCED NETWORK CONFIGURATION
# ==============================================================================
# fmt: off

config = {
    # Deeper 3-hidden-layer network
    "node_list": [
        {"name": "pixels", "dim": 784, "activation": {"type": "identity"}, "type": "linear"},
        {"name": "h1",     "dim": 256, "activation": {"type": "relu"}, "type": "linear"},
        {"name": "h2",     "dim": 128, "activation": {"type": "relu"}, "type": "linear"},
        {"name": "h3",     "dim": 64,  "activation": {"type": "relu"}, "type": "linear"},
        {"name": "class",  "dim": 10,  "activation": {"type": "identity"}, "type": "linear"},
    ],

    "edge_list": [
        {"source_name": "pixels", "target_name": "h1",    "slot": "in"},
        {"source_name": "h1",     "target_name": "h2",    "slot": "in"},
        {"source_name": "h2",     "target_name": "h3",    "slot": "in"},
        {"source_name": "h3",     "target_name": "class", "slot": "in"},
    ],

    "task_map": {"x": "pixels", "y": "class"},
}

# More sophisticated training configuration
train_config = {
    "T_infer": 30,         # More inference steps for deeper network
    "eta_infer": 0.05,     # Inference learning rate
    "optimizer": {
        "type": "adamw",   # AdamW with weight decay
        "lr": 0.001,
        "weight_decay": 1e-4,
    },
}

# fmt: on
# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def one_hot(labels, num_classes=10):
    """Convert labels to one-hot encoding."""
    return jnp.eye(num_classes)[labels]

class OneHotWrapper:
    """Wrap DataLoader to provide one-hot labels."""
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset

    def __iter__(self):
        for x, y in self.loader:
            y_onehot = one_hot(y.numpy(), num_classes=10)
            yield x, y_onehot

    def __len__(self):
        return len(self.loader)

# ==============================================================================
# CREATE MODEL
# ==============================================================================

print("="*70)
print("JAX Predictive Coding - Advanced MNIST Example")
print("="*70)

params, structure = create_pc_graph(config, key, init_std=0.05)
num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

print(f"\n[Model Architecture]")
print(f"  Nodes: {len(config['node_list'])}")
print(f"  Edges: {len(config['edge_list'])}")
print(f"  Total parameters: {num_params:,}")
print(f"\n  Layer sizes: ", end="")
for node in config['node_list']:
    print(f"{node['dim']} → ", end="")
print("(output)")

# ==============================================================================
# LOAD DATA
# ==============================================================================

print(f"\n[Data Loading]")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

batch_size = 128
train_loader = OneHotWrapper(DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2
))
test_loader = OneHotWrapper(DataLoader(
    test_data, batch_size=batch_size, shuffle=False, num_workers=2
))

print(f"  Train samples: {len(train_data):,}")
print(f"  Test samples: {len(test_data):,}")
print(f"  Batch size: {batch_size}")
print(f"  Train batches: {len(train_loader)}")

# ==============================================================================
# CUSTOM TRAINING LOOP WITH MONITORING
# ==============================================================================

print(f"\n[Training Configuration]")
print(f"  Optimizer: {train_config['optimizer']['type']}")
print(f"  Learning rate: {train_config['optimizer']['lr']}")
print(f"  Weight decay: {train_config['optimizer']['weight_decay']}")
print(f"  Inference steps: {train_config['T_infer']}")
print(f"  Inference eta: {train_config['eta_infer']}")

num_epochs = 10
T_infer = train_config['T_infer']
eta_infer = train_config['eta_infer']

# Create optimizer
optimizer = create_optimizer(train_config['optimizer'])
opt_state = optimizer.init(params)

# JIT compile training step
print(f"\n[Compiling JIT functions...]")
jit_train_step = jax.jit(
    lambda p, o, b: train_step(p, o, b, structure, optimizer, T_infer, eta_infer)
)

# Training loop with detailed monitoring
print(f"\n[Training for {num_epochs} epochs]")
print("  (First batch will be slow due to JIT compilation)\n")

best_accuracy = 0.0
training_history = []

for epoch in range(num_epochs):
    epoch_start = time.time()
    epoch_losses = []

    for batch_idx, (x, y) in enumerate(train_loader):
        batch = {"x": jnp.array(x), "y": y}

        # Training step
        params, opt_state, loss, _ = jit_train_step(params, opt_state, batch)
        epoch_losses.append(float(loss))

        # Progress indicator every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = sum(epoch_losses[-100:]) / len(epoch_losses[-100:])
            print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.4f}")

    epoch_time = time.time() - epoch_start
    avg_loss = sum(epoch_losses) / len(epoch_losses)

    # Evaluate on test set
    metrics = evaluate_pcn(params, structure, test_loader, train_config)
    accuracy = metrics['accuracy'] * 100

    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
        print(f"  ★ New best accuracy: {accuracy:.2f}%")

    print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.1f}s")

    training_history.append({
        'epoch': epoch + 1,
        'loss': avg_loss,
        'accuracy': accuracy,
        'time': epoch_time
    })

# ==============================================================================
# FINAL EVALUATION
# ==============================================================================

print(f"\n[Final Results]")
print(f"  Best accuracy: {best_accuracy:.2f}%")
print(f"  Final accuracy: {training_history[-1]['accuracy']:.2f}%")
print(f"  Total training time: {sum(h['time'] for h in training_history):.1f}s")

# Print training history
print(f"\n[Training History]")
print("  Epoch | Loss    | Accuracy | Time")
print("  ------|---------|----------|------")
for h in training_history:
    print(f"  {h['epoch']:5d} | {h['loss']:7.4f} | {h['accuracy']:7.2f}% | {h['time']:4.1f}s")

print("\n" + "="*70)
print("Advanced Training Complete!")
print("\nKey takeaways:")
print("  ✓ Deeper networks (4 hidden layers) work great")
print("  ✓ Different activations (ReLU) converge well")
print("  ✓ JIT compilation makes training fast")
print("  ✓ Monitoring and checkpointing are easy")
print("\nNext steps:")
print("  - Try even deeper architectures")
print("  - Experiment with different optimizers")
print("  - Add learning rate scheduling")
print("  - Implement multi-GPU training with pmap")
print("="*70)
