"""
Exploring Different Network Architectures with Predictive Coding
==================================================================

This demo shows how EASY it is to experiment with different architectures
by simply changing the configuration dictionary. No code changes needed!

We'll try three different architectures:
1. Shallow network (2 layers)
2. Deep network (5 layers)
3. Wide network (large hidden dimensions)
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from fabricpc.models.graph_net import PCGraphNet
from fabricpc.training.PC_trainer import train_pcn, eval_class_accuracy
from fabricpc.evaluation.visualize_experiment import plot_energy_history_interactive, plot_train_val_metric
torch.manual_seed(0)

# ==============================================================================
# ARCHITECTURE LIBRARY: Just copy-paste configs!
# ==============================================================================
# fmt: off

ARCHITECTURES = {
    "shallow": {
        "description": "Shallow network: input → hidden → output",
        "node_list": [
            {"name": "input", "dim": 784, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "hidden", "dim": 64, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "output", "dim": 10, "type": "linear", "activation": {"type": "sigmoid"}},
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "hidden", "slot": "in"},
            {"source_name": "hidden", "target_name": "output", "slot": "in"},
        ],
    },

    "deep": {
        "description": "Deep network: 5 layers of hierarchical processing",
        "node_list": [
            {"name": "input", "dim": 784, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "h1", "dim": 512, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "h2", "dim": 256, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "h3", "dim": 128, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "h4", "dim": 64, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "output", "dim": 10, "type": "linear", "activation": {"type": "sigmoid"}},
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "h1", "slot": "in"},
            {"source_name": "h1", "target_name": "h2", "slot": "in"},
            {"source_name": "h2", "target_name": "h3", "slot": "in"},
            {"source_name": "h3", "target_name": "h4", "slot": "in"},
            {"source_name": "h4", "target_name": "output", "slot": "in"},
        ],
    },

    "wide": {
        "description": "Wide network: Large hidden representations",
        "node_list": [
            {"name": "input", "dim": 784, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "h1", "dim": 1024, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "h2", "dim": 512, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "output", "dim": 10, "type": "linear", "activation": {"type": "sigmoid"}},
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "h1", "slot": "in"},
            {"source_name": "h1", "target_name": "h2", "slot": "in"},
            {"source_name": "h2", "target_name": "output", "slot": "in"},
        ],
    },

    "bottleneck": {
        "description": "Bottleneck network: Compress then expand",
        "node_list": [
            {"name": "input", "dim": 784, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "compress", "dim": 128, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "bottleneck", "dim": 32, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "expand", "dim": 128, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "output", "dim": 10, "type": "linear", "activation": {"type": "sigmoid"}},
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "compress", "slot": "in"},
            {"source_name": "compress", "target_name": "bottleneck", "slot": "in"},
            {"source_name": "bottleneck", "target_name": "expand", "slot": "in"},
            {"source_name": "expand", "target_name": "output", "slot": "in"},
        ],
    },
}

# fmt: on
# ==============================================================================
# CHOOSE ARCHITECTURE HERE
# ==============================================================================

SELECTED_ARCH = "shallow"  # Try: "shallow", "deep", "wide", or "bottleneck"

# ==============================================================================
# BUILD MODEL FROM SELECTED ARCHITECTURE
# ==============================================================================

# Get the architecture config
arch_config = ARCHITECTURES[SELECTED_ARCH].copy()
description = arch_config.pop("description")

# Add common settings
config = {
    **arch_config,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "task_map": {"x": "input", "y": "output"},
    "infer_steps": 20,
    "eta_infer": 0.05,
    "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.001},
}

# Create model
model = PCGraphNet(config=config)

print("=" * 70)
print(f"ARCHITECTURE: {SELECTED_ARCH.upper()}")
print("=" * 70)
print(f"Description: {description}")
print(f"Total nodes: {len(config['node_list'])}")
print(f"Total edges: {len(config['edge_list'])}")
print(f"Device: {config['device']}")

# Print architecture details
print("\nNetwork structure:")
for node_cfg in config['node_list']:
    print(f"  • {node_cfg['name']:12s} : {node_cfg['dim']:4d} dims")

print("\nConnections:")
for edge_cfg in config['edge_list']:
    print(f"  • {edge_cfg['source_name']:12s} → {edge_cfg['target_name']:12s}")

# ==============================================================================
# LOAD DATA
# ==============================================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=200, shuffle=True, num_workers=16)
test_loader = DataLoader(test_data, batch_size=200, shuffle=False, num_workers=16)

# ==============================================================================
# TRAIN
# ==============================================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

num_epochs = 8

energy_history, metrics = train_pcn(
    model=model,
    data_loader=train_loader,
    num_epochs=num_epochs,
    eval_callback=lambda m: (eval_class_accuracy(m, test_loader), 0),
    eval_every_epoch=True,
    measure_train_metrics=True
)

# ==============================================================================
# EVALUATE
# ==============================================================================

# Plot batch energy trajectories
plot_energy_history_interactive(energy_history)

# Plot epoch training curves
plot_train_val_metric(
    train_result=metrics["train_accuracy"],
    val_result=metrics["val_accuracy"],
    yaxis_title="Accuracy",
)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

final_accuracy = eval_class_accuracy(model, test_loader)
print(f"\nArchitecture: {SELECTED_ARCH}")
print(f"Test Accuracy: {final_accuracy:.2f}%")

# ==============================================================================
# COMPARISON GUIDE
# ==============================================================================

print("\n" + "=" * 70)
print("TRY DIFFERENT ARCHITECTURES!")
print("=" * 70)
print("""
Change SELECTED_ARCH to try different networks:

• "shallow"    - Fast training, may underfit
• "deep"       - More capacity, hierarchical features
• "wide"       - Large hidden layers, more parameters
• "bottleneck" - Forces compact representations

Just change one line of code and re-run!

Want to create your own architecture?
1. Copy one of the configs above
2. Modify node_list (add/remove/resize nodes)
3. Modify edge_list (change connections)

Example: Add a skip connection
  {"source_name": "input", "target_name": "h2", "slot": "in"}

The framework handles everything else automatically!
""")
