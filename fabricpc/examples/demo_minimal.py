"""
MINIMAL Predictive Coding Network Example
==========================================

This is the absolute SIMPLEST example showing how to:
1. Define a network with a dictionary
2. Train it on MNIST
3. Get results

Total code: ~50 lines. That's it!
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from fabricpc.models.graph_net import PCGraphNet
from fabricpc.training.PC_trainer import train_pcn, eval_class_accuracy
torch.manual_seed(0)

# ==============================================================================
# NETWORK DEFINITION: A dictionary!
# ==============================================================================
# fmt: off

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Define nodes (layers)
    "node_list": [
        {"name": "input",  "dim": 784, "type": "linear", "activation": {"type": "sigmoid"}},
        {"name": "hidden", "dim": 128, "type": "linear", "activation": {"type": "sigmoid"}},
        {"name": "output", "dim": 10,  "type": "linear", "activation": {"type": "sigmoid"}},
    ],

    # Connect nodes with edges
    "edge_list": [
        {"source_name": "input",  "target_name": "hidden", "slot": "in"},
        {"source_name": "hidden", "target_name": "output", "slot": "in"},
    ],

    # Map tasks to nodes
    "task_map": {"x": "input", "y": "output"},

    # Hyperparameters
    "infer_steps": 20,
    "eta_infer": 0.05,
    "optimizer": {"type": "adam", "lr": 0.001},
}

# fmt: on
# ==============================================================================
# CREATE MODEL: One line!
# ==============================================================================

model = PCGraphNet(config=config)
print(f"Model created: {len(config['node_list'])} nodes, {len(config['edge_list'])} edges")

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

print("\nTraining...")
energy_history, _ = train_pcn(
    model=model,
    data_loader=train_loader,
    num_epochs=5,
)

# ==============================================================================
# EVALUATE
# ==============================================================================

accuracy = eval_class_accuracy(model, test_loader)
print(f"\nFinal Accuracy: {accuracy:.2f}%")

print("\n" + "="*70)
print("That's it! Want to change the architecture?")
print("Just modify the config dictionary above:")
print("  - Add more nodes to node_list for deeper networks")
print("  - Change 'dim' values to make layers wider/narrower")
print("  - Modify edge_list to create different connection patterns")
print("  - No need to change any other code!")
print("="*70)
