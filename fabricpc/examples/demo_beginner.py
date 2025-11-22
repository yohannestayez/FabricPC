"""
Beginner-Friendly Predictive Coding Network Demo on MNIST
============================================================

This demo shows how EASY it is to build a predictive coding neural network!

Key Concepts:
1. Define NODES (layers) with dimensions and activation functions
2. Connect nodes with EDGES to create the network architecture
3. The network learns by minimizing prediction errors between nodes
4. No need to manually define forward/backward passes - the graph handles it!

The beauty: Just describe WHAT you want, not HOW to implement it.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from fabricpc.models.graph_net import PCGraphNet
from fabricpc.training.PC_trainer import train_pcn, eval_class_accuracy
from fabricpc.evaluation.visualize_experiment import plot_energy_history_interactive
torch.manual_seed(0)

import torch, torchvision
from fabricpc.models.graph_net import PCGraphNet
from fabricpc.training.PC_trainer import train_pcn, eval_class_accuracy


# ============================================================================
# STEP 1: Configure your network as a simple dictionary/JSON
# ============================================================================

# Node, edge, and slot model construction:
# - Each NODE is a block with some dimension (size), contains state vector, and weight matrices to map incoming connections to the node state vector
# - Each EDGE connects blocks together, from a node output to an input slot of a post-synaptic node

network_config = {
    # Hardware settings
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # NODES: Define the layers of your network
    # Each node has: name, dimension (size), type, and activation function
    "node_list": [
        {
            "name": "pixels",           # Input: flattened 28x28 image
            "dim": 784,                 # 28 * 28 = 784 pixels
            "type": "linear",
            "activation": {"type": "sigmoid"}
        },
        {
            "name": "hidden1",          # First hidden layer
            "dim": 256,                 # 256 latent features
            "type": "linear",
            "activation": {"type": "sigmoid"}
        },
        {
            "name": "hidden2",          # Second hidden layer
            "dim": 64,                  # 64 latent features
            "type": "linear",
            "activation": {"type": "sigmoid"}
        },
        {
            "name": "digit",            # Output: which digit (0-9)
            "dim": 10,                  # 10 classes
            "type": "linear",
            "activation": {"type": "sigmoid"}
        }
    ],

    # EDGES: Connect the nodes (defines information flow)
    # Each edge has: source node → target node
    "edge_list": [
        {"source_name": "pixels",  "target_name": "hidden1", "slot": "in"},
        {"source_name": "hidden1", "target_name": "hidden2", "slot": "in"},
        {"source_name": "hidden2", "target_name": "digit",   "slot": "in"}
    ],

    # TASK MAPPING: Tell the network which nodes correspond to inputs/outputs
    "task_map": {
        "x": "pixels",    # Input images go to 'pixels' node
        "y": "digit"      # Target labels go to 'digit' node
    },

    # TRAINING HYPERPARAMETERS
    "infer_steps": 20,                                   # Inference steps (how long to think)
    "eta_infer": 0.05,                               # Inference learning rate
    "optimizer": {
        "type": "adam",
        "lr": 0.001,                                 # Weight learning rate
        "weight_decay": 0.001                        # Regularization
    }
}

# ============================================================================
# STEP 2: Create the model (just one line!)
# ============================================================================

print("=" * 70)
print("Building Predictive Coding Network from configuration...")
print("=" * 70)

model = PCGraphNet(config=network_config)

print(f"\n✓ Network created with {len(model.node_dictionary)} nodes")
print(f"✓ Connected by {len(model.edge_dictionary)} edges")
print(f"✓ Running on: {model.device}")

# Print network structure
print("\nNetwork Architecture:")
print("-" * 70)
for node_name, node in model.node_dictionary.items():
    print(f"  Node '{node_name}': {node.dim} dimensions, {node.in_degree} inputs → {node.out_degree} outputs")

# ============================================================================
# STEP 3: Load MNIST dataset
# ============================================================================

print("\n" + "=" * 70)
print("Loading MNIST dataset...")
print("=" * 70)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

batch_size = 200
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

print(f"✓ Training samples: {len(train_dataset)}")
print(f"✓ Test samples: {len(test_dataset)}")
print(f"✓ Batch size: {batch_size}")

# ============================================================================
# STEP 4: Train the model
# ============================================================================

print("\n" + "=" * 70)
print("Training the network...")
print("=" * 70)

def evaluate_model(model):
    """Evaluate accuracy on test set"""
    accuracy = eval_class_accuracy(model=model, data_loader=test_loader)
    return accuracy, 0  # Return accuracy and dummy energy value

num_epochs = 20

energy_history, metrics = train_pcn(
    model=model,
    data_loader=train_loader,
    num_epochs=num_epochs,
    eval_callback=evaluate_model,
    eval_every_epoch=False,      # Evaluate after each epoch
    measure_train_metrics=False  # Track training metrics
)

# ============================================================================
# STEP 5: Evaluate the trained model
# ============================================================================

# Plot batch energy trajectories
plot_energy_history_interactive(energy_history)

print("\n" + "=" * 70)
print("Final Evaluation")
print("=" * 70)

final_accuracy = eval_class_accuracy(
    model=model,
    data_loader=test_loader,
    analyze_mistakes_func=None
)

print(f"\n🎯 Final Test Accuracy: {final_accuracy:.2f}%")

# ============================================================================
# STEP 6: Show what makes FabricPC special
# ============================================================================

print("\n" + "=" * 70)
print("What makes FabricPC special?")
print("=" * 70)

print("""
FLEXIBLE ARCHITECTURE: Just define nodes and edges
   - Want to add a node? Add it to the node_list
   - Want a different connection? Modify the edge_list
   - The graph handles the rest automatically

To modify this network:
- Change node dimensions in 'node_list' to make layers bigger/smaller
- Add/remove nodes to make the network deeper/shallower
- Change edge connections to create skip connections or different architectures
- Adjust infer_steps to control how long the network 'thinks' during inference
""")

print("\n" + "=" * 70)
print("Experiment complete! Try modifying the config to explore different architectures.")
print("=" * 70)
