"""
Generative Predictive Coding Network for MNIST
===============================================

This script demonstrates a GENERATOR network using predictive coding.

Key difference from discriminator:
- Information flow: Class label → Hidden layers → Image
- The network learns to GENERATE images from class labels
- Can create new digit images by providing a one-hot class vector

Network Architecture:
    class (10) → hidden2 (128) → hidden1 (512) → image (784)
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from fabricpc.models.graph_net import PCGraphNet
from fabricpc.training.PC_trainer import (
    train_pcn,
    eval_image_energy,
    eval_class_accuracy,
)
from evaluation.visualize_experiment import (
    plot_train_val_metric,
    plot_energy_history_interactive,
)

# ==============================================================================
# SETUP
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(0)

# Training parameters
batch_size = 200
num_epochs = 20
eval_each_epoch = True  # Set to True to track metrics every epoch

# ==============================================================================
# GENERATOR MODEL CONFIGURATION
# ==============================================================================

# Generator: Maps from class label → image
# Information flows BACKWARD compared to discriminator
model_config = {
    "device": str(device),
    "node_list": [
        {
            "name": "class",
            "dim": 10,
            "type": "linear",
            "activation": {"type": "sigmoid"},
        },
        {
            "name": "hidden2",
            "dim": 128,
            "type": "linear",
            "activation": {"type": "sigmoid"},
        },
        {
            "name": "hidden1",
            "dim": 512,
            "type": "linear",
            "activation": {"type": "sigmoid"},
        },
        {
            "name": "image",
            "dim": 784,
            "type": "linear",
            "activation": {"type": "sigmoid"},
        },
    ],
    "edge_list": [
        {"source_name": "class", "target_name": "hidden2", "slot": "in"},
        {"source_name": "hidden2", "target_name": "hidden1", "slot": "in"},
        {"source_name": "hidden1", "target_name": "image", "slot": "in"},
    ],
    "task_map": {"x": "image", "y": "class"},
    "infer_steps": 20,
    "eta_infer": 0.05,
    "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0001},
}

# Create model
pcn_model = PCGraphNet(config=model_config)

print("\nGenerator Network Architecture:")
print("  class (10) → hidden2 (128) → hidden1 (512) → image (784)")
print("  Information flow: Label → Image (generative)")

# ==============================================================================
# DATA LOADING
# ==============================================================================

# No normalization for generator (raw pixel values)
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

trainloader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
    pin_memory=False,
    prefetch_factor=2,
)
testloader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=16,
    pin_memory=False,
    prefetch_factor=2,
)

print(f"\nDataset Info:")
print(f"  Batch size: {batch_size}")
print(f"  Train batches: {len(trainloader)}")
print(f"  Test batches: {len(testloader)}")

# ==============================================================================
# EVALUATION CALLBACK
# ==============================================================================

def eval_callback(model):
    """Evaluate both accuracy and image generation quality"""
    val_accuracy = eval_class_accuracy(model=model, data_loader=testloader)
    val_image_energy = eval_image_energy(model=model, data_loader=testloader)
    return val_accuracy, val_image_energy

# ==============================================================================
# TRAINING
# ==============================================================================

print("\n" + "="*70)
print("Starting Generator Training...")
print("="*70)

energy_history, eval_metrics = train_pcn(
    model=pcn_model,
    data_loader=trainloader,
    num_epochs=num_epochs,
    eval_callback=eval_callback,
    eval_every_epoch=eval_each_epoch,
    measure_train_metrics=eval_each_epoch,
)

print("\nTraining finished.")

# ==============================================================================
# FINAL EVALUATION
# ==============================================================================

print("\n" + "="*70)
print("Final Evaluation")
print("="*70)

# Evaluate classification accuracy
acc = eval_class_accuracy(
    model=pcn_model,
    data_loader=testloader,
    analyze_mistakes_func=None,
    plot_generated=True,  # Visualize generated images
    image_size=(1, 28, 28),
)
print(f"\nTest Top-1 Accuracy: {acc:.2f}%")

# Evaluate image generation quality
gen_image_energy = eval_image_energy(
    model=pcn_model,
    data_loader=testloader,
    analyze_mistakes_func=None,
    image_size=(1, 28, 28),
)
print(f"Test Image Energy: {gen_image_energy:.2f}")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

print("\n" + "="*70)
print("Generating Visualizations...")
print("="*70)

# Plot batch energy trajectories
plot_energy_history_interactive(energy_history)

# Plot epoch training curves (if enabled)
if eval_each_epoch:
    plot_train_val_metric(
        train_result=eval_metrics["train_image_energy"],
        val_result=eval_metrics["val_image_energy"],
        yaxis_title="Image Energy",
    )
    plot_train_val_metric(
        train_result=eval_metrics["train_accuracy"],
        val_result=eval_metrics["val_accuracy"],
        yaxis_title="Accuracy (%)",
    )

# ==============================================================================
# NOTES ON GENERATOR NETWORKS
# ==============================================================================

print("\n" + "="*70)
print("About Generator Networks")
print("="*70)
print("""
Generator networks learn to CREATE images from class labels:

1. GENERATIVE DIRECTION:
   - Input: One-hot class label (e.g., [0,0,1,0,0,0,0,0,0,0] for digit "2")
   - Output: 28x28 image of that digit

2. KEY METRICS:
   - Classification Accuracy: Can it recognize which digit is shown?
   - Image Energy: How well does it generate realistic images?

To generate a specific digit:
  1. Clamp the class node to desired label
  2. Run inference
  3. Read the generated image from the image node
""")

print("\n" + "="*70)
print("Experiment Complete!")
print("="*70)
