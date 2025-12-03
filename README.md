# FabricPC

**A flexible, performant predictive coding framework**

FabricPC implements predictive coding networks using a clean abstraction of:
- **Nodes**: State variables (latents), projection functions, and activations
- **Wires**: Connections (edges) between nodes in the model architecture
- **Updates**: Iterative inference and learning algorithms

Uses JAX for GPU acceleration and local (node-level) automatic differentiation.

## Quick Start
'''bash'
export PYTHONPATH=$PYTHONPATH:$(pwd)

python fabricpc/examples/mnist_demo.py
'''

## Shape Conventions

 - Linear: shape=(features,) - e.g., (128,) for 128-dimensional vector
 - 1D Conv: shape=(seq_len, channels) - e.g., (100, 32) for 100 timesteps, 32 channels
 - 2D Conv: shape=(H, W, C) - e.g., (28, 28, 64) for 28x28 image, 64 channels (NHWC)
 - 3D Conv: shape=(D, H, W, C) - e.g., (32, 32, 32, 16) for 3D volume

Linear nodes flatten their inputs for transformation and then reshape their outputs to the specified shape.

Conv Node Shape Flow (Future Reference)

 - Input:  (batch, H_in, W_in, C_in)   e.g., (32, 28, 28, 1)
 - Kernel: (kH, kW, C_in, C_out)       e.g., (3, 3, 1, 64)
 - Output: (batch, H_out, W_out, C_out) e.g., (32, 26, 26, 64)