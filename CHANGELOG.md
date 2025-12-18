# Changelog

## [0.2.3] - 2025-12-18
- Change Linear node default behavior to perform matmul on the last tensor dimension. Flattening inputs now requires flag `flatten_input=True`.
- Removed gain_mod_error from NodeState, as it was not used by anything other than explicit grad linear node.
- Added softmax and Gelu activation functions.
- Added KL Divergence energy functional.

## [0.2.2] - 2025-12-05
- Unified config validation and registry pattern across nodes, energy functionals, and activations
- Custom objects now follow a consistent extensibility pattern with `CONFIG_SCHEMA` and `@register_*` decorators
- Node construction delegated to `NodeBase.from_config()` for cleaner separation of concerns
- CONFIG_SCHEMA is now a required class variable for easier access and introspection

## [0.2.1] - 2025-12-04
- Node autograd is the default behavior now; can override by subclassing a node and implementing manual gradients
- N-dimensional tensor support: breaking changes to shape conventions
  - Linear nodes: shape=(features,) e.g., (128,) for 128-dimensional vector
  - 2D Conv nodes: shape=(H, W, C) e.g., (28, 28, 64) for 28x28 image with 64 channels (NHWC)
- Plugin architecture for custom nodes with two choices for registration: decorator or setuptools entry points