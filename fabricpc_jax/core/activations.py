"""
Activation functions for predictive coding networks in JAX.

All functions are pure and compatible with JAX transformations (jit, vmap, grad).
"""

from typing import Callable, Tuple, Dict, Any
import jax.numpy as jnp
from jax import nn


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid activation: σ(x) = 1 / (1 + exp(-x))"""
    return nn.sigmoid(x)


def sigmoid_deriv(x: jnp.ndarray) -> jnp.ndarray:
    """Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))"""
    s = nn.sigmoid(x)
    return s * (1 - s)


def relu(x: jnp.ndarray) -> jnp.ndarray:
    """ReLU activation: max(0, x)"""
    return nn.relu(x)


def relu_deriv(x: jnp.ndarray) -> jnp.ndarray:
    """Derivative of ReLU: 1 if x > 0, else 0"""
    return (x > 0).astype(jnp.float32)


def tanh(x: jnp.ndarray) -> jnp.ndarray:
    """Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    return jnp.tanh(x)


def tanh_deriv(x: jnp.ndarray) -> jnp.ndarray:
    """Derivative of tanh: 1 - tanh²(x)"""
    t = jnp.tanh(x)
    return 1 - t**2


def identity(x: jnp.ndarray) -> jnp.ndarray:
    """Identity activation: f(x) = x"""
    return x


def identity_deriv(x: jnp.ndarray) -> jnp.ndarray:
    """Derivative of identity: always 1"""
    return jnp.ones_like(x)


def leaky_relu(x: jnp.ndarray, alpha: float = 0.01) -> jnp.ndarray:
    """Leaky ReLU: max(alpha * x, x)"""
    return jnp.where(x > 0, x, alpha * x)


def leaky_relu_deriv(x: jnp.ndarray, alpha: float = 0.01) -> jnp.ndarray:
    """Derivative of Leaky ReLU"""
    return jnp.where(x > 0, 1.0, alpha)


def hard_tanh(x: jnp.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> jnp.ndarray:
    """Hard tanh: clip(x, min_val, max_val)"""
    return jnp.clip(x, min_val, max_val)


def hard_tanh_deriv(x: jnp.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> jnp.ndarray:
    """Derivative of hard tanh: 1 if min_val < x < max_val, else 0"""
    return ((x > min_val) & (x < max_val)).astype(jnp.float32)


# Activation function registry
# Matches PyTorch activation names: identity, sigmoid, tanh, relu, leaky_relu, hard_tanh
ACTIVATIONS: Dict[str, Tuple[Callable, Callable]] = {
    "identity": (identity, identity_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
    "relu": (relu, relu_deriv),
    "leaky_relu": (leaky_relu, leaky_relu_deriv),
    "hard_tanh": (hard_tanh, hard_tanh_deriv),
}


def get_activation(config: Dict[str, Any]) -> Tuple[Callable, Callable]:
    """
    Get activation function and its derivative from config.

    Args:
        config: Dictionary with 'type' key and optional parameters.
                Supported types: 'identity', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'hard_tanh'
                - leaky_relu: requires 'alpha' parameter (default 0.01)
                - hard_tanh: requires 'min_val' and 'max_val' parameters (default -1 and 1)

    Returns:
        Tuple of (activation_fn, derivative_fn)

    Example:
        >>> config = {"type": "sigmoid"}
        >>> act_fn, deriv_fn = get_activation(config)
        >>> x = jnp.array([0.0, 1.0, -1.0])
        >>> act_fn(x)
        DeviceArray([0.5, 0.731, 0.269], dtype=float32)
    """
    if "type" not in config:
        raise ValueError("config['type'] is required")

    act_type = config["type"].lower()

    if act_type not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation type: '{act_type}'. "
            f"Supported: {list(ACTIVATIONS.keys())}"
        )

    base_fn, base_deriv = ACTIVATIONS[act_type]

    # Handle parameterized activations
    if act_type == "leaky_relu":
        alpha = config.get("alpha", 0.01)
        return (lambda x: base_fn(x, alpha), lambda x: base_deriv(x, alpha))
    elif act_type == "hard_tanh":
        min_val = config.get("min_val", -1.0)
        max_val = config.get("max_val", 1.0)
        return (lambda x: base_fn(x, min_val, max_val), lambda x: base_deriv(x, min_val, max_val))
    else:
        return (base_fn, base_deriv)


def get_activation_fn(config: Dict[str, Any]) -> Callable:
    """Get just the activation function (without derivative)."""
    return get_activation(config)[0]


def get_activation_deriv(config: Dict[str, Any]) -> Callable:
    """Get just the activation derivative function."""
    return get_activation(config)[1]
