"""
Initialization utilities for JAX predictive coding networks.

Provides structured initialization configurations for weights and states,
matching PyTorch API conventions.
"""

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp


# ==============================================================================
# WEIGHT INITIALIZATION
# ==============================================================================

def initialize_weights(
    config: Dict[str, Any],
    key: jax.Array,  # from jax.random.PRNGKey
    shape: Tuple[int, ...]
) -> jnp.ndarray:
    """
    Initialize weight array based on configuration.

    Args:
        config: Weight initialization configuration dict
        key: JAX random key
        shape: Shape of weight array (fan_in, fan_out)

    Returns:
        Initialized weight array

    Supported types:
        - uniform: Uniform distribution in [min, max]
        - normal: Normal distribution with mean and std
        - xavier: Xavier/Glorot initialization (balanced fan-in/fan-out, for sigmoid and tanh)
        - kaiming: Kaiming/He initialization (fan-in scaled, for ReLU)

    Example:
        >>> config = {"type": "normal", "mean": 0, "std": 0.05}
        >>> W = initialize_weights(config, key, (784, 256))
    """
    if not isinstance(config, dict):
        raise ValueError(f"Weight init config must be a dict, got {type(config)}")

    init_type = config.get("type", "normal").lower()

    if init_type == "uniform":
        min_val = config.get("min", -0.1)
        max_val = config.get("max", 0.1)
        return jax.random.uniform(key, shape, minval=min_val, maxval=max_val)

    elif init_type == "normal":
        mean = config.get("mean", 0.0)
        std = config.get("std", 0.05)
        return mean + std * jax.random.normal(key, shape)

    elif init_type == "xavier":
        # Xavier/Glorot initialization
        # Assumes shape is (fan_in, fan_out)
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]

        distribution = config.get("distribution", "normal")  # 'normal' or 'uniform'

        if distribution == "uniform":
            # Uniform Xavier: U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
            limit = jnp.sqrt(6.0 / (fan_in + fan_out))
            return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
        else:  # normal
            # Normal Xavier: N(0, std^2) where std = sqrt(2 / (fan_in + fan_out))
            std = jnp.sqrt(2.0 / (fan_in + fan_out))
            return std * jax.random.normal(key, shape)

    elif init_type == "kaiming":
        # Kaiming/He initialization: optimized for ReLU networks
        # Assumes shape is (fan_in, fan_out)
        mode = config.get("mode", "fan_in")  # 'fan_in' or 'fan_out'
        nonlinearity = config.get("nonlinearity", "relu")  # 'relu' or 'leaky_relu'
        distribution = config.get("distribution", "normal")  # 'normal' or 'uniform'

        if mode == "fan_out":
            fan = shape[1] if len(shape) > 1 else shape[0]
        else:  # fan_in
            fan = shape[0]

        # Adjust for nonlinearity
        if nonlinearity == "leaky_relu":
            a = config.get("a", 0.01)  # negative slope
            gain = jnp.sqrt(2.0 / (1 + a**2))
        else:  # relu
            gain = jnp.sqrt(2.0)

        if distribution == "uniform":
            # Uniform Kaiming: U(-limit, limit) where limit = gain * sqrt(3 / fan)
            limit = gain * jnp.sqrt(3.0 / fan)
            return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
        else:  # normal
            # Normal Kaiming: N(0, std^2) where std = gain / sqrt(fan)
            std = gain / jnp.sqrt(fan)
            return std * jax.random.normal(key, shape)

    else:
        raise ValueError(
            f"Unknown weight initialization type: '{init_type}'. "
            f"Supported: 'uniform', 'normal', 'xavier', 'kaiming'"
        )


def get_default_weight_init() -> Dict[str, Any]:
    """Get default weight initialization config (normal with std=0.05)."""
    return {"type": "normal", "mean": 0.0, "std": 0.05}


# ==============================================================================
# STATE INITIALIZATION
# ==============================================================================

# TODO define schema for state initialization configs
def initialize_state_values(
    config: Dict[str, Any],
    key: jax.Array,  # from jax.random.PRNGKey
    shape: Tuple[int, ...]
) -> jnp.ndarray:
    """
    Initialize state array based on configuration.

    Args:
        config: State initialization configuration dict
        key: JAX random key
        shape: Shape of state array (batch_size, dim)

    Returns:
        Initialized state array

    Supported types:
        - zeros: All zeros
        - uniform: Uniform distribution in [min, max]
        - normal: Normal distribution with mean and std

    Example:
        >>> config = {"type": "normal", "mean": 0, "std": 0.01}
        >>> z = initialize_state_values(config, key, (32, 256))
    """
    if not isinstance(config, dict):
        raise ValueError(f"State init config must be a dict, got {type(config)}")

    init_type = config.get("type", "normal").lower()

    if init_type == "zeros":
        return jnp.zeros(shape)

    elif init_type == "uniform":
        min_val = config.get("min", -0.1)
        max_val = config.get("max", 0.1)
        return jax.random.uniform(key, shape, minval=min_val, maxval=max_val)

    elif init_type == "normal":
        mean = config.get("mean", 0.0)
        std = config.get("std", 0.05)
        return mean + std * jax.random.normal(key, shape)

    else:
        raise ValueError(
            f"Unknown state initialization type: '{init_type}'. "
            f"Supported: 'zeros', 'uniform', 'normal'"
        )


def parse_state_init_config(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Parse state initialization config.

    Args:
        config: State initialization configuration

    Returns:
        Tuple of (init_method, fallback_config)
        - init_method: "zeros", "uniform", "normal", or "feedforward"
        - fallback_config: Config for fallback initialization (used by feedforward)

    Example:
        >>> config = {"type": "feedforward", "fallback": {"type": "normal", "std": 0.01}}
        >>> method, fallback = parse_state_init_config(config)
        >>> method
        'feedforward'
    """
    if not isinstance(config, dict):
        raise ValueError(f"State init config must be a dict, got {type(config)}")

    init_type = config.get("type", "feedforward").lower()

    if init_type == "zeros":
        return "zeros", {}

    elif init_type == "uniform":
        return "uniform", config

    elif init_type == "normal":
        return "normal", config

    elif init_type == "feedforward":
        fallback = config.get("fallback", {"type": "normal", "mean": 0.0, "std": 0.05})
        return "feedforward", fallback

    else:
        raise ValueError(
            f"Unknown state initialization type: '{init_type}'. "
            f"Supported: 'zeros', 'uniform', 'normal', 'feedforward'"
        )


def get_default_state_init() -> Dict[str, Any]:
    """Get default state initialization config (feedforward with normal fallback)."""
    return {
        "type": "feedforward",
        "fallback": {"type": "normal", "mean": 0.0, "std": 0.05}
    }
