"""Core JAX predictive coding components."""

from fabricpc_jax.core.types import GraphParams, GraphState, GraphStructure, NodeInfo, EdgeInfo
from fabricpc_jax.core.activations import get_activation, sigmoid, relu, tanh, identity
from fabricpc_jax.core.inference import (
    compute_projection,
    compute_latent_gradients,
    inference_step,
    run_inference,
)

__all__ = [
    "GraphParams",
    "GraphState",
    "GraphStructure",
    "NodeInfo",
    "EdgeInfo",
    "get_activation",
    "sigmoid",
    "relu",
    "tanh",
    "identity",
    "compute_projection",
    "compute_latent_gradients",
    "inference_step",
    "run_inference",
]
