"""
Graph-based predictive coding network construction for JAX with local Hebbian learning.

This module provides functions to build graph structures with node classes,
validate slot connections, and initialize parameters at the node level.

Edges direct the flow of information through the graph in inference and training.
- Node output is passed to input slots of post-synaptic nodes
- Node gets gradient contributions from post-synaptic nodes by querying its out_neighbors on each outgoing edge for the gradient contributions of that particular edge.

Config schemas are defined at the appropriate level:
- GRAPH_CONFIG_SCHEMA: Validates graph structure (node_list, edge_list, task_map)
- EDGE_CONFIG_SCHEMA: Validates edge fields (source_name, target_name, slot)
- Node-specific schemas: Defined in node classes (e.g., LinearNode.CONFIG_SCHEMA)
- Energy/Activation schemas: Defined in their respective classes and validated via delegation
"""

from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from fabricpc.core.types import (
    NodeParams,
    GraphParams,
    NodeState,
    GraphState,
    GraphStructure,
)
from fabricpc.nodes import get_node_class
from fabricpc.utils.helpers import update_node_in_state
from fabricpc.core.inference import gather_inputs


# TODO deprecated
def build_graph_structure(config: dict) -> GraphStructure:
    """
    Convert configuration dictionary to static GraphStructure with slot validation.

    This is a convenience function that delegates to GraphStructure.from_config().

    Args:
        config: Configuration dictionary with node_list, edge_list, task_map

    Returns:
        Immutable GraphStructure with validated slots

    Raises:
        ValueError: If graph is misspecified
        ConfigValidationError: If config validation fails
    """
    return GraphStructure.from_config(config)


def compute_local_weight_gradients(
    params: GraphParams,
    final_state: GraphState,
    structure: GraphStructure,
) -> GraphParams:
    """
    Compute local weight gradients for each node using its own error signal.

    This implements the local Hebbian learning rule for predictive coding:

    Args:
        params: Current model parameters
        final_state: Converged state after inference
        structure: Graph structure

    Returns:
        GraphParams containing gradients for the parameters
    """
    gradients = {}

    for node_name, node_info in structure.nodes.items():
        # Source nodes have no weights, but need empty gradient dict for consistency
        if node_info.in_degree == 0:
            gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        in_edges_data = gather_inputs(node_info, structure, final_state)

        # Get node class
        node_class = get_node_class(node_info.node_type)

        # Compute local gradients using node's method
        node_state, grad_params = node_class.forward_learning(
            params.nodes[node_name],
            in_edges_data,
            final_state.nodes[node_name],
            node_info,
        )

        # Store gradients
        gradients[node_name] = grad_params

    # convert to GraphParams
    params_gradients = GraphParams(nodes=gradients)

    return params_gradients


# TODO create abstraction and config schema for param initialization, similar to graph state initialization
def initialize_params(
    structure: GraphStructure,
    rng_key: jax.Array,  # from jax.random.PRNGKey
) -> GraphParams:
    """
    Initialize model parameters at the node level.

    Each node class handles its own parameter initialization,
    supporting complex nodes with multiple internal parameters.

    Args:
        structure: Graph structure
        rng_key: JAX random key

    Returns:
        GraphParams with node-based parameter organization
    """
    node_params = {}  # type: Dict[str, NodeParams]

    # Split key for each node
    num_nodes = len([n for n in structure.nodes.values() if n.in_degree > 0])
    if num_nodes > 0:
        keys = jax.random.split(rng_key, num_nodes)
    else:
        keys = []
    key_idx = 0

    for node_name, node_info in structure.nodes.items():
        # Skip source nodes (no parameters)
        if node_info.in_degree == 0:
            node_params[node_name] = NodeParams(weights={}, biases={})
            continue

        # Get node class
        node_class = get_node_class(node_info.node_type)

        # Get the input shapes for each edge (full shapes for conv support)
        input_shapes = {}
        for edge_key in node_info.in_edges:
            edge_info = structure.edges[edge_key]
            source_node = structure.nodes[edge_info.source]
            input_shapes[edge_key] = source_node.shape

        # Initialize parameters of the node
        params_obj = node_class.initialize_params(
            keys[key_idx], node_info.shape, input_shapes, node_info.node_config
        )
        key_idx += 1
        node_params[node_name] = params_obj

    return GraphParams(nodes=node_params)


def set_latents_to_clamps(
    state: GraphState,
    clamps: Dict[str, jnp.ndarray],
) -> GraphState:
    """
    Set the latent states of specified nodes to their clamped values.

    Args:
        state: Current graph state
        clamps: Dictionary of clamped values, keyed on node names

    Returns:
        Updated GraphState with latents set to clamped values
    """
    for node_name, clamp_value in clamps.items():
        if node_name in state.nodes:
            state = update_node_in_state(state, node_name, z_latent=clamp_value)
    return state


def create_pc_graph(
    config: dict,
    rng_key: jax.Array,  # from jax.random.PRNGKey
) -> Tuple[GraphParams, GraphStructure]:
    """
    Create a complete PC graph with local Hebbian learning.

    This is the main entry point for creating a JAX PC model with the new architecture.

    Args:
        config: Configuration dictionary with node_list, edge_list, task_map
        rng_key: JAX random key for initialization

    Returns:
        Tuple of (params, structure)

    Example:
        >>> config = {
        ...     "node_list": [
        ...         {
        ...             "name": "pixels",
        ...             "shape": (784,),
        ...             "type": "linear",
        ...             "activation": {"type": "identity"},
        ...             "weight_init": {"type": "xavier"}
        ...         },
        ...         {
        ...             "name": "hidden",
        ...             "shape": (256,),
        ...             "type": "linear",
        ...             "activation": {"type": "relu"}
        ...         },
        ...         {
        ...             "name": "class",
        ...             "shape": (10,),
        ...             "type": "linear",
        ...             "activation": {"type": "softmax"}
        ...         },
        ...     ],
        ...     "edge_list": [
        ...         {"source_name": "pixels", "target_name": "hidden", "slot": "in"},
        ...         {"source_name": "hidden", "target_name": "class", "slot": "in"},
        ...     ],
        ...     "task_map": {"x": "pixels", "y": "class"}
        ... }
        >>> params, structure = create_pc_graph(config, jax.random.PRNGKey(0))
    """
    structure = build_graph_structure(config)
    params = initialize_params(structure, rng_key)
    return params, structure
