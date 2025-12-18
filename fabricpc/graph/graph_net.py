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

from typing import Dict, Tuple, Any
import jax
import jax.numpy as jnp
from fabricpc.core.types import (
    NodeParams,
    GraphParams,
    NodeState,
    GraphState,
    GraphStructure,
)
from fabricpc.core.initialization import (
    initialize_state_values,
    parse_state_init_config,
    get_default_state_init,
)
from fabricpc.nodes import get_node_class
from fabricpc.core.inference import gather_inputs
from fabricpc.utils.helpers import update_node_in_state
from fabricpc.core.config import ConfigValidationError


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
            keys[key_idx],
            node_info.shape,
            input_shapes,
            node_info.node_config
        )
        key_idx += 1
        node_params[node_name] = params_obj

    return GraphParams(nodes=node_params)


def initialize_state(
    structure: GraphStructure,
    batch_size: int,
    rng_key: jax.Array,
    clamps: Dict[str, jnp.ndarray] = None,
    state_init_config: Dict[str, Any] = None,
    params: GraphParams = None,
) -> GraphState:
    """
    Initialize graph state for inference.

    Args:
        structure: Graph structure
        batch_size: Batch size
        rng_key: JAX random key for state initialization
        clamps: Optional dictionary of clamped values, keyed on node names
        state_init_config: State initialization configuration dict
        params: GraphParams (required for feedforward init)

    Returns:
        Initial GraphState with latent gradients
    """
    clamps = clamps or {}

    # Use default if not provided
    if state_init_config is None:
        state_init_config = get_default_state_init()

    # Parse initialization config
    init_method, fallback_config = parse_state_init_config(state_init_config)

    # Split rng_key for each node
    node_names = list(structure.nodes.keys())
    node_keys = jax.random.split(rng_key, len(node_names))
    node_key_map = dict(zip(node_names, node_keys))
    node_state_dict = {}

    # Initialize all nodes, respecting clamps
    for node_name, node_info in structure.nodes.items():
        shape = (batch_size, *node_info.shape)

        # Initialize z_latent
        if node_name in clamps:
            # Use clamped value
            z_latent = clamps[node_name]
        elif init_method == "zeros":
            z_latent = jnp.zeros(shape)
        elif init_method in ["uniform", "normal"]:
            # Direct initialization with split key
            z_latent = initialize_state_values(
                fallback_config, node_key_map[node_name], shape
            )
        elif init_method == "feedforward":
            # Initialize with fallback first using split key
            z_latent = initialize_state_values(
                fallback_config, node_key_map[node_name], shape
            )
        else:
            raise ConfigValidationError(f"unknown init_method: {init_method}")

        # Initialize latents, set other state components to zeros
        node_state_dict[node_name] = NodeState(
            z_latent=z_latent,  # init latent state
            z_mu=jnp.zeros(shape),
            error=jnp.zeros(shape),
            energy=jnp.zeros((batch_size,)),  # Per-sample energy
            pre_activation=jnp.zeros(shape),
            latent_grad=jnp.zeros(shape),
            substructure={},
        )

    # Create a draft graph state
    state = GraphState(
        nodes=node_state_dict,
        batch_size=batch_size,
    )
    # Feedforward initialization if requested
    if init_method == "feedforward" and params is not None:

        # Process nodes in topological order
        for node_name in structure.node_order:
            node_info = structure.nodes[node_name]
            if node_info.in_degree > 0:

                # Collect edge inputs
                node_state = state.nodes[node_name]
                node_params = params.nodes[node_name]
                node_class = get_node_class(node_info.node_type)
                edge_inputs = gather_inputs(node_info, structure, state)

                # Compute forward projection
                _, node_state = node_class.forward(node_params, edge_inputs, node_state, node_info)

                # Update the state with feedforward values
                if node_name not in clamps:
                    # z_latent <- z_mu_init
                    node_state = node_state._replace(z_latent=node_state.z_mu)
                else:
                    # Respect clamped values
                    node_state = node_state._replace(z_latent=clamps[node_name])
                # Update state
                state = state._replace(nodes={**state.nodes, node_name: node_state})

    return state

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