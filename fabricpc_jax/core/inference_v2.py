"""
Core inference dynamics for JAX predictive coding networks with local Hebbian learning.

This module implements the functional inference loop that updates latent states
using local gradients computed via Jacobian for true predictive coding.
"""

from typing import Dict, Tuple
import jax
import jax.numpy as jnp

from fabricpc_jax.core.types import GraphParams, GraphState, GraphStructure
from fabricpc_jax.core.activations import get_activation
from fabricpc_jax.nodes import get_node_class_from_type
from fabricpc_jax.core.types import NodeParams, NodeInfo
from fabricpc_jax.utils.helpers import update_node_in_state


def gather_inputs(
        node_info: NodeInfo,
        structure: GraphStructure,
        state: GraphState,
) -> Dict[str, jax.Array]:
    """
    Gather inputs for a node from the graph structure.
    """
    in_edges_data = {}
    for edge_key in node_info.in_edges:
        edge_info = structure.edges[edge_key]  # get the edge object
        node = edge_info.source
        in_edges_data[edge_key] = state.nodes[node].z_latent  # get the data sent along this edge

    return in_edges_data


def compute_node_projection(
    params: GraphParams,
    state: GraphState,
    node_name: str,
    structure: GraphStructure,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute prediction z_mu for a node from its incoming connections.

    Args:
        params: Model parameters organized by node
        state: Current states for all nodes
        node_name: Name of the node to compute prediction for
        structure: Static graph structure

    Returns:
        Tuple of (z_mu, pre_activation, substructure_state)
    """
    node_info = structure.nodes[node_name]
    node_class = get_node_class_from_type(node_info.node_type)

    # Source nodes (no incoming edges) have zero prediction
    if node_info.in_degree == 0:
        batch_size = state.batch_size
        zero_pred = jnp.zeros((batch_size, node_info.dim))
        return zero_pred, zero_pred, {}

    # Get node's parameters
    default_empty = NodeParams(weights={}, biases={})
    node_params = params.nodes.get(node_name, default_empty)

    # Gather inputs for each slot
    in_edges_data = gather_inputs(node_info, structure, state)

    # Forward pass through node
    z_mu, pre_activation, substructure_state = node_class.forward(
        node_params,
        in_edges_data,
        node_info,
        state.nodes[node_name].z_latent.shape
    )
    return z_mu, pre_activation, substructure_state


def compute_latent_gradients(
    state: GraphState,
    params: GraphParams,
    structure: GraphStructure,
) -> GraphState:
    """
    Compute gradient of energy w.r.t. latent states
    Propagate errors back to presynaptic latents

    Args:
        state: Current graph state
        params: Model parameters
        structure: Graph structure

    Returns:
        Updated graph state with latent gradients
    """

    # Zero the latent gradients
    for node_name in structure.nodes:
        # Reset gradients
        node_state = state.nodes[node_name]
        grad = jnp.zeros_like(node_state.z_latent)

        # Replace the latent gradient in state
        state = update_node_in_state(state, node_name, latent_grad=grad)

    # Backpropagate errors to pre-synaptic nodes through Jacobians
    for node_name, node_info in structure.nodes.items():
        node_state = state.nodes[node_name]
        node_class = get_node_class_from_type(node_info.node_type)

        # Collect edge inputs for Jacobian computation
        edge_inputs = gather_inputs(node_info, structure, state)

        # Backpropagate error on edges and to self
        grad_contrib = node_class.compute_gradient(
            params.nodes[node_name], edge_inputs, node_state, node_info, structure
        )
        # Accumulate gradient contributions to this node's sources (including self)
        for source_name, grad in grad_contrib.items():
            latent_grad = state.nodes[source_name].latent_grad
            latent_grad = latent_grad + grad  # Send gradient contribution to source node
            # Update the state with added gradient contribution
            state = update_node_in_state(state, source_name, latent_grad=latent_grad)

    # TODO if using preactivation latents, multiply by activation derivative here
    # if latent_type == "preactivation":
    #     for node_name in structure.nodes:
    #         node_state = state.nodes[node_name]
    #         node_info = structure.nodes[node_name]
    #         _, act_deriv = get_activation(node_info.activation_config)
    #         latent_grad = state.nodes[node_name].latent_grad
    #         latent_grad = latent_grad * act_deriv(node_state.pre_activation)
    #         # Update the state with new latent gradients
    #         state = update_node_in_state(state, node_name, latent_grad)

    return state

def compute_all_projections(
    params: GraphParams,
    state: GraphState,
    structure: GraphStructure,
) -> GraphState:
    """
    Compute predictions for all nodes in the graph.

    Args:
        params: Model parameters
        state: Current graph state
        structure: Graph structure

    Returns:
        Updated graph state with predictions
    """

    # Use node_order for efficient traversal
    for node_name in structure.node_order:
        node_state = state.nodes[node_name]

        # Compute prediction for this node
        z_mu, pre_activation, substructure_state = compute_node_projection(
            params, state, node_name, structure
        )

        # Update the state with new predictions
        state = update_node_in_state(state, node_name, z_mu=z_mu, pre_activation=pre_activation, substructure=substructure_state)

    return state

def compute_errors(
    state: GraphState,
    structure: GraphStructure,
) -> GraphState:
    """
    Compute prediction errors and gain-modulated errors.

    Args:
        state: Current graph state
        structure: Graph structure

    Returns:
        Updated graph state with errors and gain-modulated errors
    """

    for node_name in structure.nodes:
        error = None
        gain_mod_error = None
        energy = None

        node_info = structure.nodes[node_name]
        node_state = state.nodes[node_name]

        # Compute basic error
        error = node_state.z_latent - node_state.z_mu
        energy = jnp.sum(error ** 2)  # TODO call the node energy functional method

        # Compute gain-modulated error
        if node_info.in_degree == 0:
            # Source nodes have no prediction
            gain_mod_error = jnp.zeros_like(error)
        else:
            _, deriv_fn = get_activation(node_info.activation_config)
            gain = deriv_fn(node_state.pre_activation)
            gain_mod_error = error * gain

        # Update the state with new errors
        state = update_node_in_state(state, node_name, error=error, gain_mod_error=gain_mod_error, energy=energy)

    return state

def inference_step(
    params: GraphParams,
    state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    eta_infer: float,
) -> GraphState:
    """
    Single inference step with local gradient computation.

    Args:
        params: Model parameters
        state: Current graph state
        clamps: Dictionary of clamped values
        structure: Graph structure
        eta_infer: Inference learning rate

    Returns:
        Updated graph state
    """
    # 1. Compute predictions for all nodes
    state = compute_all_projections(params, state, structure)

    # 2. Compute errors
    state = compute_errors(state, structure)

    # 3. Compute gradients, local to each node
    state = compute_latent_gradients(state, params, structure)

    # 4. Update latent states by gradient descent
    for node_name in structure.nodes:
        node_state = state.nodes[node_name]
        if node_name in clamps:
            # Keep clamped nodes fixed
            new_z_latent = clamps[node_name]
        else:
            # Gradient descent
            new_z_latent = node_state.z_latent - eta_infer * node_state.latent_grad
        # Update state
        state = update_node_in_state(state, node_name, z_latent=new_z_latent)

    return state

def run_inference(
    params: GraphParams,
    initial_state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    infer_steps: int,
    eta_infer: float = 0.1,
) -> GraphState:
    """
    Run inference for infer_steps steps to converge latent states.

    Args:
        params: Model parameters
        initial_state: Initial graph state
        clamps: Dictionary of clamped values
        structure: Graph structure
        infer_steps: Number of inference steps
        eta_infer: Inference learning rate

    Returns:
        Final converged graph state
    """

    def body_fn(t, state):
        return inference_step(params, state, clamps, structure, eta_infer)

    # Use lax.fori_loop for efficiency
    final_state = jax.lax.fori_loop(0, infer_steps, body_fn, initial_state)

    return final_state