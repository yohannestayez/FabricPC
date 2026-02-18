from fabricpc.core.types import GraphState


def update_node_in_state(state: GraphState, node_name: str, **updates) -> GraphState:
    """
    Update a node's fields in GraphState.
    Args:
        state: Current GraphState
        node_name: Name of the node to update
        **updates: kv args fields to update in the NodeState
    Returns:
        New GraphState with updated node
    Usage:
    state = update_node_in_state(state, "hidden", latent_grad=grad)
    """
    updated_node = state.nodes[node_name]._replace(**updates)
    return state._replace(nodes={**state.nodes, node_name: updated_node})
