"""
Node types for JAX predictive coding networks.

This module provides:
- NodeBase: Abstract base class for all node types
- Built-in node implementations (LinearNode, etc.)
- Node registry for custom node registration
- Config validation for node parameters
"""

from fabricpc.nodes.base import (
    SlotSpec,
    Slot,
    NodeBase,
    FlattenInputMixin,
)
from fabricpc.nodes.registry import (
    register_node,
    get_node_class,
    list_node_types,
    unregister_node,
    clear_registry,
    validate_node_config,
    discover_external_nodes,
    NodeRegistrationError,
)

# Import node modules to trigger @register_node decorators
from fabricpc.nodes.linear import LinearNode, LinearExplicitGrad
from fabricpc.nodes.transformer import TransformerBlockNode

# Discover external nodes from installed packages
discover_external_nodes()

__all__ = [
    # Base classes and mixins
    "SlotSpec",
    "Slot",
    "NodeBase",
    "FlattenInputMixin",
    # Registry
    "register_node",
    "get_node_class",
    "list_node_types",
    "unregister_node",
    "clear_registry",
    "validate_node_config",
    "discover_external_nodes",
    "NodeRegistrationError",
    # Built-in nodes
    "LinearNode",
    "LinearExplicitGrad",
    "TransformerBlockNode",
]
