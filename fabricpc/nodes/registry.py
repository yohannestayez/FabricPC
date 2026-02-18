"""
Node type registry with decorator-based registration.

This module provides a plugin architecture for custom node types:
- @register_node decorator for registering node classes
- Config schema validation for custom node parameters
- Entry point discovery for external packages
"""

from typing import Type, Dict, Any, List

from fabricpc.nodes.base import NodeBase
from fabricpc.core.registry import (
    Registry,
    RegistrationError,
    validate_config_schema,
    validate_default_energy_config,
)
from fabricpc.core.config import ConfigValidationError


class NodeRegistrationError(RegistrationError):
    """Raised when node registration fails."""

    pass


# Create the node registry instance
_node_registry = Registry(
    name="node",
    entry_point_group="fabricpc.nodes",
    required_attrs=["CONFIG_SCHEMA", "DEFAULT_ENERGY_CONFIG"],
    required_methods=["get_slots", "initialize_params", "forward"],
    attr_validators={
        "CONFIG_SCHEMA": validate_config_schema,
        "DEFAULT_ENERGY_CONFIG": validate_default_energy_config,
    },
)
_node_registry.set_error_class(NodeRegistrationError)


def register_node(node_type: str):
    """
    Decorator to register a node class with the registry.

    Usage:
        @register_node("conv2d")
        class Conv2DNode(NodeBase):
            ...

    Args:
        node_type: Unique identifier for this node type (case-insensitive)

    Returns:
        Decorator function

    Raises:
        NodeRegistrationError: If registration fails (duplicate, missing methods)
    """
    return _node_registry.register(node_type)


def get_node_class(node_type: str) -> Type[NodeBase]:
    """
    Get a node class by its registered type name.

    Args:
        node_type: The registered node type (case-insensitive)

    Returns:
        The node class

    Raises:
        ValueError: If node type is not registered
    """
    return _node_registry.get(node_type)


def list_node_types() -> List[str]:
    """Return list of all registered node types."""
    return _node_registry.list_types()


def unregister_node(node_type: str) -> None:
    """
    Remove a node type from the registry.
    Primarily for testing purposes.

    Args:
        node_type: The node type to unregister (case-insensitive)
    """
    _node_registry.unregister(node_type)


def clear_registry() -> None:
    """Clear all registrations. For testing only."""
    _node_registry.clear()


def validate_node_config(
    node_class: Type[NodeBase], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and apply defaults from node's CONFIG_SCHEMA.

    Merges NodeBase.BASE_CONFIG_SCHEMA (name, shape, type) with the
    node-specific CONFIG_SCHEMA before validation.

    Args:
        node_class: The node class with CONFIG_SCHEMA
        config: The user-provided config dict

    Returns:
        Config dict with defaults applied

    Raises:
        ConfigValidationError: If required fields are missing or validation fails
    """
    from fabricpc.core.config import validate_config

    # Merge base schema with node-specific schema
    base_schema = getattr(NodeBase, "BASE_CONFIG_SCHEMA", {})
    node_schema = getattr(node_class, "CONFIG_SCHEMA", {}) or {}
    merged_schema = {**base_schema, **node_schema}

    node_name = config.get("name", "unknown")
    return validate_config(merged_schema, config, context=f"node '{node_name}'")


def discover_external_nodes() -> None:
    """
    Discover and register nodes from installed packages via entry points.

    Looks for packages with entry points in the "fabricpc.nodes" group.
    Each entry point should map a node type name to a NodeBase subclass.

    Example pyproject.toml for an external package:
        [project.entry-points."fabricpc.nodes"]
        conv2d = "my_package.nodes:Conv2DNode"
    """
    _node_registry.discover_external()
