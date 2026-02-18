"""
Test suite for node registry functionality.

Tests registration, lookup, config validation, and entry point discovery.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple

from fabricpc.nodes.base import NodeBase, SlotSpec
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
from fabricpc.nodes import (
    LinearNode,
    LinearExplicitGrad,
    get_node_class,
)
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.graph.graph_net import create_pc_graph


class TestNodeRegistration:
    """Test node registration functionality."""

    def test_builtin_nodes_registered(self):
        """Test that built-in nodes are registered on import."""
        types = list_node_types()
        assert "linear" in types
        assert "linear_explicit_grad" in types

    def test_get_node_class_returns_correct_class(self):
        """Test that get_node_class returns the correct node class."""
        assert get_node_class("linear") is LinearNode
        assert get_node_class("linear_explicit_grad") is LinearExplicitGrad

    def test_get_node_class_case_insensitive(self):
        """Test that node type lookup is case-insensitive."""
        assert get_node_class("LINEAR") is LinearNode
        assert get_node_class("Linear") is LinearNode
        assert get_node_class("LINEAR_EXPLICIT_GRAD") is LinearExplicitGrad

    def test_get_node_class_unknown_type_raises(self):
        """Test that unknown node type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_node_class("unknown_node_type")
        assert "Unknown node type" in str(exc_info.value)
        assert "unknown_node_type" in str(exc_info.value)

    def test_backward_compatibility_alias(self):
        """Test that get_node_class still works."""
        assert get_node_class("linear") is LinearNode
        assert get_node_class is get_node_class


class TestCustomNodeRegistration:
    """Test registering custom node types."""

    def test_register_custom_node(self):
        """Test registering a custom node type."""

        @register_node("test_custom")
        class TestCustomNode(NodeBase):
            CONFIG_SCHEMA = {}

            @staticmethod
            def get_slots():
                return {"in": SlotSpec(name="in", is_multi_input=True)}

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, config):
                return NodeParams(weights={}, biases={})

            @staticmethod
            def forward(params, inputs, state, node_info):
                return jnp.array(0.0), state

        try:
            assert get_node_class("test_custom") is TestCustomNode
            assert "test_custom" in list_node_types()
        finally:
            unregister_node("test_custom")

    def test_duplicate_registration_different_class_raises(self):
        """Test that registering same type with different class raises."""

        @register_node("test_dup")
        class TestNode1(NodeBase):
            CONFIG_SCHEMA = {}

            @staticmethod
            def get_slots():
                return {"in": SlotSpec(name="in", is_multi_input=True)}

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, config):
                return NodeParams(weights={}, biases={})

            @staticmethod
            def forward(params, inputs, state, node_info):
                return jnp.array(0.0), state

        try:
            with pytest.raises(NodeRegistrationError) as exc_info:

                @register_node("test_dup")
                class TestNode2(NodeBase):
                    CONFIG_SCHEMA = {}

                    @staticmethod
                    def get_slots():
                        return {"in": SlotSpec(name="in", is_multi_input=True)}

                    @staticmethod
                    def initialize_params(key, node_shape, input_shapes, config):
                        return NodeParams(weights={}, biases={})

                    @staticmethod
                    def forward(params, inputs, state, node_info):
                        return jnp.array(0.0), state

            assert "already registered" in str(exc_info.value)
        finally:
            unregister_node("test_dup")

    def test_idempotent_registration_same_class(self):
        """Test that registering same class twice is OK."""

        @register_node("test_idem")
        class TestIdemNode(NodeBase):
            CONFIG_SCHEMA = {}

            @staticmethod
            def get_slots():
                return {"in": SlotSpec(name="in", is_multi_input=True)}

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, config):
                return NodeParams(weights={}, biases={})

            @staticmethod
            def forward(params, inputs, state, node_info):
                return jnp.array(0.0), state

        try:
            # Register same class again - should not raise
            register_node("test_idem")(TestIdemNode)
            assert get_node_class("test_idem") is TestIdemNode
        finally:
            unregister_node("test_idem")

    def test_unregister_node(self):
        """Test that unregister_node removes the node type."""

        @register_node("test_unreg")
        class TestUnregNode(NodeBase):
            CONFIG_SCHEMA = {}

            @staticmethod
            def get_slots():
                return {"in": SlotSpec(name="in", is_multi_input=True)}

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, config):
                return NodeParams(weights={}, biases={})

            @staticmethod
            def forward(params, inputs, state, node_info):
                return jnp.array(0.0), state

        assert "test_unreg" in list_node_types()
        unregister_node("test_unreg")
        assert "test_unreg" not in list_node_types()


class TestInterfaceValidation:
    """Test that nodes must implement required interface."""

    def test_missing_config_schema_raises(self):
        """Test that missing CONFIG_SCHEMA raises error."""
        with pytest.raises(NodeRegistrationError) as exc_info:

            @register_node("test_missing_schema")
            class MissingSchemaNode(NodeBase):
                @staticmethod
                def get_slots():
                    return {"in": SlotSpec(name="in", is_multi_input=True)}

                @staticmethod
                def initialize_params(key, node_shape, input_shapes, config):
                    return NodeParams(weights={}, biases={})

                @staticmethod
                def forward(params, inputs, state, node_info):
                    return jnp.array(0.0), state

        assert "CONFIG_SCHEMA" in str(exc_info.value)

    def test_missing_forward_raises(self):
        """Test that missing forward method raises error."""
        with pytest.raises(NodeRegistrationError) as exc_info:

            @register_node("test_missing_forward")
            class MissingForwardNode(NodeBase):
                CONFIG_SCHEMA = {}

                @staticmethod
                def get_slots():
                    return {"in": SlotSpec(name="in", is_multi_input=True)}

                @staticmethod
                def initialize_params(key, node_shape, input_shapes, config):
                    return NodeParams(weights={}, biases={})

                # forward is abstract, not implemented

        assert "forward" in str(exc_info.value)
        assert "abstract" in str(exc_info.value)


class TestConfigValidation:
    """Test config schema validation."""

    def test_schema_applies_defaults(self):
        """Test that LinearNode's CONFIG_SCHEMA applies defaults."""
        config = {
            "name": "test",
            "shape": (10,),
            "type": "linear",
            "custom_field": "value",
        }
        result = validate_node_config(LinearNode, config)
        # Original fields preserved
        assert result["name"] == "test"
        assert result["custom_field"] == "value"
        # Defaults applied from LinearNode.CONFIG_SCHEMA
        assert result["use_bias"] == True
        assert result["weight_init"]["type"] == "normal"

    def test_required_field_missing_raises(self):
        """Test that missing required field raises ValueError."""

        class NodeWithSchema(NodeBase):
            CONFIG_SCHEMA = {
                "kernel_size": {
                    "type": tuple,
                    "required": True,
                    "description": "Kernel dims",
                },
            }

            @staticmethod
            def get_slots():
                return {"in": SlotSpec(name="in", is_multi_input=True)}

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, config):
                return NodeParams(weights={}, biases={})

            @staticmethod
            def forward(params, inputs, state, node_info):
                return jnp.array(0.0), state

        from fabricpc.core.config import ConfigValidationError

        config = {
            "name": "test",
            "shape": (10,),
            "type": "custom",
        }  # missing kernel_size
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_node_config(NodeWithSchema, config)
        assert "kernel_size" in str(exc_info.value)

    def test_type_mismatch_raises(self):
        """Test that wrong type raises ConfigValidationError."""

        class NodeWithSchema(NodeBase):
            CONFIG_SCHEMA = {
                "stride": {"type": tuple},
            }

            @staticmethod
            def get_slots():
                return {"in": SlotSpec(name="in", is_multi_input=True)}

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, config):
                return NodeParams(weights={}, biases={})

            @staticmethod
            def forward(params, inputs, state, node_info):
                return jnp.array(0.0), state

        from fabricpc.core.config import ConfigValidationError

        config = {
            "name": "test",
            "shape": (10,),
            "type": "custom",
            "stride": [1, 1],
        }  # list instead of tuple
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_node_config(NodeWithSchema, config)
        assert "stride" in str(exc_info.value)
        assert "tuple" in str(exc_info.value)

    def test_choices_validation_raises(self):
        """Test that invalid choice raises ValueError."""

        class NodeWithSchema(NodeBase):
            CONFIG_SCHEMA = {
                "padding": {"type": str, "choices": ["valid", "same"]},
            }

            @staticmethod
            def get_slots():
                return {"in": SlotSpec(name="in", is_multi_input=True)}

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, config):
                return NodeParams(weights={}, biases={})

            @staticmethod
            def forward(params, inputs, state, node_info):
                return jnp.array(0.0), state

        from fabricpc.core.config import ConfigValidationError

        config = {
            "name": "test",
            "shape": (10,),
            "type": "custom",
            "padding": "full",
        }  # not in choices
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_node_config(NodeWithSchema, config)
        assert "padding" in str(exc_info.value)
        assert "valid" in str(exc_info.value)

    def test_defaults_applied(self):
        """Test that missing optional field gets default value."""

        class NodeWithSchema(NodeBase):
            CONFIG_SCHEMA = {
                "stride": {"type": tuple, "default": (1, 1)},
                "padding": {"type": str, "default": "valid"},
            }

            @staticmethod
            def get_slots():
                return {"in": SlotSpec(name="in", is_multi_input=True)}

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, config):
                return NodeParams(weights={}, biases={})

            @staticmethod
            def forward(params, inputs, state, node_info):
                return jnp.array(0.0), state

        config = {
            "name": "test",
            "shape": (10,),
            "type": "custom",
        }  # missing stride and padding
        result = validate_node_config(NodeWithSchema, config)
        assert result["stride"] == (1, 1)
        assert result["padding"] == "valid"
        assert result["name"] == "test"


class TestEntryPointDiscovery:
    """Test entry point discovery functionality."""

    def test_discovery_runs_without_error(self):
        """Test that entry point discovery runs without error (even with no plugins)."""
        # Should not raise even if no plugins are installed
        discover_external_nodes()

    def test_builtin_takes_precedence(self):
        """Test that built-in nodes are not overwritten by external."""
        # Get the current linear node class
        original_linear = get_node_class("linear")

        # Run discovery again (should not overwrite)
        discover_external_nodes()

        # Should still be the same class
        assert get_node_class("linear") is original_linear


class TestIntegration:
    """Integration tests with graph construction."""

    def test_graph_creation_with_registered_nodes(self):
        """Test that graphs can be created with registered node types."""
        config = {
            "node_list": [
                {"name": "input", "shape": (8,), "type": "linear"},
                {"name": "output", "shape": (4,), "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "input", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "input", "y": "output"},
        }

        key = jax.random.PRNGKey(0)
        params, structure = create_pc_graph(config, key)

        assert len(structure.nodes) == 2
        assert structure.nodes["input"].node_type == "linear"
        assert structure.nodes["output"].node_type == "linear"

    def test_graph_creation_with_linear_explicit_grad(self):
        """Test graph creation with linear_explicit_grad node type."""
        config = {
            "node_list": [
                {"name": "input", "shape": (8,), "type": "linear_explicit_grad"},
                {"name": "output", "shape": (4,), "type": "linear_explicit_grad"},
            ],
            "edge_list": [
                {"source_name": "input", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "input", "y": "output"},
        }

        key = jax.random.PRNGKey(0)
        params, structure = create_pc_graph(config, key)

        assert structure.nodes["input"].node_type == "linear_explicit_grad"
        assert structure.nodes["output"].node_type == "linear_explicit_grad"

    def test_unknown_node_type_in_config_raises(self):
        """Test that unknown node type in config raises error."""
        config = {
            "node_list": [
                {"name": "input", "shape": (8,), "type": "nonexistent_type"},
            ],
            "edge_list": [],
            "task_map": {"x": "input"},
        }

        key = jax.random.PRNGKey(0)
        with pytest.raises(ValueError) as exc_info:
            create_pc_graph(config, key)
        assert "Unknown node type" in str(exc_info.value)
