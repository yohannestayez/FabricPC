#!/usr/bin/env python3
"""
Test suite for the Initializer system.

Tests:
- Built-in initializers (zeros, ones, normal, uniform, xavier, kaiming)
- Registry registration/lookup/validation
- Custom initializer registration
- Config validation with defaults
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.initializers import (
    InitializerBase,
    register_initializer,
    get_initializer_class,
    list_initializer_types,
    unregister_initializer,
    validate_initializer_config,
    initialize,
    InitializerRegistrationError,
)

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


class TestBuiltinInitializers:
    """Test suite for built-in initializer implementations."""

    def test_zeros_initializer(self, rng_key):
        """Test zeros initializer returns all zeros."""
        shape = (32, 64)
        result = initialize(rng_key, shape, {"type": "zeros"})

        assert result.shape == shape
        assert jnp.all(result == 0.0)

    def test_ones_initializer(self, rng_key):
        """Test ones initializer returns all ones."""
        shape = (16, 32)
        result = initialize(rng_key, shape, {"type": "ones"})

        assert result.shape == shape
        assert jnp.all(result == 1.0)

    def test_normal_initializer_default(self, rng_key):
        """Test normal initializer with default config."""
        shape = (1000, 100)
        result = initialize(rng_key, shape, {"type": "normal"})

        assert result.shape == shape
        # Default mean=0.0, std=0.05
        assert jnp.abs(jnp.mean(result)) < 0.01  # Should be close to 0
        assert jnp.abs(jnp.std(result) - 0.05) < 0.01  # Should be close to 0.05

    def test_normal_initializer_custom(self, rng_key):
        """Test normal initializer with custom mean and std."""
        shape = (1000, 100)
        result = initialize(rng_key, shape, {"type": "normal", "mean": 5.0, "std": 2.0})

        assert result.shape == shape
        assert jnp.abs(jnp.mean(result) - 5.0) < 0.1  # Should be close to 5
        assert jnp.abs(jnp.std(result) - 2.0) < 0.1  # Should be close to 2

    def test_uniform_initializer_default(self, rng_key):
        """Test uniform initializer with default config."""
        shape = (1000, 100)
        result = initialize(rng_key, shape, {"type": "uniform"})

        assert result.shape == shape
        # Default min=-0.1, max=0.1
        assert jnp.all(result >= -0.1)
        assert jnp.all(result <= 0.1)

    def test_uniform_initializer_custom(self, rng_key):
        """Test uniform initializer with custom min and max."""
        shape = (1000, 100)
        result = initialize(
            rng_key, shape, {"type": "uniform", "min": -1.0, "max": 1.0}
        )

        assert result.shape == shape
        assert jnp.all(result >= -1.0)
        assert jnp.all(result <= 1.0)
        # Mean should be close to 0 for uniform(-1, 1)
        assert jnp.abs(jnp.mean(result)) < 0.1

    def test_xavier_initializer_normal(self, rng_key):
        """Test Xavier initializer with normal distribution."""
        shape = (256, 128)
        result = initialize(
            rng_key, shape, {"type": "xavier", "distribution": "normal"}
        )

        assert result.shape == shape
        # Xavier std = sqrt(2 / (fan_in + fan_out)) = sqrt(2 / 384) ≈ 0.072
        expected_std = jnp.sqrt(2.0 / (256 + 128))
        assert jnp.abs(jnp.std(result) - expected_std) < 0.01

    def test_xavier_initializer_uniform(self, rng_key):
        """Test Xavier initializer with uniform distribution."""
        shape = (256, 128)
        result = initialize(
            rng_key, shape, {"type": "xavier", "distribution": "uniform"}
        )

        assert result.shape == shape
        # Xavier limit = sqrt(6 / (fan_in + fan_out)) = sqrt(6 / 384) ≈ 0.125
        expected_limit = jnp.sqrt(6.0 / (256 + 128))
        assert jnp.all(result >= -expected_limit - 0.01)
        assert jnp.all(result <= expected_limit + 0.01)

    def test_kaiming_initializer_fan_in_relu(self, rng_key):
        """Test Kaiming initializer with fan_in mode and ReLU."""
        shape = (512, 256)
        result = initialize(
            rng_key,
            shape,
            {
                "type": "kaiming",
                "mode": "fan_in",
                "nonlinearity": "relu",
                "distribution": "normal",
            },
        )

        assert result.shape == shape
        # Kaiming std = sqrt(2) / sqrt(fan_in) = sqrt(2) / sqrt(512) ≈ 0.0625
        expected_std = jnp.sqrt(2.0) / jnp.sqrt(512)
        assert jnp.abs(jnp.std(result) - expected_std) < 0.01

    def test_kaiming_initializer_fan_out(self, rng_key):
        """Test Kaiming initializer with fan_out mode."""
        shape = (512, 256)
        result = initialize(
            rng_key,
            shape,
            {
                "type": "kaiming",
                "mode": "fan_out",
                "nonlinearity": "relu",
                "distribution": "normal",
            },
        )

        assert result.shape == shape
        # Kaiming std = sqrt(2) / sqrt(fan_out) = sqrt(2) / sqrt(256)
        expected_std = jnp.sqrt(2.0) / jnp.sqrt(256)
        assert jnp.abs(jnp.std(result) - expected_std) < 0.01

    def test_kaiming_initializer_leaky_relu(self, rng_key):
        """Test Kaiming initializer with leaky ReLU."""
        shape = (512, 256)
        a = 0.2  # Leaky ReLU slope
        result = initialize(
            rng_key,
            shape,
            {
                "type": "kaiming",
                "mode": "fan_in",
                "nonlinearity": "leaky_relu",
                "distribution": "normal",
                "a": a,
            },
        )

        assert result.shape == shape
        # Kaiming std = sqrt(2 / (1 + a^2)) / sqrt(fan_in)
        gain = jnp.sqrt(2.0 / (1 + a**2))
        expected_std = gain / jnp.sqrt(512)
        assert jnp.abs(jnp.std(result) - expected_std) < 0.01


class TestInitializerRegistry:
    """Test suite for initializer registry operations."""

    def test_list_initializer_types(self):
        """Test listing registered initializer types."""
        types = list_initializer_types()

        # Should include all built-in types
        assert "zeros" in types
        assert "ones" in types
        assert "normal" in types
        assert "uniform" in types
        assert "xavier" in types
        assert "kaiming" in types

    def test_get_initializer_class(self):
        """Test getting initializer class by type."""
        normal_class = get_initializer_class("normal")

        assert normal_class is not None
        assert hasattr(normal_class, "initialize")
        assert hasattr(normal_class, "CONFIG_SCHEMA")

    def test_get_initializer_class_case_insensitive(self):
        """Test that type lookup is case-insensitive."""
        assert get_initializer_class("Normal") == get_initializer_class("normal")
        assert get_initializer_class("XAVIER") == get_initializer_class("xavier")

    def test_get_unknown_type_raises(self):
        """Test that unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            get_initializer_class("nonexistent_initializer")

    def test_register_custom_initializer(self, rng_key):
        """Test registering a custom initializer."""

        @register_initializer("test_custom")
        class TestCustomInitializer(InitializerBase):
            CONFIG_SCHEMA = {"multiplier": {"type": (int, float), "default": 2.0}}

            @staticmethod
            def initialize(key, shape, config=None):
                multiplier = config.get("multiplier", 2.0) if config else 2.0
                return multiplier * jnp.ones(shape)

        try:
            # Verify registration
            assert "test_custom" in list_initializer_types()

            # Test using the initializer
            result = initialize(rng_key, (4, 4), {"type": "test_custom"})
            assert jnp.all(result == 2.0)

            result_custom = initialize(
                rng_key, (4, 4), {"type": "test_custom", "multiplier": 5.0}
            )
            assert jnp.all(result_custom == 5.0)
        finally:
            # Cleanup
            unregister_initializer("test_custom")

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration with different class raises error."""

        @register_initializer("test_dup")
        class FirstInit(InitializerBase):
            CONFIG_SCHEMA = {}

            @staticmethod
            def initialize(key, shape, config=None):
                return jnp.zeros(shape)

        try:
            with pytest.raises(
                InitializerRegistrationError, match="already registered"
            ):

                @register_initializer("test_dup")
                class SecondInit(InitializerBase):
                    CONFIG_SCHEMA = {}

                    @staticmethod
                    def initialize(key, shape, config=None):
                        return jnp.ones(shape)

        finally:
            unregister_initializer("test_dup")

    def test_idempotent_registration_same_class(self):
        """Test that registering the same class object twice is idempotent."""

        class IdempotentInit(InitializerBase):
            CONFIG_SCHEMA = {}

            @staticmethod
            def initialize(key, shape, config=None):
                return jnp.zeros(shape)

        try:
            # First registration
            register_initializer("test_idem")(IdempotentInit)
            assert "test_idem" in list_initializer_types()

            # Second registration with same class object should be idempotent
            register_initializer("test_idem")(IdempotentInit)
            assert "test_idem" in list_initializer_types()
        finally:
            unregister_initializer("test_idem")


class TestConfigValidation:
    """Test suite for initializer config validation."""

    def test_validate_config_applies_defaults(self):
        """Test that validation applies default values."""
        normal_class = get_initializer_class("normal")
        config = {"type": "normal"}
        validated = validate_initializer_config(normal_class, config)

        assert validated["mean"] == 0.0
        assert validated["std"] == 0.05

    def test_validate_config_preserves_custom_values(self):
        """Test that validation preserves custom values."""
        normal_class = get_initializer_class("normal")
        config = {"type": "normal", "mean": 1.0, "std": 0.5}
        validated = validate_initializer_config(normal_class, config)

        assert validated["mean"] == 1.0
        assert validated["std"] == 0.5


class TestInitializerDeterminism:
    """Test that initializers are deterministic with same key."""

    def test_normal_deterministic(self, rng_key):
        """Test normal initializer is deterministic."""
        shape = (64, 64)
        config = {"type": "normal"}

        result1 = initialize(rng_key, shape, config)
        result2 = initialize(rng_key, shape, config)

        assert jnp.allclose(result1, result2)

    def test_different_keys_different_results(self, rng_key):
        """Test different keys produce different results."""
        shape = (64, 64)
        config = {"type": "normal"}

        key1, key2 = jax.random.split(rng_key)
        result1 = initialize(key1, shape, config)
        result2 = initialize(key2, shape, config)

        assert not jnp.allclose(result1, result2)

    def test_xavier_deterministic(self, rng_key):
        """Test Xavier initializer is deterministic."""
        shape = (128, 64)
        config = {"type": "xavier"}

        result1 = initialize(rng_key, shape, config)
        result2 = initialize(rng_key, shape, config)

        assert jnp.allclose(result1, result2)


class TestInitializerShapes:
    """Test initializers work with various shapes."""

    def test_1d_shape(self, rng_key):
        """Test initializer with 1D shape."""
        result = initialize(rng_key, (128,), {"type": "normal"})
        assert result.shape == (128,)

    def test_2d_shape(self, rng_key):
        """Test initializer with 2D shape."""
        result = initialize(rng_key, (64, 128), {"type": "xavier"})
        assert result.shape == (64, 128)

    def test_3d_shape(self, rng_key):
        """Test initializer with 3D shape."""
        result = initialize(rng_key, (32, 28, 28), {"type": "uniform"})
        assert result.shape == (32, 28, 28)

    def test_4d_shape(self, rng_key):
        """Test initializer with 4D shape (conv kernel)."""
        result = initialize(rng_key, (3, 3, 32, 64), {"type": "kaiming"})
        assert result.shape == (3, 3, 32, 64)


class TestInterfaceValidation:
    """Test that registration validates the initializer interface."""

    def test_missing_config_schema_raises(self):
        """Test that missing CONFIG_SCHEMA raises error."""
        with pytest.raises(InitializerRegistrationError, match="CONFIG_SCHEMA"):

            @register_initializer("test_no_schema")
            class NoSchemaInit(InitializerBase):
                @staticmethod
                def initialize(key, shape, config=None):
                    return jnp.zeros(shape)

    def test_missing_initialize_raises(self):
        """Test that missing initialize method raises error."""
        with pytest.raises(InitializerRegistrationError, match="initialize"):

            @register_initializer("test_no_init")
            class NoInitMethod(InitializerBase):
                CONFIG_SCHEMA = {}
