"""
Test suite for energy functional module.

Tests registration, lookup, config validation, built-in functionals,
and integration with node energy computation.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest
import jax
import jax.numpy as jnp
from typing import Dict, Any

from fabricpc.core.energy import (
    EnergyFunctional,
    register_energy,
    get_energy_class,
    list_energy_types,
    unregister_energy,
    clear_energy_registry,
    validate_energy_config,
    compute_energy,
    compute_energy_gradient,
    get_energy_and_gradient,
    EnergyRegistrationError,
    GaussianEnergy,
    BernoulliEnergy,
    CrossEntropyEnergy,
    LaplacianEnergy,
    HuberEnergy,
    KLDivergenceEnergy,
    _ENERGY_REGISTRY,
)
from fabricpc.graph.graph_net import create_pc_graph


class TestEnergyRegistration:
    """Test energy functional registration."""

    def test_builtin_energies_registered(self):
        """Test that built-in energy functionals are registered on import."""
        types = list_energy_types()
        assert "gaussian" in types
        assert "bernoulli" in types
        assert "cross_entropy" in types
        assert "laplacian" in types
        assert "huber" in types

    def test_get_energy_class_returns_correct_class(self):
        """Test that get_energy_class returns the correct class."""
        assert get_energy_class("gaussian") is GaussianEnergy
        assert get_energy_class("bernoulli") is BernoulliEnergy
        assert get_energy_class("cross_entropy") is CrossEntropyEnergy

    def test_get_energy_class_case_insensitive(self):
        """Test that energy type lookup is case-insensitive."""
        assert get_energy_class("GAUSSIAN") is GaussianEnergy
        assert get_energy_class("Gaussian") is GaussianEnergy
        assert get_energy_class("BERNOULLI") is BernoulliEnergy

    def test_get_energy_class_unknown_type_raises(self):
        """Test that unknown energy type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_energy_class("unknown_energy")
        assert "Unknown energy type" in str(exc_info.value)
        assert "unknown_energy" in str(exc_info.value)


class TestCustomEnergyRegistration:
    """Test registering custom energy functionals."""

    def test_register_custom_energy(self):
        """Test registering a custom energy functional."""

        @register_energy("test_custom_energy")
        class TestCustomEnergy(EnergyFunctional):
            CONFIG_SCHEMA = {}

            @staticmethod
            def energy(z_latent, z_mu, config=None):
                diff = z_latent - z_mu
                axes = tuple(range(1, len(diff.shape)))
                return jnp.sum(jnp.abs(diff), axis=axes)

            @staticmethod
            def grad_latent(z_latent, z_mu, config=None):
                return jnp.sign(z_latent - z_mu)

        try:
            assert get_energy_class("test_custom_energy") is TestCustomEnergy
            assert "test_custom_energy" in list_energy_types()
        finally:
            unregister_energy("test_custom_energy")

    def test_duplicate_registration_different_class_raises(self):
        """Test that registering same type with different class raises."""

        @register_energy("test_dup_energy")
        class TestEnergy1(EnergyFunctional):
            CONFIG_SCHEMA = {}

            @staticmethod
            def energy(z_latent, z_mu, config=None):
                return jnp.zeros((z_latent.shape[0],))

            @staticmethod
            def grad_latent(z_latent, z_mu, config=None):
                return jnp.zeros_like(z_latent)

        try:
            with pytest.raises(EnergyRegistrationError) as exc_info:

                @register_energy("test_dup_energy")
                class TestEnergy2(EnergyFunctional):
                    CONFIG_SCHEMA = {}

                    @staticmethod
                    def energy(z_latent, z_mu, config=None):
                        return jnp.zeros((z_latent.shape[0],))

                    @staticmethod
                    def grad_latent(z_latent, z_mu, config=None):
                        return jnp.zeros_like(z_latent)

            assert "already registered" in str(exc_info.value)
        finally:
            unregister_energy("test_dup_energy")

    def test_missing_config_schema_raises(self):
        """Test that missing CONFIG_SCHEMA raises error."""
        with pytest.raises(EnergyRegistrationError) as exc_info:

            @register_energy("test_no_schema")
            class NoSchemaEnergy(EnergyFunctional):
                @staticmethod
                def energy(z_latent, z_mu, config=None):
                    return jnp.zeros((z_latent.shape[0],))

                @staticmethod
                def grad_latent(z_latent, z_mu, config=None):
                    return jnp.zeros_like(z_latent)

        assert "CONFIG_SCHEMA" in str(exc_info.value)

    def test_missing_method_raises(self):
        """Test that missing required method raises error."""
        with pytest.raises(EnergyRegistrationError) as exc_info:

            @register_energy("test_missing_method")
            class MissingMethodEnergy(EnergyFunctional):
                CONFIG_SCHEMA = {}

                @staticmethod
                def energy(z_latent, z_mu, config=None):
                    return jnp.zeros((z_latent.shape[0],))

                # Missing grad_latent

        assert "grad_latent" in str(exc_info.value)


class TestGaussianEnergy:
    """Test Gaussian energy functional."""

    def test_gaussian_energy_computation(self):
        """Test Gaussian energy: E = 0.5 * ||z - mu||^2"""
        z_latent = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        z_mu = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        energy = GaussianEnergy.energy(z_latent, z_mu)

        # Expected: 0.5 * (1^2 + 2^2 + 3^2) = 0.5 * 14 = 7.0 for first sample
        # Expected: 0.5 * (4^2 + 5^2 + 6^2) = 0.5 * 77 = 38.5 for second sample
        assert energy.shape == (2,)
        assert jnp.allclose(energy[0], 7.0)
        assert jnp.allclose(energy[1], 38.5)

    def test_gaussian_gradient_computation(self):
        """Test Gaussian gradient: dE/dz = z - mu"""
        z_latent = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        z_mu = jnp.array([[0.5, 1.0], [1.5, 2.0]])

        grad = GaussianEnergy.grad_latent(z_latent, z_mu)

        expected = z_latent - z_mu
        assert jnp.allclose(grad, expected)

    def test_gaussian_precision_parameter(self):
        """Test Gaussian energy with precision parameter."""
        z_latent = jnp.array([[1.0, 2.0]])
        z_mu = jnp.array([[0.0, 0.0]])
        config = {"precision": 2.0}

        energy = GaussianEnergy.energy(z_latent, z_mu, config)
        grad = GaussianEnergy.grad_latent(z_latent, z_mu, config)

        # With precision=2: E = precision/2 * ||z - mu||^2 = 1.0 * 5 = 5.0
        assert jnp.allclose(energy[0], 5.0)
        # Gradient scaled by precision
        assert jnp.allclose(grad, 2.0 * (z_latent - z_mu))


class TestBernoulliEnergy:
    """Test Bernoulli (BCE) energy functional."""

    def test_bernoulli_energy_computation(self):
        """Test Bernoulli cross-entropy energy."""
        # Binary targets
        z_latent = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        # Predicted probabilities
        z_mu = jnp.array([[0.9, 0.1], [0.2, 0.8]])

        energy = BernoulliEnergy.energy(z_latent, z_mu)

        # BCE = -[z*log(mu) + (1-z)*log(1-mu)]
        assert energy.shape == (2,)
        assert energy[0] > 0  # Should be positive
        assert energy[1] > 0

    def test_bernoulli_perfect_prediction(self):
        """Test that perfect prediction gives near-zero energy."""
        z_latent = jnp.array([[1.0, 0.0]])
        z_mu = jnp.array([[0.9999, 0.0001]])  # Near-perfect

        energy = BernoulliEnergy.energy(z_latent, z_mu)
        assert energy[0] < 0.01  # Should be very small


class TestCrossEntropyEnergy:
    """Test Cross Entropy energy functional."""

    def test_categorical_energy_computation(self):
        """Test cross-entropy energy."""
        # One-hot targets
        z_latent = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # Softmax probabilities
        z_mu = jnp.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])

        energy = CrossEntropyEnergy.energy(z_latent, z_mu)

        # CE = -sum(z * log(mu))
        assert energy.shape == (2,)
        expected_0 = -jnp.log(0.8)  # Only the correct class contributes
        expected_1 = -jnp.log(0.7)
        assert jnp.allclose(energy[0], expected_0, atol=1e-5)
        assert jnp.allclose(energy[1], expected_1, atol=1e-5)


class TestLaplacianEnergy:
    """Test Laplacian (L1) energy functional."""

    def test_laplacian_energy_computation(self):
        """Test Laplacian energy: E = ||z - mu||_1"""
        z_latent = jnp.array([[1.0, -2.0, 3.0]])
        z_mu = jnp.array([[0.0, 0.0, 0.0]])

        energy = LaplacianEnergy.energy(z_latent, z_mu)

        # Expected: |1| + |-2| + |3| = 6.0
        assert jnp.allclose(energy[0], 6.0)

    def test_laplacian_gradient_is_sign(self):
        """Test Laplacian gradient: dE/dz = sign(z - mu)"""
        z_latent = jnp.array([[1.0, -2.0, 0.0]])
        z_mu = jnp.array([[0.0, 0.0, 0.0]])

        grad = LaplacianEnergy.grad_latent(z_latent, z_mu)

        expected = jnp.array([[1.0, -1.0, 0.0]])
        assert jnp.allclose(grad, expected)


class TestHuberEnergy:
    """Test Huber (smooth L1) energy functional."""

    def test_huber_quadratic_region(self):
        """Test Huber energy in quadratic region (|diff| <= delta)."""
        z_latent = jnp.array([[0.5]])
        z_mu = jnp.array([[0.0]])
        config = {"delta": 1.0}

        energy = HuberEnergy.energy(z_latent, z_mu, config)

        # In quadratic region: E = 0.5 * diff^2 = 0.5 * 0.25 = 0.125
        assert jnp.allclose(energy[0], 0.125)

    def test_huber_linear_region(self):
        """Test Huber energy in linear region (|diff| > delta)."""
        z_latent = jnp.array([[2.0]])
        z_mu = jnp.array([[0.0]])
        config = {"delta": 1.0}

        energy = HuberEnergy.energy(z_latent, z_mu, config)

        # In linear region: E = delta * (|diff| - 0.5 * delta) = 1.0 * (2.0 - 0.5) = 1.5
        assert jnp.allclose(energy[0], 1.5)


class TestKLDivergenceEnergy:
    """Test KL Divergence energy functional."""

    def test_kl_divergence_registered(self):
        """Test that KL divergence is registered."""
        types = list_energy_types()
        assert "kl_divergence" in types
        assert get_energy_class("kl_divergence") is KLDivergenceEnergy

    def test_kl_divergence_identical_distributions(self):
        """Test KL divergence is zero for identical distributions."""
        # When z == mu, KL divergence should be 0
        z_latent = jnp.array([[0.2, 0.3, 0.4, 0.1]])
        z_mu = jnp.array([[0.2, 0.3, 0.4, 0.1]])

        energy = KLDivergenceEnergy.energy(z_latent, z_mu)

        assert energy.shape == (1,)
        assert jnp.allclose(energy[0], 0.0, atol=1e-6)

    def test_kl_divergence_batch_computation(self):
        """Test KL divergence computes correct numerical values."""
        z_latent = jnp.array(
            [
                [0.7, 0.2, 0.1],
                [0.3, 0.3, 0.4],
            ]
        )
        z_mu = jnp.array(
            [
                [0.6, 0.3, 0.1],
                [0.5, 0.25, 0.25],
            ]
        )

        energy = KLDivergenceEnergy.energy(z_latent, z_mu)

        assert energy.shape == (2,)

        # Manually compute expected values
        expected_0 = (
            0.7 * jnp.log(0.7 / 0.6)
            + 0.2 * jnp.log(0.2 / 0.3)
            + 0.1 * jnp.log(0.1 / 0.1)
        )
        expected_1 = (
            0.3 * jnp.log(0.3 / 0.5)
            + 0.3 * jnp.log(0.3 / 0.25)
            + 0.4 * jnp.log(0.4 / 0.25)
        )

        assert jnp.allclose(energy[0], expected_0, atol=1e-5)
        assert jnp.allclose(energy[1], expected_1, atol=1e-5)

    def test_kl_divergence_always_non_negative(self):
        """Test that KL divergence is always >= 0 (Gibbs' inequality)."""
        # Random probability distributions
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        # Generate random positive values and normalize
        raw_z = jax.random.uniform(key1, (10, 5)) + 0.01
        raw_mu = jax.random.uniform(key2, (10, 5)) + 0.01
        z_latent = raw_z / raw_z.sum(axis=-1, keepdims=True)
        z_mu = raw_mu / raw_mu.sum(axis=-1, keepdims=True)

        energy = KLDivergenceEnergy.energy(z_latent, z_mu)

        # KL divergence should always be non-negative
        assert jnp.all(energy >= -1e-6)  # Small tolerance for numerical precision

    def test_kl_divergence_zero_probability_handling(self):
        """Test that zero probabilities are handled correctly."""
        # When z has zeros, those terms should contribute 0 (0 * log(0) = 0)
        z_latent = jnp.array([[1.0, 0.0, 0.0]])
        z_mu = jnp.array([[0.8, 0.1, 0.1]])

        energy = KLDivergenceEnergy.energy(z_latent, z_mu)

        # Only the first term contributes: 1.0 * log(1.0 / 0.8)
        expected = 1.0 * jnp.log(1.0 / 0.8)

        assert jnp.isfinite(energy[0])  # Should not be inf or nan
        assert jnp.allclose(energy[0], expected, atol=1e-5)


class TestConfigValidation:
    """Test energy config validation."""

    def test_default_applied(self):
        """Test that missing config gets default values."""
        config = {"type": "gaussian"}
        result = validate_energy_config(GaussianEnergy, config)
        assert result["precision"] == 1.0

    def test_type_mismatch_raises(self):
        """Test that wrong type raises ConfigValidationError."""
        from fabricpc.core.config import ConfigValidationError

        config = {"precision": "high"}  # Should be float
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_energy_config(GaussianEnergy, config)
        assert "precision" in str(exc_info.value)


class TestConvenienceFunctions:
    """Test convenience functions for energy computation."""

    def test_compute_energy(self):
        """Test compute_energy dispatches correctly."""
        z_latent = jnp.array([[1.0, 2.0]])
        z_mu = jnp.array([[0.0, 0.0]])

        # Default (gaussian)
        energy = compute_energy(z_latent, z_mu)
        assert jnp.allclose(energy[0], 2.5)  # 0.5 * (1 + 4)

        # Explicit gaussian
        energy = compute_energy(z_latent, z_mu, {"type": "gaussian"})
        assert jnp.allclose(energy[0], 2.5)

    def test_compute_energy_gradient(self):
        """Test compute_energy_gradient dispatches correctly."""
        z_latent = jnp.array([[1.0, 2.0]])
        z_mu = jnp.array([[0.5, 1.0]])

        grad = compute_energy_gradient(z_latent, z_mu)
        expected = z_latent - z_mu
        assert jnp.allclose(grad, expected)

    def test_get_energy_and_gradient(self):
        """Test combined energy and gradient computation."""
        z_latent = jnp.array([[1.0, 2.0]])
        z_mu = jnp.array([[0.0, 0.0]])

        energy, grad = get_energy_and_gradient(z_latent, z_mu)

        assert jnp.allclose(energy[0], 2.5)
        assert jnp.allclose(grad, z_latent - z_mu)


class TestIntegration:
    """Integration tests with graph construction."""

    def test_graph_creation_with_default_energy(self):
        """Test that graphs use default energy config."""
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

        # Both nodes should have gaussian energy by default
        assert structure.nodes["input"].node_config["energy"]["type"] == "gaussian"
        assert structure.nodes["output"].node_config["energy"]["type"] == "gaussian"

    def test_graph_creation_with_custom_energy(self):
        """Test that graphs can use custom energy config."""
        config = {
            "node_list": [
                {"name": "input", "shape": (8,), "type": "linear"},
                {
                    "name": "output",
                    "shape": (4,),
                    "type": "linear",
                    "energy": {"type": "bernoulli"},
                },
            ],
            "edge_list": [
                {"source_name": "input", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "input", "y": "output"},
        }

        key = jax.random.PRNGKey(0)
        params, structure = create_pc_graph(config, key)

        assert structure.nodes["input"].node_config["energy"]["type"] == "gaussian"
        assert structure.nodes["output"].node_config["energy"]["type"] == "bernoulli"

    def test_graph_creation_energy_shorthand(self):
        """Test that energy can be specified as string shorthand."""
        config = {
            "node_list": [
                {"name": "input", "shape": (8,), "type": "linear"},
                {
                    "name": "output",
                    "shape": (4,),
                    "type": "linear",
                    "energy": "cross_entropy",  # String shorthand
                },
            ],
            "edge_list": [
                {"source_name": "input", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "input", "y": "output"},
        }

        key = jax.random.PRNGKey(0)
        params, structure = create_pc_graph(config, key)

        assert (
            structure.nodes["output"].node_config["energy"]["type"] == "cross_entropy"
        )

    def test_unknown_energy_type_raises(self):
        """Test that unknown energy type raises error during graph creation."""
        config = {
            "node_list": [
                {
                    "name": "input",
                    "shape": (8,),
                    "type": "linear",
                    "energy": {"type": "nonexistent_energy"},
                },
            ],
            "edge_list": [],
            "task_map": {"x": "input"},
        }

        key = jax.random.PRNGKey(0)
        with pytest.raises(ValueError) as exc_info:
            create_pc_graph(config, key)
        assert "Unknown energy type" in str(exc_info.value)


class TestNDimensionalShapes:
    """Test energy computation with various tensor shapes."""

    def test_1d_tensors(self):
        """Test energy with 1D tensors (batch, features)."""
        z_latent = jnp.ones((4, 10))
        z_mu = jnp.zeros((4, 10))

        energy = GaussianEnergy.energy(z_latent, z_mu)
        assert energy.shape == (4,)
        assert jnp.allclose(energy, 5.0)  # 0.5 * 10 = 5.0

    def test_2d_tensors(self):
        """Test energy with 2D tensors (batch, h, w)."""
        z_latent = jnp.ones((2, 4, 4))
        z_mu = jnp.zeros((2, 4, 4))

        energy = GaussianEnergy.energy(z_latent, z_mu)
        assert energy.shape == (2,)
        assert jnp.allclose(energy, 8.0)  # 0.5 * 16 = 8.0

    def test_3d_tensors(self):
        """Test energy with 3D tensors (batch, h, w, c)."""
        z_latent = jnp.ones((2, 4, 4, 3))
        z_mu = jnp.zeros((2, 4, 4, 3))

        energy = GaussianEnergy.energy(z_latent, z_mu)
        assert energy.shape == (2,)
        assert jnp.allclose(energy, 24.0)  # 0.5 * 48 = 24.0
