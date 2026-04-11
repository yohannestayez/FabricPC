"""
Proximal weight carryover transforms for temporal predictive coding.

These optax gradient transformations implement the Temporal Carryover
framework (TPC), which stabilises incremental PC by penalising large
weight excursions from a carried anchor point.

Two tiers are provided:
- Tier 1 (Euclidean): adds ``penalty_strength * (params - anchor)`` to the gradient.
- Tier 2 (Fisher-weighted): adds ``diag(F) * penalty_strength * (params - anchor)``.

The anchor is updated at cadence boundaries by the training loop via
``update_anchor`` / ``update_anchor_in_chain``.

Reference: "Proximal Stabilisation for Incremental Predictive Coding" (TPC v0.3)
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


class CarryoverState(NamedTuple):
    """State for proximal weight carryover transforms."""

    anchor: Any  # theta_bar — previous settled weights (same pytree as params)
    fisher_diag: Any  # diagonal Fisher estimate (Tier 2 only; None for Tier 1)
    step_count: jnp.ndarray  # scalar, steps since last anchor update


# ---------------------------------------------------------------------------
# Tier 1 — Euclidean (proximal / MAP under Gaussian prior)
# ---------------------------------------------------------------------------


def proximal_carryover_euclidean(
    penalty_strength: float,
) -> optax.GradientTransformation:
    """
    Tier 1 proximal weight carryover.

    Adds ``penalty_strength * (params - anchor)`` to every gradient update,
    equivalent to L2 weight decay toward a shifting centre.

    The anchor is **not** updated by this transform — call
    :func:`update_anchor` at the desired cadence boundary.

    Args:
        penalty_strength: Scalar λ > 0 controlling the penalty magnitude.

    Returns:
        An ``optax.GradientTransformation`` composable via ``optax.chain``.
    """
    _validate_penalty_strength(penalty_strength)

    def init_fn(params):
        anchor = jax.tree_util.tree_map(lambda p: p.copy(), params)
        return CarryoverState(
            anchor=anchor,
            fisher_diag=None,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError(
                "proximal_carryover_euclidean requires params in "
                "optimizer.update(grads, state, params)."
            )
        penalty = jax.tree_util.tree_map(
            lambda p, a: penalty_strength * (p - a), params, state.anchor
        )
        new_updates = jax.tree_util.tree_map(
            lambda u, pen: u + pen, updates, penalty
        )
        new_state = CarryoverState(
            anchor=state.anchor,
            fisher_diag=None,
            step_count=state.step_count + 1,
        )
        return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Tier 2 — Fisher-weighted (EWC-like / natural-metric trust region)
# ---------------------------------------------------------------------------


def proximal_carryover_fisher(
    penalty_strength: float,
    fisher_decay: float = 0.95,
    damping: float = 1e-3,
) -> optax.GradientTransformation:
    """
    Tier 2 Fisher-weighted proximal weight carryover.

    Adds ``(diag(F) + damping) * penalty_strength * (params - anchor)`` to
    every gradient update, penalising movement more strongly in directions
    that affect predictions.

    A running diagonal Fisher estimate is maintained via exponential moving
    average of squared gradients (same estimator as in
    :func:`~fabricpc.training.natural_gradients.scale_by_natural_gradient_diag`).

    Args:
        penalty_strength: Scalar λ > 0 controlling the penalty magnitude.
        fisher_decay: EMA decay for the Fisher diagonal, in [0, 1).
        damping: Positive constant added to the Fisher diagonal.

    Returns:
        An ``optax.GradientTransformation`` composable via ``optax.chain``.
    """
    _validate_penalty_strength(penalty_strength)
    _validate_fisher_hparams(fisher_decay, damping)
    one_minus_decay = 1.0 - fisher_decay

    def init_fn(params):
        anchor = jax.tree_util.tree_map(lambda p: p.copy(), params)
        fisher_diag = jax.tree_util.tree_map(jnp.zeros_like, params)
        return CarryoverState(
            anchor=anchor,
            fisher_diag=fisher_diag,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError(
                "proximal_carryover_fisher requires params in "
                "optimizer.update(grads, state, params)."
            )
        # Update running diagonal Fisher estimate
        fisher_diag = jax.tree_util.tree_map(
            lambda f, g: fisher_decay * f + one_minus_decay * jnp.square(g),
            state.fisher_diag,
            updates,
        )
        # Fisher-weighted penalty: (F + damping) * λ * (θ - θ̄)
        penalty = jax.tree_util.tree_map(
            lambda f, p, a: penalty_strength * (f + damping) * (p - a),
            fisher_diag,
            params,
            state.anchor,
        )
        new_updates = jax.tree_util.tree_map(
            lambda u, pen: u + pen, updates, penalty
        )
        new_state = CarryoverState(
            anchor=state.anchor,
            fisher_diag=fisher_diag,
            step_count=state.step_count + 1,
        )
        return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Anchor management
# ---------------------------------------------------------------------------


def update_anchor(carryover_state: CarryoverState, params) -> CarryoverState:
    """
    Snapshot current params as the new anchor θ̄.

    Call this at the desired cadence boundary (per-data-point, per-epoch, etc.)
    from the training loop.

    Args:
        carryover_state: Current ``CarryoverState``.
        params: Current model parameters (same pytree structure as anchor).

    Returns:
        New ``CarryoverState`` with updated anchor and reset step count.
    """
    new_anchor = jax.tree_util.tree_map(lambda p: p.copy(), params)
    return carryover_state._replace(
        anchor=new_anchor,
        step_count=jnp.array(0, dtype=jnp.int32),
    )


def update_anchor_in_chain(
    chain_state: tuple, params, carryover_index: int
) -> tuple:
    """
    Update the carryover anchor inside an ``optax.chain`` state.

    ``optax.chain(t1, t2, ...)`` produces a state that is a tuple of
    sub-states.  This helper extracts element *carryover_index*, calls
    :func:`update_anchor`, and returns a reconstructed chain state.

    Args:
        chain_state: The full optimizer state returned by ``optax.chain(...).init()``.
        params: Current model parameters.
        carryover_index: Position of the carryover transform in the chain.

    Returns:
        Updated chain state with refreshed anchor.
    """
    # optax.chain state may be a plain tuple or a NamedTuple with inner_state
    if hasattr(chain_state, 'inner_state'):
        inner = list(chain_state.inner_state)
        inner[carryover_index] = update_anchor(inner[carryover_index], params)
        return chain_state._replace(inner_state=tuple(inner))
    elif isinstance(chain_state, tuple):
        inner = list(chain_state)
        inner[carryover_index] = update_anchor(inner[carryover_index], params)
        return tuple(inner)
    else:
        raise TypeError(
            f"Unsupported chain state type: {type(chain_state)}. "
            "Expected a tuple or NamedTuple with inner_state."
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_penalty_strength(penalty_strength: float) -> None:
    if penalty_strength <= 0.0:
        raise ValueError(
            f"penalty_strength must be > 0. Got {penalty_strength}"
        )


def _validate_fisher_hparams(fisher_decay: float, damping: float) -> None:
    if not 0.0 <= fisher_decay < 1.0:
        raise ValueError(
            f"fisher_decay must be in [0, 1). Got {fisher_decay}"
        )
    if damping <= 0.0:
        raise ValueError(f"damping must be > 0. Got {damping}")
