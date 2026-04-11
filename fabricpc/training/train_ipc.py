"""
Incremental Predictive Coding (iPC) training loop.

Unlike standard PC (which runs inference to convergence, then updates weights
once), iPC interleaves single inference steps with single weight steps,
creating a coupled dynamical system in (z, θ).

This module provides:
- ``train_step_ipc``: process one batch with K interleaved steps
- ``train_pcn_ipc``: epoch-level training loop

Both support optional proximal weight carryover via the transforms in
``fabricpc.training.carryover``.

Reference: "Proximal Stabilisation for Incremental Predictive Coding" (TPC v0.3)
"""

from typing import Dict, Tuple, Any, cast, List
import math

import jax
import jax.numpy as jnp
import optax

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.core.inference import InferenceBase
from fabricpc.graph.graph_net import compute_local_weight_gradients
from fabricpc.training.carryover import update_anchor_in_chain


# ---------------------------------------------------------------------------
# Single-batch iPC step
# ---------------------------------------------------------------------------


def train_step_ipc(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    ipc_steps: int = 20,
    carryover_index: int = -1,
    anchor_cadence: str = "per_data_point",
) -> Tuple[GraphParams, optax.OptState, float, GraphState]:
    """
    Process one batch with K interleaved inference + weight steps (iPC).

    Algorithm (per data point, per-data-point cadence)::

        initialise latents z from data
        for k = 1 … ipc_steps:
            z  ← z − η_z ∇_z F(z; θ)           # single inference step
            g  ← local Hebbian weight gradients  # from current (non-converged) state
            θ  ← optimizer.step(g)               # includes carryover penalty if present
        θ̄  ← θ                                   # update anchor

    Args:
        params: Current model parameters.
        opt_state: Optimizer state (may include carryover state).
        batch: Batch dict with task-specific keys (e.g. ``{"x": …, "y": …}``).
        structure: Graph structure.
        optimizer: Optax optimizer (possibly ``optax.chain(carryover, adam)``).
        rng_key: JAX random key for state initialisation.
        ipc_steps: Number of interleaved inference + weight steps per batch.
        carryover_index: Position of carryover transform in the optimizer chain.
            Set to ``-1`` (default) when no carryover is used.
        anchor_cadence: When to update the anchor.
            ``"per_data_point"`` (default) — after processing each batch.
            ``"per_epoch"`` — managed by the outer training loop.
            ``"per_step"`` — after every weight update inside the inner loop.

    Returns:
        ``(updated_params, updated_opt_state, energy, final_state)``
    """
    from fabricpc.graph.state_initializer import initialize_graph_state

    batch_size = next(iter(batch.values())).shape[0]

    # Map task names to node names
    clamps = {}
    for task_name, task_value in batch.items():
        if task_name in structure.task_map:
            node_name = structure.task_map[task_name]
            clamps[node_name] = task_value

    # Initialise graph state
    state = initialize_graph_state(
        structure, batch_size, rng_key, clamps=clamps, params=params
    )

    # Retrieve inference config
    inference_obj = structure.config["inference"]
    inference_cls = type(inference_obj)
    config = inference_obj.config

    # --- K interleaved inference + weight steps ---
    for _k in range(ipc_steps):
        # 1. Single inference step (update latents)
        state = inference_cls.inference_step(
            params, state, clamps, structure, config
        )

        # 2. Compute local Hebbian weight gradients from current state
        grads = compute_local_weight_gradients(params, state, structure)

        # 3. Optimizer step (includes carryover penalty if in the chain)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = cast(GraphParams, optax.apply_updates(params, updates))

        # Per-step anchor cadence
        if anchor_cadence == "per_step" and carryover_index >= 0:
            opt_state = update_anchor_in_chain(
                opt_state, params, carryover_index
            )

    # Per-data-point anchor cadence
    if anchor_cadence == "per_data_point" and carryover_index >= 0:
        opt_state = update_anchor_in_chain(
            opt_state, params, carryover_index
        )

    # Compute energy from final state
    energy = sum(
        [
            sum(state.nodes[node_name].energy)
            for node_name in structure.nodes
            if structure.nodes[node_name].node_info.in_degree > 0
        ]
    )

    return params, opt_state, energy, state


# ---------------------------------------------------------------------------
# Epoch-level iPC training loop
# ---------------------------------------------------------------------------


def train_pcn_ipc(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    optimizer: optax.GradientTransformation,
    config: dict,
    rng_key: jax.Array,
    verbose: bool = True,
    epoch_callback=None,
    iter_callback=None,
) -> Tuple[GraphParams, List[Any], List[Any]]:
    """
    Train a predictive coding network using incremental PC (iPC).

    This mirrors the API of :func:`~fabricpc.training.train.train_pcn` but
    uses interleaved inference + weight steps instead of running inference
    to convergence.

    Config keys:
        - ``num_epochs`` (int): Number of training epochs.
        - ``ipc_steps`` (int, default 20): Interleaved steps per batch.
        - ``carryover_index`` (int, default -1): Position of carryover
          transform in the optimizer chain.  ``-1`` means no carryover.
        - ``anchor_cadence`` (str, default ``"per_data_point"``):
          ``"per_data_point"`` | ``"per_epoch"`` | ``"per_step"``.

    Args:
        params: Initial parameters.
        structure: Graph structure.
        train_loader: Data loader yielding batches.
        optimizer: Optax optimizer.
        config: Training configuration dict.
        rng_key: JAX random key.
        verbose: Print epoch summaries.
        epoch_callback: ``(epoch_idx, params, structure, config, rng_key) -> any``
        iter_callback: ``(epoch_idx, batch_idx, energy) -> any``

    Returns:
        ``(trained_params, energy_history, epoch_results)``
    """
    opt_state = optimizer.init(params)

    num_epochs = config.get("num_epochs", 10)
    ipc_steps = config.get("ipc_steps", 20)
    carryover_index = config.get("carryover_index", -1)
    anchor_cadence = config.get("anchor_cadence", "per_data_point")

    total_epochs = math.ceil(num_epochs)
    frac = num_epochs - math.floor(num_epochs)

    # JIT-compile the inner step
    jit_train_step = jax.jit(
        lambda p, o, b, k: train_step_ipc(
            p, o, b, structure, optimizer, k,
            ipc_steps=ipc_steps,
            carryover_index=carryover_index,
            anchor_cadence=anchor_cadence,
        )
    )

    iter_results: List[Any] = []
    epoch_results: List[Any] = []

    for epoch_idx in range(total_epochs):
        try:
            num_batches = len(train_loader)
        except TypeError:
            raise TypeError("train_loader must support len()")

        is_last_epoch = epoch_idx == total_epochs - 1
        if is_last_epoch and frac > 0:
            max_batches = round(frac * num_batches)
        else:
            max_batches = num_batches

        epoch_rng_key, rng_key = jax.random.split(rng_key)
        batch_keys = jax.random.split(epoch_rng_key, max_batches)

        batch_energies: List[float] = []
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            # Convert batch
            if isinstance(batch_data, (list, tuple)):
                batch = {
                    "x": jnp.array(batch_data[0]),
                    "y": jnp.array(batch_data[1]),
                }
            elif isinstance(batch_data, dict):
                batch = {k: jnp.array(v) for k, v in batch_data.items()}
            else:
                raise ValueError(f"Unsupported batch format: {type(batch_data)}")

            params, opt_state, energy, _ = jit_train_step(
                params, opt_state, batch, batch_keys[batch_idx]
            )

            if iter_callback is not None:
                batch_energies.append(iter_callback(epoch_idx, batch_idx, energy))
            else:
                batch_energies.append(
                    float(energy) / next(iter(batch.values())).shape[0]
                )

        iter_results.append(batch_energies)

        avg_energy = (
            sum(batch_energies) / len(batch_energies) if batch_energies else 0.0
        )

        # Per-epoch anchor cadence
        if anchor_cadence == "per_epoch" and carryover_index >= 0:
            opt_state = update_anchor_in_chain(
                opt_state, params, carryover_index
            )

        epoch_results.append(
            epoch_callback(epoch_idx, params, structure, config, rng_key)
            if epoch_callback
            else None
        )

        if verbose:
            print(f"Epoch {epoch_idx + 1}/{total_epochs}, energy: {avg_energy:.4f}")

    return params, iter_results, epoch_results
