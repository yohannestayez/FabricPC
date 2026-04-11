"""
Incremental PC with Temporal Carryover — KMNIST Experiments
============================================================

Six experiments validating the Temporal Carryover framework on KMNIST
(harder than MNIST, same 28x28 grayscale format).

Phase 1 (quick, ~10-15 min each):
  A. **Comparison** — Vanilla iPC vs Tier 1 vs Tier 2
  D. **Lambda sweep** — penalty strength sensitivity
  E. **Cadence** — per-step vs per-data-point vs per-epoch anchor
  F. **K sensitivity** — iPC steps interaction with carryover

Phase 2 (expensive, ~20-30 min each):
  B. **Stability map** — (eta_infer, lr) grid, continuous heatmaps
  C. **Warm-start ablation** — Standard PC vs Vanilla iPC vs iPC+Tether

Usage::

    python examples/kmnist_ipc_carryover_demo.py --experiment all
    python examples/kmnist_ipc_carryover_demo.py --experiment phase1
    python examples/kmnist_ipc_carryover_demo.py --experiment phase2
    python examples/kmnist_ipc_carryover_demo.py --experiment comparison
    python examples/kmnist_ipc_carryover_demo.py --experiment lambda_sweep
    python examples/kmnist_ipc_carryover_demo.py --experiment cadence
    python examples/kmnist_ipc_carryover_demo.py --experiment k_sensitivity
    python examples/kmnist_ipc_carryover_demo.py --experiment stability
    python examples/kmnist_ipc_carryover_demo.py --experiment ablation

Reference: "Proximal Stabilisation for Incremental Predictive Coding" (TPC v0.3)
"""

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cuda")

import argparse
import os
import time
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation, ReLUActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer
from fabricpc.core.types import GraphParams
from fabricpc.training import train_pcn_ipc, train_pcn, evaluate_pcn
from fabricpc.training.carryover import (
    proximal_carryover_euclidean,
    proximal_carryover_fisher,
)
from fabricpc.utils.data.dataloader import KmnistLoader
from fabricpc.utils.dashboarding.extractors import extract_weight_statistics

jax.config.update("jax_default_prng_impl", "threefry2x32")

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

# KMNIST normalization stats
KMNIST_MEAN = 0.1917
KMNIST_STD = 0.3483


# =====================================================================
# Metrics collection
# =====================================================================


def compute_param_drift(params_a: GraphParams, params_b: GraphParams) -> Dict[str, float]:
    """Compute per-layer L2 norm of (params_a - params_b)."""
    result = {}
    for node_name in params_a.nodes:
        node_a = params_a.nodes[node_name]
        node_b = params_b.nodes[node_name]
        for edge_key in node_a.weights:
            diff = node_a.weights[edge_key] - node_b.weights[edge_key]
            result[f"{node_name}/{edge_key}"] = float(jnp.linalg.norm(diff))
    return result


def total_drift(drift_dict: Dict[str, float]) -> float:
    """Sum all per-layer drift values."""
    return sum(drift_dict.values())


class MetricsCollector:
    """Collects per-epoch metrics without modifying the training loop.

    Uses the epoch_callback API to gather accuracy, weight norms, drift,
    and per-epoch weight deltas. All stored in plain lists for plotting.
    """

    def __init__(self, test_loader, initial_params: GraphParams):
        self.test_loader = test_loader
        self.initial_params = initial_params
        self.prev_params = initial_params

        # Storage
        self.epoch_accuracies: List[float] = []
        self.epoch_weight_norms: List[Dict[str, float]] = []
        self.epoch_weight_drift: List[Dict[str, float]] = []
        self.epoch_weight_delta: List[Dict[str, float]] = []

    def make_epoch_callback(self):
        def epoch_callback(epoch_idx, params, structure, config, rng_key):
            # 1. Accuracy
            metrics = evaluate_pcn(params, structure, self.test_loader, config, rng_key)
            acc = metrics["accuracy"]
            self.epoch_accuracies.append(acc)

            # 2. Per-layer weight norms
            w_stats = extract_weight_statistics(params)
            norms = {}
            for node, edges in w_stats.items():
                for edge, stats in edges.items():
                    norms[f"{node}/{edge}"] = stats["norm"]
            self.epoch_weight_norms.append(norms)

            # 3. Drift from initial params
            drift = compute_param_drift(params, self.initial_params)
            self.epoch_weight_drift.append(drift)

            # 4. Delta from previous epoch
            delta = compute_param_drift(params, self.prev_params)
            self.epoch_weight_delta.append(delta)
            self.prev_params = jax.tree_util.tree_map(lambda p: p.copy(), params)

            return {"accuracy": acc}

        return epoch_callback

    @staticmethod
    def compute_batch_energy_stats(iter_results: List[List[float]]) -> List[Dict[str, float]]:
        """Compute per-epoch energy statistics from iter_results."""
        stats = []
        for epoch_energies in iter_results:
            arr = np.array(epoch_energies)
            mean = float(np.mean(arr))
            stats.append({
                "mean": mean,
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "cv": float(np.std(arr) / mean) if mean > 0 else 0.0,
            })
        return stats


# =====================================================================
# Shared helpers
# =====================================================================


def build_structure(eta_infer=0.05, infer_steps=20):
    """Build the 3-layer PC network used across all experiments."""
    pixels = IdentityNode(shape=(784,), name="pixels")
    hidden1 = Linear(
        shape=(256,),
        activation=ReLUActivation(),
        name="hidden1",
        weight_init=XavierInitializer(),
    )
    hidden2 = Linear(
        shape=(64,),
        activation=ReLUActivation(),
        name="hidden2",
        weight_init=XavierInitializer(),
    )
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
        weight_init=XavierInitializer(),
    )
    return graph(
        nodes=[pixels, hidden1, hidden2, output],
        edges=[
            Edge(source=pixels, target=hidden1.slot("in")),
            Edge(source=hidden1, target=hidden2.slot("in")),
            Edge(source=hidden2, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
    )


def get_loaders(batch_size=200):
    train_loader = KmnistLoader(
        "train", batch_size=batch_size, tensor_format="flat",
        shuffle=True, seed=42,
        normalize_mean=KMNIST_MEAN, normalize_std=KMNIST_STD,
    )
    test_loader = KmnistLoader(
        "test", batch_size=batch_size, tensor_format="flat",
        shuffle=False,
        normalize_mean=KMNIST_MEAN, normalize_std=KMNIST_STD,
    )
    return train_loader, test_loader


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  [matplotlib not installed — skipping plot]")
        return None


def _run_ipc_with_metrics(
    params, structure, train_loader, test_loader, optimizer, config, rng_key,
):
    """Run train_pcn_ipc with MetricsCollector attached."""
    collector = MetricsCollector(test_loader, params)
    trained_params, iter_results, epoch_results = train_pcn_ipc(
        params=params, structure=structure, train_loader=train_loader,
        optimizer=optimizer, config=config, rng_key=rng_key, verbose=True,
        epoch_callback=collector.make_epoch_callback(),
    )
    energy_stats = MetricsCollector.compute_batch_energy_stats(iter_results)
    epoch_energies = [s["mean"] for s in energy_stats]
    return trained_params, collector, epoch_energies, energy_stats


# =====================================================================
# Experiment A: Comparison (Vanilla iPC vs Tier 1 vs Tier 2)
# =====================================================================


def run_comparison():
    print("\n" + "=" * 70)
    print("  EXPERIMENT A: Condition Comparison (KMNIST)")
    print("=" * 70)

    NUM_EPOCHS = 10
    BATCH_SIZE = 200
    LR = 0.001
    IPC_STEPS = 20
    PENALTY = 0.1

    train_loader, test_loader = get_loaders(BATCH_SIZE)
    master_key = jax.random.PRNGKey(0)

    conditions = [
        ("Vanilla iPC", optax.adamw(LR, weight_decay=0.1),
         {"num_epochs": NUM_EPOCHS, "ipc_steps": IPC_STEPS, "carryover_index": -1}),
        ("iPC + Tier 1",
         optax.chain(proximal_carryover_euclidean(PENALTY), optax.adamw(LR, weight_decay=0.1)),
         {"num_epochs": NUM_EPOCHS, "ipc_steps": IPC_STEPS, "carryover_index": 0,
          "anchor_cadence": "per_data_point"}),
        ("iPC + Tier 2",
         optax.chain(proximal_carryover_fisher(PENALTY), optax.adamw(LR, weight_decay=0.1)),
         {"num_epochs": NUM_EPOCHS, "ipc_steps": IPC_STEPS, "carryover_index": 0,
          "anchor_cadence": "per_data_point"}),
    ]

    all_results = []

    for cond_name, optimizer, config in conditions:
        print(f"\n--- {cond_name} ---")
        structure = build_structure()
        gk, tk, master_key = jax.random.split(master_key, 3)
        params = initialize_params(structure, gk)

        start = time.time()
        trained_params, collector, epoch_energies, energy_stats = _run_ipc_with_metrics(
            params, structure, train_loader, test_loader, optimizer, config, tk,
        )
        elapsed = time.time() - start

        all_results.append({
            "name": cond_name,
            "accuracies": collector.epoch_accuracies,
            "epoch_energies": epoch_energies,
            "energy_stats": energy_stats,
            "weight_norms": collector.epoch_weight_norms,
            "weight_drift": collector.epoch_weight_drift,
            "weight_delta": collector.epoch_weight_delta,
            "time": elapsed,
        })

    # --- Summary table ---
    print(f"\n{'Condition':<20} {'Acc@1':>7} {'Acc@5':>7} {'Acc@10':>7} "
          f"{'Final E':>9} {'Drift':>8} {'dW/ep':>8} {'Time':>7}")
    print("-" * 85)
    for r in all_results:
        acc1 = r["accuracies"][0] * 100 if len(r["accuracies"]) > 0 else 0
        acc5 = r["accuracies"][4] * 100 if len(r["accuracies"]) > 4 else 0
        acc10 = r["accuracies"][-1] * 100
        drift = total_drift(r["weight_drift"][-1])
        avg_delta = np.mean([total_drift(d) for d in r["weight_delta"]])
        print(f"{r['name']:<20} {acc1:>6.2f}% {acc5:>6.2f}% {acc10:>6.2f}% "
              f"{r['epoch_energies'][-1]:>9.4f} {drift:>8.2f} {avg_delta:>8.3f} "
              f"{r['time']:>6.1f}s")

    # --- Plot ---
    _plot_comparison(all_results)

    return all_results


def _plot_comparison(results):
    plt = _import_matplotlib()
    if plt is None:
        return

    ensure_plot_dir()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Top-left: Energy curves with std band
    ax = axes[0, 0]
    for r, c in zip(results, colors):
        epochs = list(range(1, len(r["epoch_energies"]) + 1))
        means = [s["mean"] for s in r["energy_stats"]]
        stds = [s["std"] for s in r["energy_stats"]]
        ax.plot(epochs, means, marker="o", color=c, label=r["name"], markersize=4)
        ax.fill_between(epochs,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=c)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy (per sample)")
    ax.set_title("Energy Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: Accuracy curves
    ax = axes[0, 1]
    for r, c in zip(results, colors):
        epochs = list(range(1, len(r["accuracies"]) + 1))
        ax.plot(epochs, [a * 100 for a in r["accuracies"]],
                marker="o", color=c, label=r["name"], markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Learning Trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Per-layer weight norm evolution
    ax = axes[1, 0]
    if results:
        layer_names = list(results[0]["weight_norms"][0].keys())
        for li, layer in enumerate(layer_names):
            for r, c in zip(results, colors):
                epochs = list(range(1, len(r["weight_norms"]) + 1))
                norms = [wn[layer] for wn in r["weight_norms"]]
                linestyle = ["-", "--", ":"][li % 3]
                ax.plot(epochs, norms, color=c, linestyle=linestyle,
                        label=f"{r['name']} / {layer.split('/')[-1]}" if li == 0 or c == colors[0] else "",
                        alpha=0.8, markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight L2 Norm")
    ax.set_title("Per-Layer Weight Norms")
    # Simplified legend: just show conditions
    handles = [plt.Line2D([0], [0], color=c, label=r["name"])
               for r, c in zip(results, colors)]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Total weight drift from init
    ax = axes[1, 1]
    for r, c in zip(results, colors):
        epochs = list(range(1, len(r["weight_drift"]) + 1))
        drifts = [total_drift(d) for d in r["weight_drift"]]
        ax.plot(epochs, drifts, marker="o", color=c, label=r["name"], markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Weight Drift ||θ - θ₀||")
    ax.set_title("Parameter Travel from Init")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("KMNIST iPC Comparison: Vanilla vs Carryover", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "kmnist_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


# =====================================================================
# Experiment B: Stability Map
# =====================================================================


def run_stability_map():
    print("\n" + "=" * 70)
    print("  EXPERIMENT B: Stability Map (KMNIST)")
    print("=" * 70)

    ETA_INFERS = [0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
    LRS = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    PENALTY = 0.1
    IPC_STEPS = 10
    NUM_EPOCHS = 5
    BATCH_SIZE = 200

    train_loader, test_loader = get_loaders(BATCH_SIZE)

    def make_optimizer(cond_name, lr):
        base = optax.adamw(lr, weight_decay=0.1)
        if cond_name == "Vanilla iPC":
            return base, -1
        elif cond_name == "Tier 1":
            return optax.chain(proximal_carryover_euclidean(PENALTY), base), 0
        elif cond_name == "Tier 2":
            return optax.chain(proximal_carryover_fisher(PENALTY), base), 0

    cond_names = ["Vanilla iPC", "Tier 1", "Tier 2"]
    energy_grids = {}
    acc_grids = {}

    total_runs = len(cond_names) * len(ETA_INFERS) * len(LRS)
    run_idx = 0

    for cond_name in cond_names:
        e_grid = np.full((len(ETA_INFERS), len(LRS)), np.nan)
        a_grid = np.full((len(ETA_INFERS), len(LRS)), np.nan)

        for i, eta in enumerate(ETA_INFERS):
            for j, lr in enumerate(LRS):
                run_idx += 1
                key = jax.random.PRNGKey(42)
                gk, tk, ek = jax.random.split(key, 3)

                structure = build_structure(eta_infer=eta, infer_steps=IPC_STEPS)
                optimizer, co_idx = make_optimizer(cond_name, lr)
                config = {
                    "num_epochs": NUM_EPOCHS,
                    "ipc_steps": IPC_STEPS,
                    "carryover_index": co_idx,
                    "anchor_cadence": "per_data_point",
                }

                try:
                    params = initialize_params(structure, gk)
                    trained_params, energy_history, _ = train_pcn_ipc(
                        params=params, structure=structure,
                        train_loader=train_loader, optimizer=optimizer,
                        config=config, rng_key=tk, verbose=False,
                    )
                    last_epoch = energy_history[-1] if energy_history else []
                    if last_epoch and all(np.isfinite(e) for e in last_epoch):
                        e_grid[i, j] = np.mean(last_epoch)
                        # Evaluate accuracy
                        metrics = evaluate_pcn(
                            trained_params, structure, test_loader, config, ek
                        )
                        a_grid[i, j] = metrics["accuracy"] * 100
                except Exception:
                    pass  # stays NaN

                status = f"E={e_grid[i,j]:.4f}" if np.isfinite(e_grid[i,j]) else "DIVERGED"
                print(f"  [{run_idx}/{total_runs}] {cond_name} "
                      f"eta={eta} lr={lr} -> {status}")

        energy_grids[cond_name] = e_grid
        acc_grids[cond_name] = a_grid

    # --- Summary ---
    for cond_name in cond_names:
        eg = energy_grids[cond_name]
        ag = acc_grids[cond_name]
        stable = np.sum(np.isfinite(eg))
        print(f"\n  {cond_name}: {int(stable)}/{eg.size} stable, "
              f"avg energy={np.nanmean(eg):.4f}, avg acc={np.nanmean(ag):.1f}%")

    # --- Plot ---
    _plot_stability_map(energy_grids, acc_grids, ETA_INFERS, LRS)

    return energy_grids, acc_grids


def _plot_stability_map(energy_grids, acc_grids, eta_infers, lrs):
    plt = _import_matplotlib()
    if plt is None:
        return

    ensure_plot_dir()
    cond_names = list(energy_grids.keys())
    fig, axes = plt.subplots(2, len(cond_names), figsize=(5.5 * len(cond_names), 9))

    eta_labels = [str(e) for e in eta_infers]
    lr_labels = [str(l) for l in lrs]

    # Top row: energy heatmaps
    all_energies = np.concatenate([energy_grids[c].flatten() for c in cond_names])
    e_vmin = np.nanmin(all_energies) if np.any(np.isfinite(all_energies)) else 0
    e_vmax = np.nanmax(all_energies) if np.any(np.isfinite(all_energies)) else 1

    for col, cond_name in enumerate(cond_names):
        ax = axes[0, col]
        grid = energy_grids[cond_name].copy()
        masked = np.ma.masked_invalid(grid)

        im = ax.imshow(masked, cmap="viridis_r", vmin=e_vmin, vmax=e_vmax,
                       aspect="auto", origin="lower")
        # Mark NaN cells in red
        nan_mask = np.isnan(grid)
        if nan_mask.any():
            ax.imshow(np.where(nan_mask, 1.0, np.nan), cmap="Reds",
                      vmin=0, vmax=1, aspect="auto", origin="lower", alpha=0.7)

        ax.set_xticks(range(len(lr_labels)))
        ax.set_xticklabels(lr_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(eta_labels)))
        ax.set_yticklabels(eta_labels, fontsize=7)
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Inference rate (eta)")
        stable = int(np.sum(np.isfinite(grid)))
        ax.set_title(f"{cond_name}\nEnergy ({stable}/{grid.size} stable)", fontsize=10)

        for ii in range(grid.shape[0]):
            for jj in range(grid.shape[1]):
                val = grid[ii, jj]
                if np.isfinite(val):
                    ax.text(jj, ii, f"{val:.3f}", ha="center", va="center",
                            fontsize=6, color="white" if val > (e_vmin + e_vmax) / 2 else "black")
                else:
                    ax.text(jj, ii, "X", ha="center", va="center",
                            fontsize=8, fontweight="bold", color="white")

    fig.colorbar(im, ax=axes[0, :].tolist(), label="Final Energy", shrink=0.8)

    # Bottom row: accuracy heatmaps
    all_accs = np.concatenate([acc_grids[c].flatten() for c in cond_names])
    a_vmin = np.nanmin(all_accs) if np.any(np.isfinite(all_accs)) else 0
    a_vmax = np.nanmax(all_accs) if np.any(np.isfinite(all_accs)) else 100

    for col, cond_name in enumerate(cond_names):
        ax = axes[1, col]
        grid = acc_grids[cond_name].copy()
        masked = np.ma.masked_invalid(grid)

        im2 = ax.imshow(masked, cmap="RdYlGn", vmin=a_vmin, vmax=a_vmax,
                        aspect="auto", origin="lower")
        nan_mask = np.isnan(grid)
        if nan_mask.any():
            ax.imshow(np.where(nan_mask, 1.0, np.nan), cmap="Reds",
                      vmin=0, vmax=1, aspect="auto", origin="lower", alpha=0.7)

        ax.set_xticks(range(len(lr_labels)))
        ax.set_xticklabels(lr_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(eta_labels)))
        ax.set_yticklabels(eta_labels, fontsize=7)
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Inference rate (eta)")
        ax.set_title(f"{cond_name}\nAccuracy (%)", fontsize=10)

        for ii in range(grid.shape[0]):
            for jj in range(grid.shape[1]):
                val = grid[ii, jj]
                if np.isfinite(val):
                    ax.text(jj, ii, f"{val:.0f}", ha="center", va="center",
                            fontsize=7, color="black")
                else:
                    ax.text(jj, ii, "X", ha="center", va="center",
                            fontsize=8, fontweight="bold", color="white")

    fig.colorbar(im2, ax=axes[1, :].tolist(), label="Test Accuracy (%)", shrink=0.8)

    fig.suptitle("KMNIST Stability Map: Energy & Accuracy", fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "kmnist_stability_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


# =====================================================================
# Experiment C: Warm-Start Ablation
# =====================================================================


def run_ablation():
    print("\n" + "=" * 70)
    print("  EXPERIMENT C: Warm-Start Ablation (KMNIST)")
    print("=" * 70)

    NUM_EPOCHS = 10
    BATCH_SIZE = 200
    LR = 0.005
    IPC_STEPS = 20
    PENALTY = 0.1

    train_loader, test_loader = get_loaders(BATCH_SIZE)
    master_key = jax.random.PRNGKey(0)

    ablation_results = []

    # --- Condition 1: Standard PC ---
    print("\n--- Standard PC (baseline) ---")
    structure = build_structure()
    gk, tk, ek, master_key = jax.random.split(master_key, 4)
    params = initialize_params(structure, gk)
    optimizer = optax.adamw(LR, weight_decay=0.1)

    collector = MetricsCollector(test_loader, params)
    start = time.time()
    trained_params, energy_history, _ = train_pcn(
        params=params, structure=structure, train_loader=train_loader,
        optimizer=optimizer, config={"num_epochs": NUM_EPOCHS},
        rng_key=tk, verbose=True,
        epoch_callback=collector.make_epoch_callback(),
    )
    elapsed = time.time() - start
    epoch_energies = [
        sum(be) / len(be) if be else float("nan") for be in energy_history
    ]
    energy_stats = MetricsCollector.compute_batch_energy_stats(energy_history)
    ablation_results.append({
        "name": "Standard PC",
        "accuracies": collector.epoch_accuracies,
        "epoch_energies": epoch_energies,
        "energy_stats": energy_stats,
        "weight_drift": collector.epoch_weight_drift,
        "weight_delta": collector.epoch_weight_delta,
        "time": elapsed,
    })
    print(f"  Final accuracy: {collector.epoch_accuracies[-1]*100:.2f}%")

    # --- Condition 2: Vanilla iPC ---
    print("\n--- Vanilla iPC (warm start only) ---")
    structure = build_structure()
    gk, tk, ek, master_key = jax.random.split(master_key, 4)
    params = initialize_params(structure, gk)
    optimizer = optax.adamw(LR, weight_decay=0.1)

    start = time.time()
    trained_params, collector, epoch_energies, energy_stats = _run_ipc_with_metrics(
        params, structure, train_loader, test_loader, optimizer,
        {"num_epochs": NUM_EPOCHS, "ipc_steps": IPC_STEPS, "carryover_index": -1},
        tk,
    )
    elapsed = time.time() - start
    ablation_results.append({
        "name": "Vanilla iPC",
        "accuracies": collector.epoch_accuracies,
        "epoch_energies": epoch_energies,
        "energy_stats": energy_stats,
        "weight_drift": collector.epoch_weight_drift,
        "weight_delta": collector.epoch_weight_delta,
        "time": elapsed,
    })
    print(f"  Final accuracy: {collector.epoch_accuracies[-1]*100:.2f}%")

    # --- Condition 3: iPC + Tether ---
    print("\n--- iPC + Tier 1 (warm start + tether) ---")
    structure = build_structure()
    gk, tk, ek, master_key = jax.random.split(master_key, 4)
    params = initialize_params(structure, gk)
    optimizer = optax.chain(
        proximal_carryover_euclidean(PENALTY),
        optax.adamw(LR, weight_decay=0.1),
    )

    start = time.time()
    trained_params, collector, epoch_energies, energy_stats = _run_ipc_with_metrics(
        params, structure, train_loader, test_loader, optimizer,
        {"num_epochs": NUM_EPOCHS, "ipc_steps": IPC_STEPS,
         "carryover_index": 0, "anchor_cadence": "per_data_point"},
        tk,
    )
    elapsed = time.time() - start
    ablation_results.append({
        "name": "iPC + Tether",
        "accuracies": collector.epoch_accuracies,
        "epoch_energies": epoch_energies,
        "energy_stats": energy_stats,
        "weight_drift": collector.epoch_weight_drift,
        "weight_delta": collector.epoch_weight_delta,
        "time": elapsed,
    })
    print(f"  Final accuracy: {collector.epoch_accuracies[-1]*100:.2f}%")

    # --- Summary ---
    print(f"\n{'Condition':<20} {'Acc@10':>8} {'Final E':>9} {'Drift':>8} {'Time':>7}")
    print("-" * 56)
    for r in ablation_results:
        acc = r["accuracies"][-1] * 100
        fe = r["epoch_energies"][-1]
        drift = total_drift(r["weight_drift"][-1])
        print(f"{r['name']:<20} {acc:>7.2f}% {fe:>9.4f} {drift:>8.2f} {r['time']:>6.1f}s")

    # --- Plot ---
    _plot_ablation(ablation_results)

    return ablation_results


def _plot_ablation(results):
    plt = _import_matplotlib()
    if plt is None:
        return

    ensure_plot_dir()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ["#78909c", "#42a5f5", "#66bb6a"]

    # Top-left: Accuracy bar chart
    ax = axes[0, 0]
    names = [r["name"] for r in results]
    accs = [r["accuracies"][-1] * 100 for r in results]
    bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Final Accuracy")
    ax.set_ylim(min(accs) - 3, max(accs) + 3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=9)

    # Top-center: Energy curves
    ax = axes[0, 1]
    for r, c in zip(results, colors):
        epochs = list(range(1, len(r["epoch_energies"]) + 1))
        ax.plot(epochs, r["epoch_energies"], marker="o", color=c,
                label=r["name"], markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy (per sample)")
    ax.set_title("Energy Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: Accuracy curves
    ax = axes[0, 2]
    for r, c in zip(results, colors):
        epochs = list(range(1, len(r["accuracies"]) + 1))
        ax.plot(epochs, [a * 100 for a in r["accuracies"]],
                marker="o", color=c, label=r["name"], markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy Over Epochs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Per-epoch energy variance (smoothness)
    ax = axes[1, 0]
    for r, c in zip(results, colors):
        epochs = list(range(1, len(r["energy_stats"]) + 1))
        stds = [s["std"] for s in r["energy_stats"]]
        ax.plot(epochs, stds, marker="o", color=c, label=r["name"], markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy Std (within epoch)")
    ax.set_title("Training Smoothness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-center: Weight drift from init
    ax = axes[1, 1]
    for r, c in zip(results, colors):
        epochs = list(range(1, len(r["weight_drift"]) + 1))
        drifts = [total_drift(d) for d in r["weight_drift"]]
        ax.plot(epochs, drifts, marker="o", color=c, label=r["name"], markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||θ - θ₀||")
    ax.set_title("Weight Drift from Init")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Per-epoch weight delta
    ax = axes[1, 2]
    for r, c in zip(results, colors):
        epochs = list(range(1, len(r["weight_delta"]) + 1))
        deltas = [total_drift(d) for d in r["weight_delta"]]
        ax.plot(epochs, deltas, marker="o", color=c, label=r["name"], markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||θₙ - θₙ₋₁||")
    ax.set_title("Per-Epoch Weight Delta")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("KMNIST Warm-Start Ablation", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "kmnist_ablation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


# =====================================================================
# Experiment D: Lambda (λ) Sensitivity Sweep
# =====================================================================


def run_lambda_sweep():
    print("\n" + "=" * 70)
    print("  EXPERIMENT D: Lambda Sensitivity Sweep (KMNIST)")
    print("=" * 70)

    NUM_EPOCHS = 10
    BATCH_SIZE = 200
    LR = 0.001
    IPC_STEPS = 20
    LAMBDAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    train_loader, test_loader = get_loaders(BATCH_SIZE)
    master_key = jax.random.PRNGKey(0)

    all_results = []

    # --- Baseline: Vanilla iPC (no carryover) ---
    print("\n--- Baseline: Vanilla iPC ---")
    structure = build_structure()
    gk, tk, master_key = jax.random.split(master_key, 3)
    params = initialize_params(structure, gk)
    optimizer = optax.adamw(LR, weight_decay=0.1)

    start = time.time()
    trained_params, collector, epoch_energies, energy_stats = _run_ipc_with_metrics(
        params, structure, train_loader, test_loader, optimizer,
        {"num_epochs": NUM_EPOCHS, "ipc_steps": IPC_STEPS, "carryover_index": -1},
        tk,
    )
    elapsed = time.time() - start

    baseline_result = {
        "lambda": 0.0,
        "label": "Vanilla (λ=0)",
        "accuracies": collector.epoch_accuracies,
        "epoch_energies": epoch_energies,
        "weight_drift": collector.epoch_weight_drift,
        "time": elapsed,
    }
    all_results.append(baseline_result)
    print(f"  Accuracy: {collector.epoch_accuracies[-1]*100:.2f}%")

    # --- Sweep λ values ---
    for lam in LAMBDAS:
        print(f"\n--- Tier 1: λ={lam} ---")
        structure = build_structure()
        gk, tk, master_key = jax.random.split(master_key, 3)
        params = initialize_params(structure, gk)
        optimizer = optax.chain(
            proximal_carryover_euclidean(lam),
            optax.adamw(LR, weight_decay=0.1),
        )

        start = time.time()
        trained_params, collector, epoch_energies, energy_stats = _run_ipc_with_metrics(
            params, structure, train_loader, test_loader, optimizer,
            {"num_epochs": NUM_EPOCHS, "ipc_steps": IPC_STEPS,
             "carryover_index": 0, "anchor_cadence": "per_data_point"},
            tk,
        )
        elapsed = time.time() - start

        all_results.append({
            "lambda": lam,
            "label": f"λ={lam}",
            "accuracies": collector.epoch_accuracies,
            "epoch_energies": epoch_energies,
            "weight_drift": collector.epoch_weight_drift,
            "time": elapsed,
        })
        print(f"  Accuracy: {collector.epoch_accuracies[-1]*100:.2f}%")

    # --- Summary table ---
    print(f"\n{'Lambda':>10} {'Final Acc':>10} {'Final E':>10} {'Drift':>10} {'Time':>8}")
    print("-" * 52)
    for r in all_results:
        acc = r["accuracies"][-1] * 100
        fe = r["epoch_energies"][-1]
        drift = total_drift(r["weight_drift"][-1])
        print(f"{r['label']:>10} {acc:>9.2f}% {fe:>10.4f} {drift:>10.2f} {r['time']:>7.1f}s")

    # --- Plot ---
    _plot_lambda_sweep(all_results, baseline_result)

    return all_results


def _plot_lambda_sweep(results, baseline):
    plt = _import_matplotlib()
    if plt is None:
        return

    ensure_plot_dir()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Separate baseline from sweep
    sweep = [r for r in results if r["lambda"] > 0]
    lambdas = [r["lambda"] for r in sweep]
    baseline_acc = baseline["accuracies"][-1] * 100
    baseline_energy = baseline["epoch_energies"][-1]
    baseline_drift = total_drift(baseline["weight_drift"][-1])

    # Top-left: Final accuracy vs λ
    ax = axes[0, 0]
    accs = [r["accuracies"][-1] * 100 for r in sweep]
    ax.semilogx(lambdas, accs, marker="o", color="#2ca02c", linewidth=2)
    ax.axhline(baseline_acc, color="#1f77b4", linestyle="--",
               label=f"Vanilla iPC ({baseline_acc:.1f}%)")
    ax.set_xlabel("Penalty Strength (λ)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Accuracy vs λ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: Final energy vs λ
    ax = axes[0, 1]
    energies = [r["epoch_energies"][-1] for r in sweep]
    ax.semilogx(lambdas, energies, marker="o", color="#ff7f0e", linewidth=2)
    ax.axhline(baseline_energy, color="#1f77b4", linestyle="--",
               label=f"Vanilla iPC ({baseline_energy:.4f})")
    ax.set_xlabel("Penalty Strength (λ)")
    ax.set_ylabel("Final Energy")
    ax.set_title("Energy vs λ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Total weight drift vs λ
    ax = axes[1, 0]
    drifts = [total_drift(r["weight_drift"][-1]) for r in sweep]
    ax.semilogx(lambdas, drifts, marker="o", color="#d62728", linewidth=2)
    ax.axhline(baseline_drift, color="#1f77b4", linestyle="--",
               label=f"Vanilla iPC ({baseline_drift:.1f})")
    ax.set_xlabel("Penalty Strength (λ)")
    ax.set_ylabel("Total Weight Drift ||θ - θ₀||")
    ax.set_title("Weight Drift vs λ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Accuracy curves for all λ values
    ax = axes[1, 1]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(sweep)))
    # Plot baseline
    epochs = list(range(1, len(baseline["accuracies"]) + 1))
    ax.plot(epochs, [a * 100 for a in baseline["accuracies"]],
            color="black", linestyle="--", linewidth=2, label="Vanilla", alpha=0.7)
    for r, c in zip(sweep, cmap):
        epochs = list(range(1, len(r["accuracies"]) + 1))
        ax.plot(epochs, [a * 100 for a in r["accuracies"]],
                color=c, label=r["label"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Learning Trajectories by λ")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("KMNIST Lambda Sensitivity (Tier 1 Euclidean)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "kmnist_lambda_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


# =====================================================================
# Experiment E: Anchor Cadence Comparison
# =====================================================================


def run_cadence():
    print("\n" + "=" * 70)
    print("  EXPERIMENT E: Anchor Cadence Comparison (KMNIST)")
    print("=" * 70)

    NUM_EPOCHS = 10
    BATCH_SIZE = 200
    LR = 0.001
    IPC_STEPS = 20
    PENALTY = 0.1

    train_loader, test_loader = get_loaders(BATCH_SIZE)
    master_key = jax.random.PRNGKey(0)

    cadences = ["per_step", "per_data_point", "per_epoch"]
    all_results = []

    for cadence in cadences:
        print(f"\n--- Cadence: {cadence} ---")
        structure = build_structure()
        gk, tk, master_key = jax.random.split(master_key, 3)
        params = initialize_params(structure, gk)
        optimizer = optax.chain(
            proximal_carryover_euclidean(PENALTY),
            optax.adamw(LR, weight_decay=0.1),
        )

        start = time.time()
        trained_params, collector, epoch_energies, energy_stats = _run_ipc_with_metrics(
            params, structure, train_loader, test_loader, optimizer,
            {"num_epochs": NUM_EPOCHS, "ipc_steps": IPC_STEPS,
             "carryover_index": 0, "anchor_cadence": cadence},
            tk,
        )
        elapsed = time.time() - start

        all_results.append({
            "name": cadence,
            "accuracies": collector.epoch_accuracies,
            "epoch_energies": epoch_energies,
            "energy_stats": energy_stats,
            "weight_drift": collector.epoch_weight_drift,
            "weight_delta": collector.epoch_weight_delta,
            "time": elapsed,
        })
        print(f"  Accuracy: {collector.epoch_accuracies[-1]*100:.2f}%")

    # --- Summary ---
    print(f"\n{'Cadence':<20} {'Final Acc':>10} {'Final E':>10} "
          f"{'Drift':>8} {'dW/ep':>8} {'Time':>7}")
    print("-" * 68)
    for r in all_results:
        acc = r["accuracies"][-1] * 100
        fe = r["epoch_energies"][-1]
        drift = total_drift(r["weight_drift"][-1])
        avg_delta = np.mean([total_drift(d) for d in r["weight_delta"]])
        print(f"{r['name']:<20} {acc:>9.2f}% {fe:>10.4f} "
              f"{drift:>8.2f} {avg_delta:>8.3f} {r['time']:>6.1f}s")

    # --- Plot ---
    _plot_cadence(all_results)

    return all_results


def _plot_cadence(results):
    plt = _import_matplotlib()
    if plt is None:
        return

    ensure_plot_dir()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#e91e63", "#2196f3", "#4caf50"]

    # Top-left: Energy curves
    ax = axes[0, 0]
    for r, c in zip(results, colors):
        epochs = list(range(1, len(r["epoch_energies"]) + 1))
        ax.plot(epochs, r["epoch_energies"], marker="o", color=c,
                label=r["name"], markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy (per sample)")
    ax.set_title("Energy Convergence by Cadence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: Accuracy curves
    ax = axes[0, 1]
    for r, c in zip(results, colors):
        epochs = list(range(1, len(r["accuracies"]) + 1))
        ax.plot(epochs, [a * 100 for a in r["accuracies"]],
                marker="o", color=c, label=r["name"], markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy by Cadence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Weight drift
    ax = axes[1, 0]
    for r, c in zip(results, colors):
        epochs = list(range(1, len(r["weight_drift"]) + 1))
        drifts = [total_drift(d) for d in r["weight_drift"]]
        ax.plot(epochs, drifts, marker="o", color=c, label=r["name"], markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||θ - θ₀||")
    ax.set_title("Weight Drift from Init")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Per-epoch weight delta
    ax = axes[1, 1]
    for r, c in zip(results, colors):
        epochs = list(range(1, len(r["weight_delta"]) + 1))
        deltas = [total_drift(d) for d in r["weight_delta"]]
        ax.plot(epochs, deltas, marker="o", color=c, label=r["name"], markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||θₙ - θₙ₋₁||")
    ax.set_title("Per-Epoch Weight Delta")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("KMNIST Anchor Cadence Comparison (Tier 1, λ=0.1)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "kmnist_cadence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


# =====================================================================
# Experiment F: iPC Steps (K) Sensitivity
# =====================================================================


def run_k_sensitivity():
    print("\n" + "=" * 70)
    print("  EXPERIMENT F: iPC Steps (K) Sensitivity (KMNIST)")
    print("=" * 70)

    NUM_EPOCHS = 10
    BATCH_SIZE = 200
    LR = 0.001
    PENALTY = 0.1
    K_VALUES = [1, 3, 5, 10, 20, 50]

    train_loader, test_loader = get_loaders(BATCH_SIZE)
    master_key = jax.random.PRNGKey(0)

    vanilla_results = []
    tier1_results = []

    for k in K_VALUES:
        print(f"\n--- K={k} ---")

        # Vanilla iPC
        structure = build_structure()
        gk, tk, master_key = jax.random.split(master_key, 3)
        params = initialize_params(structure, gk)
        optimizer = optax.adamw(LR, weight_decay=0.1)

        start = time.time()
        trained_params, collector, epoch_energies, _ = _run_ipc_with_metrics(
            params, structure, train_loader, test_loader, optimizer,
            {"num_epochs": NUM_EPOCHS, "ipc_steps": k, "carryover_index": -1},
            tk,
        )
        v_time = time.time() - start

        vanilla_results.append({
            "k": k,
            "accuracies": collector.epoch_accuracies,
            "final_acc": collector.epoch_accuracies[-1] * 100,
            "final_energy": epoch_energies[-1],
            "time": v_time,
        })

        # Tier 1
        structure = build_structure()
        gk, tk, master_key = jax.random.split(master_key, 3)
        params = initialize_params(structure, gk)
        optimizer = optax.chain(
            proximal_carryover_euclidean(PENALTY),
            optax.adamw(LR, weight_decay=0.1),
        )

        start = time.time()
        trained_params, collector, epoch_energies, _ = _run_ipc_with_metrics(
            params, structure, train_loader, test_loader, optimizer,
            {"num_epochs": NUM_EPOCHS, "ipc_steps": k,
             "carryover_index": 0, "anchor_cadence": "per_data_point"},
            tk,
        )
        t1_time = time.time() - start

        tier1_results.append({
            "k": k,
            "accuracies": collector.epoch_accuracies,
            "final_acc": collector.epoch_accuracies[-1] * 100,
            "final_energy": epoch_energies[-1],
            "time": t1_time,
        })

        print(f"  Vanilla: {vanilla_results[-1]['final_acc']:.2f}%  "
              f"Tier1: {tier1_results[-1]['final_acc']:.2f}%  "
              f"Gap: {tier1_results[-1]['final_acc'] - vanilla_results[-1]['final_acc']:+.2f}%")

    # --- Summary table ---
    print(f"\n{'K':>5} {'Vanilla Acc':>12} {'Tier1 Acc':>12} {'Gap':>8} "
          f"{'Vanilla E':>12} {'Tier1 E':>12}")
    print("-" * 65)
    for v, t in zip(vanilla_results, tier1_results):
        gap = t["final_acc"] - v["final_acc"]
        print(f"{v['k']:>5} {v['final_acc']:>11.2f}% {t['final_acc']:>11.2f}% "
              f"{gap:>+7.2f}% {v['final_energy']:>12.4f} {t['final_energy']:>12.4f}")

    # --- Plot ---
    _plot_k_sensitivity(vanilla_results, tier1_results, K_VALUES)

    return vanilla_results, tier1_results


def _plot_k_sensitivity(vanilla_results, tier1_results, k_values):
    plt = _import_matplotlib()
    if plt is None:
        return

    ensure_plot_dir()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    v_accs = [r["final_acc"] for r in vanilla_results]
    t_accs = [r["final_acc"] for r in tier1_results]
    v_energies = [r["final_energy"] for r in vanilla_results]
    t_energies = [r["final_energy"] for r in tier1_results]
    gaps = [t - v for t, v in zip(t_accs, v_accs)]

    # Left: Accuracy vs K
    ax = axes[0]
    ax.plot(k_values, v_accs, marker="o", color="#1f77b4",
            label="Vanilla iPC", linewidth=2)
    ax.plot(k_values, t_accs, marker="s", color="#2ca02c",
            label="iPC + Tier 1", linewidth=2)
    ax.set_xlabel("iPC Steps (K)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Accuracy vs K")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Center: Energy vs K
    ax = axes[1]
    ax.plot(k_values, v_energies, marker="o", color="#1f77b4",
            label="Vanilla iPC", linewidth=2)
    ax.plot(k_values, t_energies, marker="s", color="#2ca02c",
            label="iPC + Tier 1", linewidth=2)
    ax.set_xlabel("iPC Steps (K)")
    ax.set_ylabel("Final Energy")
    ax.set_title("Energy vs K")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Accuracy gap vs K
    ax = axes[2]
    bar_colors = ["#2ca02c" if g >= 0 else "#d32f2f" for g in gaps]
    ax.bar(range(len(k_values)), gaps, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel("iPC Steps (K)")
    ax.set_ylabel("Accuracy Gap (Tier 1 - Vanilla) %")
    ax.set_title("Carryover Benefit by K")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("KMNIST iPC Steps (K) Sensitivity", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "kmnist_k_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


# =====================================================================
# Round 2 Experiments — Addressing reviewer gaps
# =====================================================================
# Key changes vs Round 1:
#   - optax.adam (NO weight decay) to isolate carryover effect
#   - Multi-seed runs with error bars
#   - Aggressive grids for stability map
#   - Extended lambda range to find over-regularization boundary
# =====================================================================


# -- Optimal lambda placeholder (updated by R2-A, used by R2-B..E) --
OPTIMAL_LAMBDA = 5.0  # default; update after running lambda_sweep_v2


def _run_single_seed(seed, optimizer_fn, config, train_loader, test_loader,
                     use_standard_pc=False):
    """Run one training trial with the given seed. Returns dict of metrics."""
    key = jax.random.PRNGKey(seed)
    gk, tk = jax.random.split(key, 2)
    structure = build_structure()
    params = initialize_params(structure, gk)
    optimizer = optimizer_fn()

    start = time.time()
    if use_standard_pc:
        trained_params, iter_results, epoch_results = train_pcn(
            params=params, structure=structure, train_loader=train_loader,
            optimizer=optimizer, config=config, rng_key=tk, verbose=False,
        )
        collector_proxy_acc = evaluate_pcn(
            trained_params, structure, test_loader, config,
            jax.random.split(tk, 2)[1],
        )["accuracy"]
        energy_stats = MetricsCollector.compute_batch_energy_stats(iter_results)
        return {
            "final_acc": collector_proxy_acc,
            "final_energy": energy_stats[-1]["mean"] if energy_stats else float("nan"),
            "time": time.time() - start,
        }
    else:
        trained_params, collector, epoch_energies, energy_stats = _run_ipc_with_metrics(
            params, structure, train_loader, test_loader, optimizer, config, tk,
        )
        return {
            "final_acc": collector.epoch_accuracies[-1] if collector.epoch_accuracies else 0.0,
            "accuracies": collector.epoch_accuracies,
            "final_energy": epoch_energies[-1] if epoch_energies else float("nan"),
            "epoch_energies": epoch_energies,
            "weight_drift": collector.epoch_weight_drift,
            "time": time.time() - start,
        }


def _aggregate_seeds(seed_results):
    """Compute mean ± SE of final accuracy across seeds."""
    accs = np.array([r["final_acc"] * 100 for r in seed_results])
    return {
        "mean_acc": float(np.mean(accs)),
        "se_acc": float(np.std(accs, ddof=1) / np.sqrt(len(accs))) if len(accs) > 1 else 0.0,
        "all_accs": accs,
        "mean_energy": float(np.mean([r["final_energy"] for r in seed_results])),
        "mean_drift": float(np.mean([
            total_drift(r["weight_drift"][-1]) for r in seed_results
            if "weight_drift" in r and r["weight_drift"]
        ])) if any("weight_drift" in r and r["weight_drift"] for r in seed_results) else 0.0,
    }


# =====================================================================
# R2-A: Extended Lambda Sweep (find over-regularization boundary)
# =====================================================================


def run_lambda_sweep_v2():
    print("\n" + "=" * 70)
    print("  R2-A: Extended Lambda Sweep (adam, multi-seed)")
    print("=" * 70)

    NUM_EPOCHS = 10
    BATCH_SIZE = 200
    LR = 0.001
    IPC_STEPS = 20
    LAMBDAS = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    SEEDS = [0, 1000, 2000]

    train_loader, test_loader = get_loaders(BATCH_SIZE)

    all_results = []

    for lam in LAMBDAS:
        label = "Vanilla (λ=0)" if lam == 0.0 else f"λ={lam}"
        print(f"\n--- {label} ---")

        seed_results = []
        for seed in SEEDS:
            if lam == 0.0:
                opt_fn = lambda: optax.adam(LR)
                config = {"num_epochs": NUM_EPOCHS, "ipc_steps": IPC_STEPS,
                          "carryover_index": -1}
            else:
                _lam = lam  # capture
                opt_fn = lambda _l=_lam: optax.chain(
                    proximal_carryover_euclidean(_l), optax.adam(LR))
                config = {"num_epochs": NUM_EPOCHS, "ipc_steps": IPC_STEPS,
                          "carryover_index": 0, "anchor_cadence": "per_data_point"}

            r = _run_single_seed(seed, opt_fn, config, train_loader, test_loader)
            seed_results.append(r)
            print(f"    seed={seed}: acc={r['final_acc']*100:.2f}%")

        agg = _aggregate_seeds(seed_results)
        all_results.append({
            "lambda": lam, "label": label,
            "mean_acc": agg["mean_acc"], "se_acc": agg["se_acc"],
            "mean_energy": agg["mean_energy"], "mean_drift": agg["mean_drift"],
            "all_accs": agg["all_accs"],
        })
        print(f"    => {agg['mean_acc']:.2f}% ± {agg['se_acc']:.2f}%")

    # --- Summary table ---
    print(f"\n{'Lambda':>10} {'Accuracy':>16} {'Energy':>10} {'Drift':>10}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['label']:>10} {r['mean_acc']:>8.2f} ± {r['se_acc']:.2f}% "
              f"{r['mean_energy']:>10.4f} {r['mean_drift']:>10.2f}")

    # Find optimal λ
    best = max(all_results, key=lambda r: r["mean_acc"])
    global OPTIMAL_LAMBDA
    if best["lambda"] > 0:
        OPTIMAL_LAMBDA = best["lambda"]
    print(f"\n  Optimal λ = {OPTIMAL_LAMBDA} (accuracy = {best['mean_acc']:.2f}%)")

    _plot_lambda_sweep_v2(all_results)
    return all_results


def _plot_lambda_sweep_v2(results):
    plt = _import_matplotlib()
    if plt is None:
        return

    ensure_plot_dir()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    baseline = [r for r in results if r["lambda"] == 0.0][0]
    sweep = [r for r in results if r["lambda"] > 0]
    lambdas = [r["lambda"] for r in sweep]
    accs = [r["mean_acc"] for r in sweep]
    acc_errs = [r["se_acc"] for r in sweep]
    energies = [r["mean_energy"] for r in sweep]
    drifts = [r["mean_drift"] for r in sweep]

    # Top-left: Accuracy vs λ with error bars
    ax = axes[0, 0]
    ax.errorbar(lambdas, accs, yerr=acc_errs, marker="o", color="#2ca02c",
                linewidth=2, capsize=4, capthick=1.5)
    ax.axhline(baseline["mean_acc"], color="#1f77b4", linestyle="--",
               label=f"Vanilla iPC ({baseline['mean_acc']:.1f}%)")
    ax.axhspan(baseline["mean_acc"] - baseline["se_acc"],
               baseline["mean_acc"] + baseline["se_acc"],
               color="#1f77b4", alpha=0.15)
    ax.set_xlabel("Penalty Strength (λ)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Accuracy vs λ (mean ± SE)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: Energy vs λ
    ax = axes[0, 1]
    ax.semilogx(lambdas, energies, marker="o", color="#ff7f0e", linewidth=2)
    ax.axhline(baseline["mean_energy"], color="#1f77b4", linestyle="--",
               label=f"Vanilla iPC ({baseline['mean_energy']:.4f})")
    ax.set_xlabel("Penalty Strength (λ)")
    ax.set_ylabel("Final Energy")
    ax.set_title("Energy vs λ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Weight drift vs λ
    ax = axes[1, 0]
    ax.semilogx(lambdas, drifts, marker="o", color="#d62728", linewidth=2)
    ax.axhline(baseline["mean_drift"], color="#1f77b4", linestyle="--",
               label=f"Vanilla iPC ({baseline['mean_drift']:.1f})")
    ax.set_xlabel("Penalty Strength (λ)")
    ax.set_ylabel("Total Weight Drift ||θ - θ₀||")
    ax.set_title("Weight Drift vs λ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: per-seed scatter + mean
    ax = axes[1, 1]
    for r in results:
        lam_pos = r["lambda"] if r["lambda"] > 0 else 0.05  # offset vanilla for log scale
        ax.scatter([lam_pos] * len(r["all_accs"]), r["all_accs"],
                   alpha=0.5, s=30, color="#2ca02c" if r["lambda"] > 0 else "#1f77b4")
    ax.plot([r["lambda"] if r["lambda"] > 0 else 0.05 for r in results],
            [r["mean_acc"] for r in results], marker="D", color="black",
            linewidth=1.5, markersize=5, label="Mean")
    ax.set_xlabel("Penalty Strength (λ)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Per-Seed Results")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("R2-A: Extended Lambda Sweep (adam, no weight decay)", fontsize=14, y=1.02)
    fig.tight_layout()
    
    path = os.path.join(PLOT_DIR, "kmnist_lambda_sweep_v2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


# =====================================================================
# R2-B: Stress-Test Stability Map (K=1, adam, aggressive grid)
# =====================================================================


def run_stability_map_v2():
    print("\n" + "=" * 70)
    print("  R2-B: Stress-Test Stability Map (K=1, adam, no weight decay)")
    print("=" * 70)

    ETA_INFERS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    LRS = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    PENALTY = OPTIMAL_LAMBDA
    IPC_STEPS = 1  # maximally aggressive
    NUM_EPOCHS = 10
    BATCH_SIZE = 200
    ENERGY_DIVERGE_THRESHOLD = 1e6

    print(f"  Using λ={PENALTY} (from lambda sweep)")
    print(f"  K={IPC_STEPS}, epochs={NUM_EPOCHS}, no weight decay")

    train_loader, test_loader = get_loaders(BATCH_SIZE)

    def make_optimizer(cond_name, lr):
        base = optax.adam(lr)
        if cond_name == "Vanilla iPC":
            return base, -1
        elif cond_name == "Tier 1":
            return optax.chain(proximal_carryover_euclidean(PENALTY), base), 0
        elif cond_name == "Tier 2":
            return optax.chain(proximal_carryover_fisher(PENALTY), base), 0

    cond_names = ["Vanilla iPC", "Tier 1", "Tier 2"]
    energy_grids = {}
    acc_grids = {}
    oscillation_grids = {}

    total_runs = len(cond_names) * len(ETA_INFERS) * len(LRS)
    run_idx = 0

    for cond_name in cond_names:
        e_grid = np.full((len(ETA_INFERS), len(LRS)), np.nan)
        a_grid = np.full((len(ETA_INFERS), len(LRS)), np.nan)
        o_grid = np.full((len(ETA_INFERS), len(LRS)), np.nan)

        for i, eta in enumerate(ETA_INFERS):
            for j, lr in enumerate(LRS):
                run_idx += 1
                key = jax.random.PRNGKey(42)
                gk, tk, ek = jax.random.split(key, 3)

                structure = build_structure(eta_infer=eta, infer_steps=IPC_STEPS)
                optimizer, co_idx = make_optimizer(cond_name, lr)
                config = {
                    "num_epochs": NUM_EPOCHS,
                    "ipc_steps": IPC_STEPS,
                    "carryover_index": co_idx,
                    "anchor_cadence": "per_data_point",
                }

                try:
                    params = initialize_params(structure, gk)
                    trained_params, energy_history, _ = train_pcn_ipc(
                        params=params, structure=structure,
                        train_loader=train_loader, optimizer=optimizer,
                        config=config, rng_key=tk, verbose=False,
                    )

                    # Check for divergence in energy history
                    diverged = False
                    for epoch_energies_list in energy_history:
                        if any(not np.isfinite(e) or e > ENERGY_DIVERGE_THRESHOLD
                               for e in epoch_energies_list):
                            diverged = True
                            break

                    if not diverged and energy_history:
                        last_epoch = energy_history[-1]
                        if last_epoch and all(np.isfinite(e) for e in last_epoch):
                            e_grid[i, j] = np.mean(last_epoch)

                            # Oscillation: energy variance over last 2 epochs
                            tail_epochs = energy_history[-2:]
                            all_tail = []
                            for ep in tail_epochs:
                                all_tail.extend(ep)
                            if all_tail:
                                o_grid[i, j] = float(np.var(all_tail))

                            metrics = evaluate_pcn(
                                trained_params, structure, test_loader, config, ek
                            )
                            a_grid[i, j] = metrics["accuracy"] * 100
                except Exception:
                    pass  # stays NaN

                status = f"E={e_grid[i,j]:.4f}" if np.isfinite(e_grid[i,j]) else "DIVERGED"
                print(f"  [{run_idx}/{total_runs}] {cond_name} "
                      f"eta={eta} lr={lr} -> {status}")

        energy_grids[cond_name] = e_grid
        acc_grids[cond_name] = a_grid
        oscillation_grids[cond_name] = o_grid

    # --- Summary ---
    for cond_name in cond_names:
        eg = energy_grids[cond_name]
        ag = acc_grids[cond_name]
        stable = int(np.sum(np.isfinite(eg)))
        print(f"\n  {cond_name}: {stable}/{eg.size} stable, "
              f"avg energy={np.nanmean(eg):.4f}, avg acc={np.nanmean(ag):.1f}%")

    _plot_stability_map_v2(energy_grids, acc_grids, oscillation_grids,
                           ETA_INFERS, LRS, PENALTY)
    return energy_grids, acc_grids, oscillation_grids


def _plot_stability_map_v2(energy_grids, acc_grids, osc_grids,
                           eta_infers, lrs, penalty):
    plt = _import_matplotlib()
    if plt is None:
        return

    ensure_plot_dir()
    cond_names = list(energy_grids.keys())
    fig, axes = plt.subplots(3, len(cond_names),
                             figsize=(5.5 * len(cond_names), 14))

    eta_labels = [str(e) for e in eta_infers]
    lr_labels = [str(l) for l in lrs]

    def _draw_heatmap(ax, grid, cmap, title, fmt=".3f", vmin=None, vmax=None):
        masked = np.ma.masked_invalid(grid)
        if vmin is None:
            vmin = np.nanmin(grid) if np.any(np.isfinite(grid)) else 0
        if vmax is None:
            vmax = np.nanmax(grid) if np.any(np.isfinite(grid)) else 1
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto", origin="lower")
        nan_mask = np.isnan(grid)
        if nan_mask.any():
            ax.imshow(np.where(nan_mask, 1.0, np.nan), cmap="Reds",
                      vmin=0, vmax=1, aspect="auto", origin="lower", alpha=0.7)
        ax.set_xticks(range(len(lr_labels)))
        ax.set_xticklabels(lr_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(eta_labels)))
        ax.set_yticklabels(eta_labels, fontsize=7)
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Inference rate (eta)")
        stable = int(np.sum(np.isfinite(grid)))
        ax.set_title(f"{title}\n({stable}/{grid.size} stable)", fontsize=10)
        for ii in range(grid.shape[0]):
            for jj in range(grid.shape[1]):
                val = grid[ii, jj]
                if np.isfinite(val):
                    txt = f"{val:{fmt}}" if fmt != ".0f" else f"{val:.0f}"
                    ax.text(jj, ii, txt, ha="center", va="center",
                            fontsize=6, color="white" if val > (vmin + vmax) / 2 else "black")
                else:
                    ax.text(jj, ii, "X", ha="center", va="center",
                            fontsize=8, fontweight="bold", color="white")
        return im

    # Shared color scales
    all_e = np.concatenate([energy_grids[c].flatten() for c in cond_names])
    e_vmin = np.nanmin(all_e) if np.any(np.isfinite(all_e)) else 0
    e_vmax = np.nanmax(all_e) if np.any(np.isfinite(all_e)) else 1
    all_a = np.concatenate([acc_grids[c].flatten() for c in cond_names])
    a_vmin = np.nanmin(all_a) if np.any(np.isfinite(all_a)) else 0
    a_vmax = np.nanmax(all_a) if np.any(np.isfinite(all_a)) else 100
    all_o = np.concatenate([osc_grids[c].flatten() for c in cond_names])
    o_vmin = 0
    o_vmax = np.nanmax(all_o) if np.any(np.isfinite(all_o)) else 1

    for col, cn in enumerate(cond_names):
        _draw_heatmap(axes[0, col], energy_grids[cn], "viridis_r",
                      f"{cn} — Energy", ".3f", e_vmin, e_vmax)
        _draw_heatmap(axes[1, col], acc_grids[cn], "RdYlGn",
                      f"{cn} — Accuracy (%)", ".0f", a_vmin, a_vmax)
        _draw_heatmap(axes[2, col], osc_grids[cn], "YlOrRd",
                      f"{cn} — Oscillation", ".2e", o_vmin, o_vmax)

    fig.suptitle(f"R2-B: Stress-Test Stability Map (K=1, adam, λ={penalty})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "kmnist_stability_map_v2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


# =====================================================================
# R2-C: Corrected Ablation (two lr values, optimal λ, multi-seed)
# =====================================================================


def run_ablation_v2():
    print("\n" + "=" * 70)
    print("  R2-C: Corrected Ablation (adamw, two lr values, multi-seed)")
    print("=" * 70)

    NUM_EPOCHS = 10
    BATCH_SIZE = 200
    IPC_STEPS = 20
    PENALTY = OPTIMAL_LAMBDA
    LR_VALUES = [0.001, 0.005]
    SEEDS = [0, 1000, 2000]

    print(f"  Using λ={PENALTY}, cadence=per_data_point, K={IPC_STEPS}")

    train_loader, test_loader = get_loaders(BATCH_SIZE)

    all_results = {}  # {lr: {condition_name: aggregated_results}}

    for lr in LR_VALUES:
        print(f"\n{'='*40}")
        print(f"  lr = {lr}")
        print(f"{'='*40}")
        all_results[lr] = {}

        conditions = [
            ("Standard PC", True, lambda _lr=lr: optax.adamw(_lr, weight_decay=0.01),
             {"num_epochs": NUM_EPOCHS}),
            ("Vanilla iPC", False, lambda _lr=lr: optax.adamw(_lr, weight_decay=0.01),
             {"num_epochs": NUM_EPOCHS, "ipc_steps": IPC_STEPS, "carryover_index": -1}),
            ("iPC + Tether", False,
             lambda _lr=lr, _p=PENALTY: optax.chain(
                 proximal_carryover_euclidean(_p), optax.adamw(_lr, weight_decay=0.01)),
             {"num_epochs": NUM_EPOCHS, "ipc_steps": IPC_STEPS,
              "carryover_index": 0, "anchor_cadence": "per_data_point"}),
        ]

        for cond_name, use_std_pc, opt_fn, config in conditions:
            print(f"\n  --- {cond_name} ---")
            seed_results = []
            for seed in SEEDS:
                r = _run_single_seed(seed, opt_fn, config, train_loader, test_loader,
                                     use_standard_pc=use_std_pc)
                seed_results.append(r)
                print(f"    seed={seed}: acc={r['final_acc']*100:.2f}%")

            agg = _aggregate_seeds(seed_results)
            all_results[lr][cond_name] = agg
            print(f"    => {agg['mean_acc']:.2f}% ± {agg['se_acc']:.2f}%")

    # --- Summary ---
    for lr in LR_VALUES:
        print(f"\n  lr={lr}:")
        for cond, agg in all_results[lr].items():
            print(f"    {cond:>15}: {agg['mean_acc']:.2f} ± {agg['se_acc']:.2f}%")

    _plot_ablation_v2(all_results, LR_VALUES, PENALTY)
    return all_results


def _plot_ablation_v2(all_results, lr_values, penalty):
    plt = _import_matplotlib()
    if plt is None:
        return

    ensure_plot_dir()
    fig, axes = plt.subplots(1, len(lr_values), figsize=(7 * len(lr_values), 6))
    if len(lr_values) == 1:
        axes = [axes]

    colors = {"Standard PC": "#1f77b4", "Vanilla iPC": "#ff7f0e",
              "iPC + Tier 1": "#2ca02c", "iPC + Tier 2": "#d62728"}

    for idx, lr in enumerate(lr_values):
        ax = axes[idx]
        cond_names = list(all_results[lr].keys())
        means = [all_results[lr][c]["mean_acc"] for c in cond_names]
        errs = [all_results[lr][c]["se_acc"] for c in cond_names]
        bar_colors = [colors.get(c, "gray") for c in cond_names]

        bars = ax.bar(cond_names, means, yerr=errs, color=bar_colors,
                      edgecolor="black", linewidth=0.5, capsize=5)
        for bar, m, e in zip(bars, means, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e + 0.3,
                    f"{m:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(f"lr = {lr}")
        ax.grid(True, alpha=0.3, axis="y")
        # Set y axis to start near the minimum
        y_min = max(0, min(means) - max(errs) - 5)
        ax.set_ylim(y_min, max(means) + max(errs) + 3)

    fig.suptitle(f"R2-C: Corrected Ablation (adamw, λ={penalty}, mean ± SE over 3 seeds)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "kmnist_ablation_v2_adamw.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


# =====================================================================
# R2-D: Multi-Seed Statistical Comparison (ABExperiment)
# =====================================================================


def run_multi_seed_comparison():
    print("\n" + "=" * 70)
    print("  R2-D: Multi-Seed Statistical Comparison (paired t-test)")
    print("=" * 70)

    from fabricpc.experiments import ExperimentArm, ABExperiment

    NUM_EPOCHS = 10
    IPC_STEPS = 20
    LR = 0.001
    PENALTY = OPTIMAL_LAMBDA
    N_TRIALS = 10
    BATCH_SIZE = 200

    print(f"  λ={PENALTY}, lr={LR}, K={IPC_STEPS}, {N_TRIALS} paired trials")

    def model_factory(rng_key):
        structure = build_structure()
        params = initialize_params(structure, rng_key)
        return params, structure

    def data_loader_factory(seed):
        return get_loaders(BATCH_SIZE)

    arm_a = ExperimentArm(
        name="iPC + Tier 1",
        model_factory=model_factory,
        train_fn=train_pcn_ipc,
        eval_fn=evaluate_pcn,
        optimizer=optax.chain(
            proximal_carryover_euclidean(PENALTY), optax.adam(LR)
        ),
        train_config={
            "num_epochs": NUM_EPOCHS,
            "ipc_steps": IPC_STEPS,
            "carryover_index": 0,
            "anchor_cadence": "per_data_point",
        },
    )

    arm_b = ExperimentArm(
        name="Vanilla iPC",
        model_factory=model_factory,
        train_fn=train_pcn_ipc,
        eval_fn=evaluate_pcn,
        optimizer=optax.adam(LR),
        train_config={
            "num_epochs": NUM_EPOCHS,
            "ipc_steps": IPC_STEPS,
            "carryover_index": -1,
        },
    )

    experiment = ABExperiment(
        arm_a=arm_a,
        arm_b=arm_b,
        metric="accuracy",
        data_loader_factory=data_loader_factory,
        n_trials=N_TRIALS,
        verbose=False,
    )

    results = experiment.run()
    results.print_summary()

    # Save summary to file
    ensure_plot_dir()
    summary_path = os.path.join(PLOT_DIR, "kmnist_ab_comparison.txt")
    import io
    import sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    results.print_summary()
    sys.stdout = old_stdout
    with open(summary_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\n  Summary saved: {summary_path}")

    return results


# =====================================================================
# R2-E: K Sensitivity v2 (optimal λ, multi-seed)
# =====================================================================


def run_k_sensitivity_v2():
    print("\n" + "=" * 70)
    print("  R2-E: K Sensitivity v2 (adam, optimal λ, multi-seed)")
    print("=" * 70)

    NUM_EPOCHS = 8
    BATCH_SIZE = 200
    LR = 0.001
    PENALTY = OPTIMAL_LAMBDA
    K_VALUES = [1, 3, 5, 10, 20, 50]
    SEEDS = [0, 1000, 2000]

    print(f"  λ={PENALTY}, lr={LR}, {len(SEEDS)} seeds per condition")

    train_loader, test_loader = get_loaders(BATCH_SIZE)

    vanilla_results = []
    tier1_results = []

    for k in K_VALUES:
        print(f"\n--- K={k} ---")

        # Vanilla iPC
        v_seeds = []
        for seed in SEEDS:
            opt_fn = lambda: optax.adam(LR)
            config = {"num_epochs": NUM_EPOCHS, "ipc_steps": k, "carryover_index": -1}
            r = _run_single_seed(seed, opt_fn, config, train_loader, test_loader)
            v_seeds.append(r)
        v_agg = _aggregate_seeds(v_seeds)
        vanilla_results.append({"k": k, **v_agg})

        # Tier 1
        t_seeds = []
        for seed in SEEDS:
            _k = k
            opt_fn = lambda _kk=_k: optax.chain(
                proximal_carryover_euclidean(PENALTY), optax.adam(LR))
            config = {"num_epochs": NUM_EPOCHS, "ipc_steps": _k,
                      "carryover_index": 0, "anchor_cadence": "per_data_point"}
            r = _run_single_seed(seed, opt_fn, config, train_loader, test_loader)
            t_seeds.append(r)
        t_agg = _aggregate_seeds(t_seeds)
        tier1_results.append({"k": k, **t_agg})

        gap = t_agg["mean_acc"] - v_agg["mean_acc"]
        print(f"  Vanilla: {v_agg['mean_acc']:.2f}±{v_agg['se_acc']:.2f}%  "
              f"Tier1: {t_agg['mean_acc']:.2f}±{t_agg['se_acc']:.2f}%  "
              f"Gap: {gap:+.2f}%")

    # --- Summary table ---
    print(f"\n{'K':>5} {'Vanilla':>18} {'Tier1':>18} {'Gap':>8}")
    print("-" * 55)
    for v, t in zip(vanilla_results, tier1_results):
        gap = t["mean_acc"] - v["mean_acc"]
        print(f"{v['k']:>5} {v['mean_acc']:>8.2f}±{v['se_acc']:.2f}% "
              f"{t['mean_acc']:>8.2f}±{t['se_acc']:.2f}% {gap:>+7.2f}%")

    _plot_k_sensitivity_v2(vanilla_results, tier1_results, K_VALUES, PENALTY)
    return vanilla_results, tier1_results


def _plot_k_sensitivity_v2(vanilla_results, tier1_results, k_values, penalty):
    plt = _import_matplotlib()
    if plt is None:
        return

    ensure_plot_dir()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    v_accs = [r["mean_acc"] for r in vanilla_results]
    t_accs = [r["mean_acc"] for r in tier1_results]
    v_errs = [r["se_acc"] for r in vanilla_results]
    t_errs = [r["se_acc"] for r in tier1_results]
    gaps = [t - v for t, v in zip(t_accs, v_accs)]
    # Propagate error for gap
    gap_errs = [np.sqrt(ve**2 + te**2) for ve, te in zip(v_errs, t_errs)]

    # Left: Accuracy vs K with error bars
    ax = axes[0]
    ax.errorbar(k_values, v_accs, yerr=v_errs, marker="o", color="#1f77b4",
                label="Vanilla iPC", linewidth=2, capsize=4)
    ax.errorbar(k_values, t_accs, yerr=t_errs, marker="s", color="#2ca02c",
                label="iPC + Tier 1", linewidth=2, capsize=4)
    ax.set_xlabel("iPC Steps (K)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Accuracy vs K (mean ± SE)")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Center: Energy vs K
    ax = axes[1]
    v_energies = [r["mean_energy"] for r in vanilla_results]
    t_energies = [r["mean_energy"] for r in tier1_results]
    ax.plot(k_values, v_energies, marker="o", color="#1f77b4",
            label="Vanilla iPC", linewidth=2)
    ax.plot(k_values, t_energies, marker="s", color="#2ca02c",
            label="iPC + Tier 1", linewidth=2)
    ax.set_xlabel("iPC Steps (K)")
    ax.set_ylabel("Final Energy")
    ax.set_title("Energy vs K")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Gap with error bars
    ax = axes[2]
    bar_colors = ["#2ca02c" if g >= 0 else "#d32f2f" for g in gaps]
    bars = ax.bar(range(len(k_values)), gaps, yerr=gap_errs,
                  color=bar_colors, edgecolor="black", linewidth=0.5, capsize=4)
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel("iPC Steps (K)")
    ax.set_ylabel("Accuracy Gap (Tier 1 - Vanilla) %")
    ax.set_title("Carryover Benefit by K (± SE)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"R2-E: K Sensitivity (adam, λ={penalty}, mean ± SE)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "kmnist_k_sensitivity_v2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


# =====================================================================
# R2-F: Adam vs AdamW Ablation (fixed λ and lr, side-by-side comparison)
# =====================================================================


def _run_adam_vs_adamw_for_cadence(cadence, train_loader, test_loader,
                                   num_epochs, batch_size, ipc_steps,
                                   penalty, lr, seeds):
    """Run the Adam vs AdamW ablation for a single anchor cadence."""
    print(f"\n{'#'*70}")
    print(f"  Cadence: {cadence}")
    print(f"{'#'*70}")

    optimizer_families = {
        "Adam": lambda: optax.adam(lr),
        "AdamW": lambda: optax.adamw(lr, weight_decay=0.01),
    }

    all_results = {}

    for opt_name, base_opt_fn in optimizer_families.items():
        print(f"\n{'='*40}")
        print(f"  Optimizer: {opt_name}")
        print(f"{'='*40}")
        all_results[opt_name] = {}

        conditions = [
            ("Standard PC", True,
             base_opt_fn,
             {"num_epochs": num_epochs}),
            ("Vanilla iPC", False,
             base_opt_fn,
             {"num_epochs": num_epochs, "ipc_steps": ipc_steps,
              "carryover_index": -1}),
            ("iPC + Tier 1", False,
             lambda _bof=base_opt_fn, _p=penalty: optax.chain(
                 proximal_carryover_euclidean(_p), _bof()),
             {"num_epochs": num_epochs, "ipc_steps": ipc_steps,
              "carryover_index": 0, "anchor_cadence": cadence}),
            ("iPC + Tier 2", False,
             lambda _bof=base_opt_fn, _p=penalty: optax.chain(
                 proximal_carryover_fisher(_p), _bof()),
             {"num_epochs": num_epochs, "ipc_steps": ipc_steps,
              "carryover_index": 0, "anchor_cadence": cadence}),
        ]

        for cond_name, use_std_pc, opt_fn, config in conditions:
            print(f"\n  --- {cond_name} ---")
            seed_results = []
            for seed in seeds:
                r = _run_single_seed(seed, opt_fn, config, train_loader,
                                     test_loader, use_standard_pc=use_std_pc)
                seed_results.append(r)
                print(f"    seed={seed}: acc={r['final_acc']*100:.2f}%")

            agg = _aggregate_seeds(seed_results)
            all_results[opt_name][cond_name] = agg
            print(f"    => {agg['mean_acc']:.2f}% ± {agg['se_acc']:.2f}%")

    return all_results


def run_adam_vs_adamw_ablation():
    print("\n" + "=" * 70)
    print("  R2-F: Adam vs AdamW Ablation (λ=5.0, lr=0.001, 3 seeds)")
    print("=" * 70)

    NUM_EPOCHS = 10
    BATCH_SIZE = 200
    IPC_STEPS = 20
    PENALTY = 5.0
    LR = 0.001
    SEEDS = [0, 1000, 2000]
    CADENCES = ["per_data_point", "per_epoch"]

    print(f"  λ={PENALTY}, lr={LR}, K={IPC_STEPS}")
    print(f"  Cadences: {CADENCES}")

    train_loader, test_loader = get_loaders(BATCH_SIZE)

    for cadence in CADENCES:
        results = _run_adam_vs_adamw_for_cadence(
            cadence, train_loader, test_loader,
            NUM_EPOCHS, BATCH_SIZE, IPC_STEPS, PENALTY, LR, SEEDS)

        # Summary
        print(f"\n{'='*60}")
        print(f"  Summary ({cadence})")
        print(f"{'='*60}")
        for opt_name in results:
            print(f"\n  {opt_name}:")
            for cond, agg in results[opt_name].items():
                print(f"    {cond:>15}: {agg['mean_acc']:.2f} ± {agg['se_acc']:.2f}%")

        _plot_adam_vs_adamw(results, PENALTY, LR, cadence)


def _plot_adam_vs_adamw(all_results, penalty, lr, cadence):
    plt = _import_matplotlib()
    if plt is None:
        return

    ensure_plot_dir()
    opt_names = list(all_results.keys())
    fig, axes = plt.subplots(1, len(opt_names), figsize=(7 * len(opt_names), 6))
    if len(opt_names) == 1:
        axes = [axes]

    colors = {"Standard PC": "#1f77b4", "Vanilla iPC": "#ff7f0e",
              "iPC + Tier 1": "#2ca02c", "iPC + Tier 2": "#d62728"}

    for idx, opt_name in enumerate(opt_names):
        ax = axes[idx]
        cond_names = list(all_results[opt_name].keys())
        means = [all_results[opt_name][c]["mean_acc"] for c in cond_names]
        errs = [all_results[opt_name][c]["se_acc"] for c in cond_names]
        bar_colors = [colors.get(c, "gray") for c in cond_names]

        bars = ax.bar(cond_names, means, yerr=errs, color=bar_colors,
                      edgecolor="black", linewidth=0.5, capsize=5)
        for bar, m, e in zip(bars, means, errs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + e + 0.3,
                    f"{m:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(opt_name)
        ax.grid(True, alpha=0.3, axis="y")
        y_min = max(0, min(means) - max(errs) - 5)
        ax.set_ylim(y_min, max(means) + max(errs) + 3)

    fig.suptitle(
        f"R2-F: Adam vs AdamW Ablation (λ={penalty}, lr={lr}, "
        f"{cadence}, mean ± SE over 3 seeds)",
        fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f"kmnist_adam_vs_adamw_{cadence}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


# =====================================================================
# Main
# =====================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TPC experiments on KMNIST with incremental predictive coding"
    )
    parser.add_argument(
        "--experiment",
        choices=["all", "phase1", "phase2",
                 "comparison", "stability", "ablation",
                 "lambda_sweep", "cadence", "k_sensitivity",
                 "round2", "lambda_sweep_v2", "stability_v2",
                 "ablation_v2", "multi_seed", "k_sensitivity_v2",
                 "adam_vs_adamw"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    args = parser.parse_args()
    exp = args.experiment

    # Phase 1: quick experiments
    if exp in ("all", "phase1", "comparison"):
        run_comparison()

    if exp in ("all", "phase1", "lambda_sweep"):
        run_lambda_sweep()

    if exp in ("all", "phase1", "cadence"):
        run_cadence()

    if exp in ("all", "phase1", "k_sensitivity"):
        run_k_sensitivity()

    # Phase 2: expensive experiments
    if exp in ("all", "phase2", "stability"):
        run_stability_map()

    if exp in ("all", "phase2", "ablation"):
        run_ablation()

    # Round 2: addressing reviewer gaps
    if exp in ("round2", "lambda_sweep_v2"):
        run_lambda_sweep_v2()

    if exp in ("round2", "stability_v2"):
        run_stability_map_v2()

    if exp in ("round2", "ablation_v2"):
        run_ablation_v2()

    if exp in ("round2", "multi_seed"):
        run_multi_seed_comparison()

    if exp in ("round2", "k_sensitivity_v2"):
        run_k_sensitivity_v2()

    if exp == "adam_vs_adamw":
        run_adam_vs_adamw_ablation()

    print("\nDone.")
