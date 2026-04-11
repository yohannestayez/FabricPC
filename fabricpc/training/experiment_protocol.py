"""Utilities for experiment logging, aggregation, and stability checks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any, Iterable, Sequence

import numpy as np


def flatten_iter_results(iter_results: Sequence[Sequence[float]]) -> list[float]:
    """Flatten nested epoch -> batch energy history into one list."""
    return [float(value) for epoch in iter_results for value in epoch]


def compute_epoch_energy_stats(iter_results: Sequence[Sequence[float]]) -> list[dict[str, float]]:
    """Compute descriptive statistics for each epoch's batch energies."""
    stats: list[dict[str, float]] = []
    for epoch_energies in iter_results:
        arr = np.asarray(epoch_energies, dtype=np.float64)
        if arr.size == 0:
            stats.append({
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "cv": float("nan"),
            })
            continue
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        stats.append({
            "mean": mean,
            "std": std,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "cv": float(std / mean) if mean > 0 else 0.0,
        })
    return stats


def assess_stability(
    iter_results: Sequence[Sequence[float]],
    *,
    energy_cap: float,
    oscillation_window: int,
    patience: int,
    min_improvement: float = 1e-4,
    energy_threshold: float | None = None,
) -> dict[str, Any]:
    """Classify a run as stable or unstable from its batch-energy history."""
    flat = flatten_iter_results(iter_results)
    epoch_means = [
        float(np.mean(np.asarray(epoch, dtype=np.float64)))
        for epoch in iter_results
        if len(epoch) > 0
    ]
    assessment = {
        "is_finite": True,
        "is_stable": True,
        "failure_reason": None,
        "max_energy": float("nan"),
        "final_energy": float("nan"),
        "stable_epochs": 0,
        "oscillation_std": float("nan"),
        "oscillation_amplitude": float("nan"),
        "time_to_threshold": None,
        "plateau_triggered": False,
    }
    if not flat:
        assessment.update({
            "is_finite": False,
            "is_stable": False,
            "failure_reason": "no_history",
            "stable_epochs": 0,
        })
        return assessment

    flat_arr = np.asarray(flat, dtype=np.float64)
    finite_mask = np.isfinite(flat_arr)
    assessment["is_finite"] = bool(np.all(finite_mask))
    if not assessment["is_finite"]:
        first_bad = int(np.argmax(~finite_mask))
        stable_batches = first_bad
        assessment.update({
            "is_stable": False,
            "failure_reason": "non_finite",
            "stable_epochs": int(sum(1 for epoch in iter_results if len(epoch) > 0 and np.all(np.isfinite(epoch)))),
            "max_energy": float(np.nanmax(flat_arr)),
            "final_energy": float(flat_arr[first_bad - 1]) if stable_batches > 0 else float("nan"),
        })
        return assessment

    assessment["max_energy"] = float(np.max(flat_arr))
    assessment["final_energy"] = float(flat_arr[-1])
    assessment["stable_epochs"] = int(sum(1 for epoch in iter_results if len(epoch) > 0 and np.all(np.isfinite(epoch))))

    window = min(len(flat), max(1, oscillation_window))
    tail = flat_arr[-window:]
    assessment["oscillation_std"] = float(np.std(tail))
    assessment["oscillation_amplitude"] = float(np.max(tail) - np.min(tail))

    if energy_threshold is not None:
        hits = np.flatnonzero(flat_arr <= energy_threshold)
        if hits.size:
            assessment["time_to_threshold"] = int(hits[0] + 1)

    if assessment["max_energy"] > energy_cap:
        assessment.update({
            "is_stable": False,
            "failure_reason": "energy_cap",
        })
        return assessment

    if patience > 0 and epoch_means:
        best = float("inf")
        stagnant = 0
        for mean in epoch_means:
            if best - mean > min_improvement:
                best = mean
                stagnant = 0
            else:
                stagnant += 1
            if stagnant >= patience:
                assessment.update({
                    "is_stable": False,
                    "failure_reason": "no_progress",
                    "plateau_triggered": True,
                })
                return assessment

    return assessment


def aggregate_runs(
    runs: Sequence[dict[str, Any]],
    *,
    series_keys: Iterable[str],
    scalar_keys: Iterable[str],
) -> dict[str, Any]:
    """Aggregate matched-seed runs into mean/std summaries."""
    if not runs:
        raise ValueError("aggregate_runs requires at least one run")

    result: dict[str, Any] = {
        "num_runs": len(runs),
        "series": {},
        "summary": {},
    }
    for key in series_keys:
        values = [run[key] for run in runs if run.get(key) is not None]
        if not values:
            continue
        min_len = min(len(value) for value in values)
        arr = np.asarray([np.asarray(value[:min_len], dtype=np.float64) for value in values])
        result["series"][key] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
        }
    for key in scalar_keys:
        values = [float(run[key]) for run in runs if run.get(key) is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        result["summary"][key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return result


def aggregate_failure_reasons(runs: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count failure reasons across a set of runs."""
    counts = Counter((run.get("failure_reason") or "stable") for run in runs)
    return dict(sorted(counts.items()))


def to_serializable(value: Any) -> Any:
    """Convert numpy and JAX-like objects into JSON-friendly values."""
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            return value.tolist()
        except TypeError:
            pass
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact with deterministic formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_serializable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON artifact previously saved by :func:`save_json`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
