"""
Training utilities for JAX predictive coding networks.
"""

from fabricpc_jax.training.train_loop import train_step, train_pcn, evaluate_pcn
from fabricpc_jax.training.optimizers import create_optimizer
from fabricpc_jax.training.multi_gpu import (
    train_pcn_multi_gpu,
    evaluate_pcn_multi_gpu,
    replicate_params,
    shard_batch,
)

__all__ = [
    "train_step",
    "train_pcn",
    "evaluate_pcn",
    "create_optimizer",
    "train_pcn_multi_gpu",
    "evaluate_pcn_multi_gpu",
    "replicate_params",
    "shard_batch",
]
