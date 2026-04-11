"""
Optimizer utilities using Optax.
"""

from fabricpc.training.natural_gradients import (
    scale_by_natural_gradient_diag,
    scale_by_natural_gradient_layerwise,
)
from fabricpc.training.carryover import (
    proximal_carryover_euclidean,
    proximal_carryover_fisher,
    update_anchor,
    update_anchor_in_chain,
)

__all__ = [
    "scale_by_natural_gradient_diag",
    "scale_by_natural_gradient_layerwise",
    "proximal_carryover_euclidean",
    "proximal_carryover_fisher",
    "update_anchor",
    "update_anchor_in_chain",
]
