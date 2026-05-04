"""
verispan.training — Training loop for VeriSpan-RGAT.

Public API
----------
    from verispan.training import Trainer, TrainingConfig
    from verispan.training import build_optimiser, build_scheduler, compute_step_counts
"""

from .trainer import Trainer, TrainingConfig
from .optimiser import build_optimiser, build_scheduler, compute_step_counts

__all__ = [
    "Trainer",
    "TrainingConfig",
    "build_optimiser",
    "build_scheduler",
    "compute_step_counts",
]
