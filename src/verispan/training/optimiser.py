"""
optimiser.py — Optimiser and learning rate scheduler for VeriSpan-RGAT.

Standard recipe for fine-tuning transformers:
    - AdamW with weight decay (decoupled from the adaptive learning rate)
    - Linear warmup for the first `warmup_steps` steps
    - Linear decay to zero over the remaining steps
    - Separate parameter groups: no weight decay on biases and LayerNorm params
"""

from __future__ import annotations

from typing import List, NamedTuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ── Parameter group construction ──────────────────────────────────────────────

# Parameters that should NOT have weight decay applied.
# Applying weight decay to biases and normalisation scales/biases
# hurts performance and is universally omitted in transformer fine-tuning.
_NO_DECAY = {"bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight"}


def build_optimiser(
    model: torch.nn.Module,
    lr: float = 2e-5,
    weight_decay: float = 1e-2,
    eps: float = 1e-8,
) -> AdamW:
    """
    Construct an AdamW optimiser with two parameter groups:
        - Decayed  : all parameters whose name does NOT contain a no-decay key
        - Undecayed: biases and LayerNorm parameters

    Parameters
    ----------
    model : nn.Module
        The full VeriSpanModel.
    lr : float
        Peak learning rate (reached after warmup).
    weight_decay : float
        L2 penalty applied to decayed parameters.
    eps : float
        Adam epsilon for numerical stability.
    """
    decayed, undecayed = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in _NO_DECAY):
            undecayed.append(param)
        else:
            decayed.append(param)

    param_groups = [
        {"params": decayed,   "weight_decay": weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=lr, eps=eps)


# ── Learning rate scheduler ───────────────────────────────────────────────────

def build_scheduler(
    optimiser: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    Linear warmup then linear decay to zero.

    Schedule
    --------
        step < warmup_steps  →  lr = peak_lr × (step / warmup_steps)
        step ≥ warmup_steps  →  lr = peak_lr × (total - step) / (total - warmup)

    This is the canonical HuggingFace get_linear_schedule_with_warmup,
    re-implemented here to avoid importing transformers in the training module.

    Parameters
    ----------
    optimiser : AdamW
        The optimiser whose lr groups will be scheduled.
    warmup_steps : int
        Number of steps over which lr ramps up linearly.
    total_steps : int
        Total number of training steps (epochs × steps_per_epoch).
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimiser, lr_lambda)


# ── Training step count helper ────────────────────────────────────────────────

class StepCounts(NamedTuple):
    steps_per_epoch: int    # number of optimiser steps per epoch
    total_steps: int        # total optimiser steps across all epochs
    warmup_steps: int       # warmup steps (computed from ratio or fixed)


def compute_step_counts(
    dataset_size: int,
    batch_size: int,
    grad_accum_steps: int,
    num_epochs: int,
    warmup_steps: int = 500,
) -> StepCounts:
    """
    Compute training step counts for scheduler construction.

    An 'optimiser step' = one AdamW.step() call, which occurs every
    `grad_accum_steps` forward passes.

    Parameters
    ----------
    dataset_size : int
        Number of training examples.
    batch_size : int
        Per-device batch size (before gradient accumulation).
    grad_accum_steps : int
        Number of forward passes before each optimiser step.
    num_epochs : int
        Total training epochs.
    warmup_steps : int
        Fixed warmup steps (default 500, matching the thesis config).
    """
    steps_per_epoch = (dataset_size + batch_size - 1) // batch_size // grad_accum_steps
    total_steps     = steps_per_epoch * num_epochs
    return StepCounts(
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        warmup_steps=min(warmup_steps, total_steps // 10),
    )
