"""
trainer.py — Training loop for VeriSpan-RGAT.

What this file does
-------------------
    1. Runs the training loop: epochs → batches → forward → loss → backward
    2. Handles fp16 mixed precision via torch.cuda.amp
    3. Handles gradient accumulation (simulates larger batch sizes)
    4. Evaluates on the dev set after every epoch
    5. Logs all metrics to Weights & Biases
    6. Saves checkpoints: best_model.pt (highest dev F1) and last_model.pt

W&B in 30 seconds
-----------------
    Weights & Biases (wandb) is an experiment tracking tool.  You run
    `wandb login` once in your terminal, then every call to wandb.log()
    sends metrics to your online dashboard at wandb.ai.  You can:
        - Plot training curves in real time
        - Compare runs side-by-side
        - Store hyperparameter configs alongside results

    The key calls we make:
        wandb.init(project=..., config=...)   — start a run
        wandb.log({...}, step=...)            — log a dict of metrics
        wandb.finish()                        — end the run cleanly

Usage
-----
    from verispan.training import Trainer, TrainingConfig

    cfg = TrainingConfig(
        output_dir="checkpoints/run1",
        num_epochs=5,
        per_device_batch_size=16,
        grad_accum_steps=4,
    )
    trainer = Trainer(model, train_loader, dev_loader, cfg)
    trainer.train()
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from ..model.verispan import VeriSpanModel
from .optimiser import build_optimiser, build_scheduler, compute_step_counts

logger = logging.getLogger(__name__)


# ── TrainingConfig ────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # I/O
    output_dir:            str   = "checkpoints"
    wandb_project:         str   = "verispan-rgat"
    wandb_run_name:        Optional[str] = None    # None = W&B auto-names the run
    wandb_tags:            list  = field(default_factory=list)

    # Training schedule
    num_epochs:            int   = 5
    per_device_batch_size: int   = 16
    grad_accum_steps:      int   = 4              # effective batch = 16 × 4 = 64
    warmup_steps:          int   = 500

    # Optimiser
    learning_rate:         float = 2e-5
    weight_decay:          float = 1e-2
    max_grad_norm:         float = 1.0            # gradient clipping

    # Evaluation
    eval_every_n_epochs:   int   = 1              # evaluate on dev after N epochs

    # Precision
    fp16:                  bool  = True

    # Logging
    log_every_n_steps:     int   = 50             # log to W&B every N opt steps


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Training orchestrator for VeriSpanModel.

    Parameters
    ----------
    model : VeriSpanModel
        The model to train.  Should already be on the target device.
    train_loader : DataLoader
        Yields batches from ClaimVerificationDataset (training split).
    dev_loader : DataLoader
        Yields batches from ClaimVerificationDataset (dev split).
    config : TrainingConfig
        All training hyperparameters.
    device : torch.device, optional
        Defaults to CUDA if available.
    """

    def __init__(
        self,
        model: VeriSpanModel,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.dev_loader   = dev_loader
        self.config       = config
        self.device       = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Compute step counts for the scheduler
        self._step_counts = compute_step_counts(
            dataset_size=len(train_loader.dataset),
            batch_size=config.per_device_batch_size,
            grad_accum_steps=config.grad_accum_steps,
            num_epochs=config.num_epochs,
            warmup_steps=config.warmup_steps,
        )

        # Optimiser and scheduler
        self.optimiser = build_optimiser(
            model=self.model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = build_scheduler(
            optimiser=self.optimiser,
            warmup_steps=self._step_counts.warmup_steps,
            total_steps=self._step_counts.total_steps,
        )

        # fp16 gradient scaler
        # GradScaler dynamically scales the loss to prevent fp16 underflow,
        # then unscales before gradient clipping and the optimiser step.
        self.scaler = GradScaler(enabled=config.fp16)

        # State
        self._global_step = 0
        self._best_f1     = -1.0

        logger.info(
            f"Trainer ready | device={self.device} | "
            f"total_steps={self._step_counts.total_steps:,} | "
            f"warmup={self._step_counts.warmup_steps}"
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop."""
        self._init_wandb()

        for epoch in range(1, self.config.num_epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch} / {self.config.num_epochs}")
            logger.info(f"{'='*60}")

            train_metrics = self._train_epoch(epoch)
            logger.info(f"Epoch {epoch} train | {_fmt(train_metrics)}")

            if epoch % self.config.eval_every_n_epochs == 0:
                dev_metrics = self._evaluate(epoch)
                logger.info(f"Epoch {epoch} dev   | {_fmt(dev_metrics)}")

                # Log dev metrics to W&B
                # wandb.log() sends a dict of {metric_name: value} to the
                # dashboard.  The `step` argument pins it to a global step
                # on the x-axis of all plots.
                wandb.log(
                    {f"dev/{k}": v for k, v in dev_metrics.items()},
                    step=self._global_step,
                )

                # Checkpoint: always save latest, save best on F1 improvement
                self._save_checkpoint("last_model.pt", epoch, dev_metrics)

                verdict_f1 = dev_metrics.get("verdict_f1_macro", 0.0)
                if verdict_f1 > self._best_f1:
                    self._best_f1 = verdict_f1
                    self._save_checkpoint("best_model.pt", epoch, dev_metrics)
                    logger.info(
                        f"  ✓ New best verdict F1: {verdict_f1:.4f} "
                        f"— saved best_model.pt"
                    )

        wandb.finish()
        logger.info(f"\nTraining complete.  Best dev verdict F1: {self._best_f1:.4f}")

    # ── Training epoch ────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.optimiser.zero_grad()

        accum_loss        = 0.0
        accum_span_loss   = 0.0
        accum_verdict_loss = 0.0
        accum_contrast_loss = 0.0
        n_accum_batches   = 0
        epoch_start       = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = _to_device(batch, self.device)

            # ── Forward + loss (inside autocast for fp16) ─────────────────
            # autocast automatically uses fp16 for eligible ops (matmul,
            # conv) and fp32 for others (softmax, loss).  It's a context
            # manager — everything inside runs in mixed precision.
            with torch.amp.autocast(device_type='cuda', enabled=self.config.fp16):
                output, loss_out = self.model.forward_and_loss(batch)
                # Divide loss by grad_accum_steps so the accumulated
                # gradient equals the gradient of the mean loss over
                # the effective batch.
                loss = loss_out.total / self.config.grad_accum_steps

            # ── Backward (scaler handles fp16 loss scaling) ───────────────
            # scaler.scale(loss) multiplies loss by a dynamic scale factor
            # before .backward() to prevent fp16 gradient underflow.
            self.scaler.scale(loss).backward()

            accum_loss         += loss_out.total.item()
            accum_span_loss    += loss_out.span.item()
            accum_verdict_loss += loss_out.verdict.item()
            accum_contrast_loss += loss_out.contrast.item()
            n_accum_batches    += 1

            # ── Optimiser step every grad_accum_steps batches ─────────────
            is_accum_step = (batch_idx + 1) % self.config.grad_accum_steps == 0
            is_last_batch = (batch_idx + 1) == len(self.train_loader)

            if is_accum_step or is_last_batch:
                # Unscale gradients before clipping
                # scaler.unscale_() reverses the loss scaling on the
                # gradients so that max_grad_norm applies to the true values.
                self.scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                # scaler.step() skips the update if gradients contain inf/nan
                # (which can happen transiently in fp16) and adjusts the scale.
                self.scaler.step(self.optimiser)
                self.scaler.update()
                self.scheduler.step()
                self.optimiser.zero_grad()
                self._global_step += 1

                # ── Log to W&B ────────────────────────────────────────────
                if self._global_step % self.config.log_every_n_steps == 0:
                    avg = 1.0 / n_accum_batches
                    step_metrics = {
                        "train/loss":          accum_loss * avg,
                        "train/loss_span":     accum_span_loss * avg,
                        "train/loss_verdict":  accum_verdict_loss * avg,
                        "train/loss_contrast": accum_contrast_loss * avg,
                        "train/lr":            self.scheduler.get_last_lr()[0],
                        "train/epoch":         epoch,
                    }
                    # wandb.log sends these values to your W&B dashboard.
                    # On wandb.ai you'll see real-time curves for each key.
                    wandb.log(step_metrics, step=self._global_step)

                    accum_loss = accum_span_loss = 0.0
                    accum_verdict_loss = accum_contrast_loss = 0.0
                    n_accum_batches = 0

        epoch_time = time.time() - epoch_start
        return {"epoch_time_s": epoch_time}

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self, epoch: int) -> Dict[str, float]:
        """
        Run inference on the dev set and compute:
            - Verdict macro F1, precision, recall (per class)
            - Span token-level F1
            - Mean loss terms
        """
        self.model.eval()

        all_verdict_preds = []
        all_verdict_labels = []
        all_span_preds  = []    # flattened doc-token predictions (0/1)
        all_span_labels = []    # flattened doc-token ground truth (0/1)

        total_loss = total_span = total_verdict = total_contrast = 0.0
        n_batches = 0

        for batch in self.dev_loader:
            batch = _to_device(batch, self.device)

            with torch.amp.autocast(device_type='cuda', enabled=self.config.fp16):
                output, loss_out = self.model.forward_and_loss(batch)

            total_loss     += loss_out.total.item()
            total_span     += loss_out.span.item()
            total_verdict  += loss_out.verdict.item()
            total_contrast += loss_out.contrast.item()
            n_batches      += 1

            # Verdict predictions
            preds = output.verdict_logits.argmax(dim=-1)   # [B]
            all_verdict_preds.extend(preds.cpu().tolist())
            all_verdict_labels.extend(batch["verdict_labels"].cpu().tolist())

            # Span predictions (doc tokens only, span_labels != -1)
            sp_labels = batch["span_labels"]               # [B, L]
            sp_probs  = output.span_probs                  # [B, L]
            valid     = sp_labels != -1.0
            all_span_preds.extend(
                (sp_probs[valid] > 0.5).long().cpu().tolist()
            )
            all_span_labels.extend(
                sp_labels[valid].long().cpu().tolist()
            )

        n = n_batches or 1
        metrics = {
            # Loss terms
            "loss":          total_loss / n,
            "loss_span":     total_span / n,
            "loss_verdict":  total_verdict / n,
            "loss_contrast": total_contrast / n,
            # Verdict classification
            "verdict_f1_macro":        f1_score(all_verdict_labels, all_verdict_preds, average="macro", zero_division=0),
            "verdict_precision_macro": precision_score(all_verdict_labels, all_verdict_preds, average="macro", zero_division=0),
            "verdict_recall_macro":    recall_score(all_verdict_labels, all_verdict_preds, average="macro", zero_division=0),
            "verdict_f1_supports":     f1_score(all_verdict_labels, all_verdict_preds, labels=[0], average="macro", zero_division=0),
            "verdict_f1_refutes":      f1_score(all_verdict_labels, all_verdict_preds, labels=[1], average="macro", zero_division=0),
            "verdict_f1_nei":          f1_score(all_verdict_labels, all_verdict_preds, labels=[2], average="macro", zero_division=0),
            # Span extraction
            "span_f1":       f1_score(all_span_labels, all_span_preds, average="binary", zero_division=0),
        }
        return metrics

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """
        Save model weights, optimiser state, scheduler state, and metadata.

        Saving the optimiser + scheduler state means you can resume
        training from a checkpoint without restarting the lr schedule.
        """
        path = self.output_dir / filename
        torch.save(
            {
                "epoch":           epoch,
                "global_step":     self._global_step,
                "best_f1":         self._best_f1,
                "model_state":     self.model.state_dict(),
                "optimiser_state": self.optimiser.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state":    self.scaler.state_dict(),
                "metrics":         metrics,
                "config":          asdict(self.model.config),
            },
            path,
        )
        logger.info(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> int:
        """
        Resume training from a checkpoint.  Returns the epoch to resume from.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimiser.load_state_dict(ckpt["optimiser_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self._global_step = ckpt["global_step"]
        self._best_f1     = ckpt["best_f1"]
        logger.info(
            f"Resumed from {path} | epoch={ckpt['epoch']} | "
            f"global_step={self._global_step} | best_f1={self._best_f1:.4f}"
        )
        return ckpt["epoch"]

    # ── W&B initialisation ────────────────────────────────────────────────────

    def _init_wandb(self) -> None:
        """
        Initialise a W&B run.

        wandb.init() does three things:
            1. Creates a new run on your wandb.ai account
            2. Stores the config dict (hyperparameters) alongside the run
            3. Returns a run object you can use to log artifacts, etc.

        After this call, every wandb.log({...}) call in the training loop
        sends data to that run's dashboard page.

        First-time setup (run once in your terminal before training):
            pip install wandb
            wandb login          ← paste your API key from wandb.ai/settings
        """
        config_dict = {
            **asdict(self.config),
            **asdict(self.model.config),
            "total_steps":    self._step_counts.total_steps,
            "warmup_steps":   self._step_counts.warmup_steps,
            "train_examples": len(self.train_loader.dataset),
            "dev_examples":   len(self.dev_loader.dataset),
        }
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            tags=self.config.wandb_tags,
            config=config_dict,
            # resume="allow" lets W&B reconnect to an existing run if the
            # run name matches — useful when resuming from a checkpoint.
            resume="allow",
        )
        # wandb.watch() hooks into PyTorch to log gradient histograms and
        # parameter norms every `log_freq` steps.  Useful for debugging
        # vanishing/exploding gradients.
        wandb.watch(self.model, log=None, log_freq=200)
        logger.info(f"W&B run initialised: {wandb.run.url}")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensor values in a batch dict to the target device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _fmt(metrics: Dict[str, float]) -> str:
    """Format a metrics dict as a compact string for console logging."""
    return " | ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in metrics.items()
    )
