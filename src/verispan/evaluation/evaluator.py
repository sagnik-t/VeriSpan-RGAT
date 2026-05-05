"""
evaluator.py — Standalone evaluator for VeriSpan-RGAT.

Used both by the trainer (in-loop dev evaluation) and by the evaluation
script for zero-shot cross-domain eval on SciFact and WiCE.

Usage (programmatic)
--------------------
    evaluator = Evaluator.from_checkpoint("checkpoints/best_model.pt")
    metrics   = evaluator.evaluate(scifact_loader)

Usage (via scripts/evaluate.py)
--------------------------------
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best_model.pt \\
        --dataset scifact \\
        --data_dir data/raw/scifact \\
        --output_dir results/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from ..model.verispan import VeriSpanConfig, VeriSpanModel
from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """All metrics from one evaluation pass."""
    dataset:  str
    split:    str
    metrics:  Dict[str, float]
    n_examples: int

    # Raw collected outputs — useful for error analysis / visualisation
    verdict_preds:  List[int]  = field(default_factory=list, repr=False)
    verdict_labels: List[int]  = field(default_factory=list, repr=False)

    def summary(self) -> str:
        """Human-readable one-liner for console output."""
        m = self.metrics
        return (
            f"{self.dataset}/{self.split}  n={self.n_examples}  "
            f"verdict_F1={m.get('verdict_f1_macro', 0):.4f}  "
            f"span_F1={m.get('span_f1', 0):.4f}  "
            f"CCS={m.get('ccs', 0):.4f}  "
            f"SBA={m.get('sba', 0):.4f}"
        )

    def to_dict(self) -> Dict:
        return {
            "dataset":    self.dataset,
            "split":      self.split,
            "n_examples": self.n_examples,
            "metrics":    self.metrics,
        }


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs inference on a DataLoader and returns EvalResult.

    Parameters
    ----------
    model   : VeriSpanModel (already on device, in eval mode)
    device  : torch.device
    fp16    : whether to use autocast during inference
    threshold : span binarisation threshold
    """

    def __init__(
        self,
        model:     VeriSpanModel,
        device:    Optional[torch.device] = None,
        fp16:      bool = True,
        threshold: float = 0.5,
    ) -> None:
        self.model     = model
        self.device    = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.fp16      = fp16 and self.device.type == "cuda"
        self.threshold = threshold
        self.model.to(self.device)
        self.model.eval()

    # ── Core evaluation loop ──────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        dataset_name: str = "unknown",
        split:        str = "eval",
    ) -> EvalResult:
        """
        Run full evaluation pass.

        Parameters
        ----------
        dataloader   : DataLoader over ClaimVerificationDataset
        dataset_name : label for the result (e.g. 'scifact', 'wice', 'fever')
        split        : split label (e.g. 'dev', 'test')

        Returns
        -------
        EvalResult
        """
        all_verdict_preds:  List[int]          = []
        all_verdict_labels: List[int]          = []
        all_span_probs:     List[np.ndarray]   = []  # per-example [L]
        all_span_labels:    List[np.ndarray]   = []  # per-example [L]

        for batch in dataloader:
            batch = _to_device(batch, self.device)

            with autocast(enabled=self.fp16):
                output, _ = self.model.forward_and_loss(batch)

            # Verdict
            preds = output.verdict_logits.argmax(dim=-1)  # [B]
            all_verdict_preds.extend(preds.cpu().tolist())
            all_verdict_labels.extend(batch["verdict_labels"].cpu().tolist())

            # Span — keep per-example arrays (variable length before padding)
            sp_probs  = output.span_probs.cpu().numpy()   # [B, L]
            sp_labels = batch["span_labels"].cpu().numpy() # [B, L]
            for b in range(sp_probs.shape[0]):
                all_span_probs.append(sp_probs[b])
                all_span_labels.append(sp_labels[b])

        # Stack into 2-D arrays (all sequences share the same padded length
        # within a batch but may differ across batches — pad to max length)
        max_len = max(a.shape[0] for a in all_span_probs)
        B_total = len(all_span_probs)

        span_probs_2d  = np.full((B_total, max_len), fill_value=0.0,  dtype=np.float32)
        span_labels_2d = np.full((B_total, max_len), fill_value=-1.0, dtype=np.float32)

        for i, (sp, sl) in enumerate(zip(all_span_probs, all_span_labels)):
            L = sp.shape[0]
            span_probs_2d[i, :L]  = sp
            span_labels_2d[i, :L] = sl

        metrics = compute_all_metrics(
            verdict_preds  = all_verdict_preds,
            verdict_labels = all_verdict_labels,
            span_probs     = span_probs_2d,
            span_labels    = span_labels_2d,
            threshold      = self.threshold,
        )

        return EvalResult(
            dataset         = dataset_name,
            split           = split,
            metrics         = metrics,
            n_examples      = B_total,
            verdict_preds   = all_verdict_preds,
            verdict_labels  = all_verdict_labels,
        )

    # ── Checkpoint loading ────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        fp16:   bool = True,
        threshold: float = 0.5,
    ) -> "Evaluator":
        """
        Build an Evaluator from a saved checkpoint.

        The checkpoint must have been saved by Trainer._save_checkpoint()
        and include 'model_state' and 'config' keys.
        """
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        ckpt = torch.load(checkpoint_path, map_location=device)

        config = VeriSpanConfig(**ckpt["config"])
        model  = VeriSpanModel(config)
        model.load_state_dict(ckpt["model_state"])

        logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(epoch={ckpt.get('epoch', '?')}, "
            f"best_f1={ckpt.get('best_f1', 0):.4f})"
        )
        return cls(model=model, device=device, fp16=fp16, threshold=threshold)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
