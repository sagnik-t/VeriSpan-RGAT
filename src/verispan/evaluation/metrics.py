"""
metrics.py — Pure metric functions for VeriSpan-RGAT evaluation.

Metrics
-------
    compute_ccs        : Claim Coverage Score   (token-level recall variant)
    compute_sba        : Span Boundary Accuracy (boundary-error metric)
    compute_span_f1    : Standard binary token-level F1
    compute_verdict_metrics : Macro P/R/F1 + per-class F1
    compute_all_metrics     : All metrics in one pass

All functions operate on Python lists or 1-D / 2-D tensors and return
plain floats or dicts — no model dependencies.

CCS formula (from thesis §14.3)
--------------------------------
    CCS = |T ∩ P| / |T|
    T = ground-truth evidence token set
    P = predicted evidence token set
    CCS ∈ [0, 1];  undefined (→ NaN, skipped) when |T| = 0.

SBA formula (from thesis §14.4)
---------------------------------
    E = |î − i| + |ĵ − j|   (total boundary error)
    L = j − i + 1            (gold span length, inclusive)
    SBA = max(0, 1 − E / L)

    Span extraction: contiguous runs of predicted-positive tokens.
    Matching: for each gold span, find the predicted span that minimises E.
    If no predicted spans exist, SBA = 0 for that gold span.
    Final SBA is the mean over all gold spans across all examples.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


# ── Span extraction ───────────────────────────────────────────────────────────

def extract_contiguous_spans(binary: Sequence[int]) -> List[Tuple[int, int]]:
    """
    Extract contiguous (start, end) inclusive spans from a binary sequence.

    Parameters
    ----------
    binary : sequence of 0/1 ints
        Token-level predictions or labels.

    Returns
    -------
    spans : list of (start, end)
        Inclusive index pairs.

    Examples
    --------
    >>> extract_contiguous_spans([0, 1, 1, 0, 1])
    [(1, 2), (4, 4)]
    """
    spans: List[Tuple[int, int]] = []
    in_span = False
    start = 0
    for i, val in enumerate(binary):
        if val and not in_span:
            in_span = True
            start = i
        elif not val and in_span:
            spans.append((start, i - 1))
            in_span = False
    if in_span:
        spans.append((start, len(binary) - 1))
    return spans


# ── CCS ───────────────────────────────────────────────────────────────────────

def compute_ccs(
    span_probs: np.ndarray,     # [B, L]  sigmoid probabilities
    span_labels: np.ndarray,    # [B, L]  ground truth; -1 = ignore
    threshold: float = 0.5,
) -> float:
    """
    Claim Coverage Score averaged over examples that have at least one
    positive gold token.

    CCS = |T ∩ P| / |T|

    Parameters
    ----------
    span_probs  : float array [B, L]
    span_labels : int/float array [B, L];  -1 = claim token or padding (ignored)
    threshold   : binarisation threshold for span_probs

    Returns
    -------
    float : mean CCS over examples with |T| > 0; 0.0 if no such examples.
    """
    preds  = (span_probs > threshold).astype(int)   # [B, L]
    labels = span_labels.astype(int)                 # [B, L]

    scores: List[float] = []
    for b in range(labels.shape[0]):
        valid = labels[b] != -1             # doc tokens only
        t     = labels[b][valid]            # gold 0/1
        p     = preds[b][valid]             # pred 0/1

        n_gold = t.sum()
        if n_gold == 0:
            continue                         # NEI example — skip

        intersection = int((t & p).sum())
        scores.append(intersection / int(n_gold))

    return float(np.mean(scores)) if scores else 0.0


# ── SBA ───────────────────────────────────────────────────────────────────────

def _sba_single_pair(
    gold_start: int, gold_end: int,
    pred_start: int, pred_end: int,
) -> float:
    """SBA for one gold span matched to one predicted span."""
    L = gold_end - gold_start + 1          # gold span length (≥ 1)
    E = abs(pred_start - gold_start) + abs(pred_end - gold_end)
    return max(0.0, 1.0 - E / L)


def _best_sba_for_gold(
    gold_start: int,
    gold_end: int,
    pred_spans: List[Tuple[int, int]],
) -> float:
    """
    Match one gold span to the predicted span that minimises boundary error.
    Returns SBA = 0 if there are no predictions.
    """
    if not pred_spans:
        return 0.0
    scores = [
        _sba_single_pair(gold_start, gold_end, ps, pe)
        for ps, pe in pred_spans
    ]
    return max(scores)


def compute_sba(
    span_probs: np.ndarray,     # [B, L]
    span_labels: np.ndarray,    # [B, L]; -1 = ignore
    threshold: float = 0.5,
) -> float:
    """
    Span Boundary Accuracy averaged over gold spans across all examples.

    For each gold span (i, j), find the predicted span (î, ĵ) that minimises
    boundary error E, then SBA = max(0, 1 − E / L).

    Parameters
    ----------
    span_probs  : float array [B, L]
    span_labels : int/float array [B, L]; -1 = ignore
    threshold   : binarisation threshold

    Returns
    -------
    float : mean SBA; 0.0 if no gold spans exist.
    """
    preds  = (span_probs > threshold).astype(int)
    labels = span_labels.astype(int)

    all_scores: List[float] = []

    for b in range(labels.shape[0]):
        valid_mask = labels[b] != -1
        valid_idxs = np.where(valid_mask)[0]

        if len(valid_idxs) == 0:
            continue

        # Remap to positions within the valid (doc) region for span extraction,
        # then translate extracted spans back to full-sequence token indices.
        t_local = labels[b][valid_mask]
        p_local = preds[b][valid_mask]

        gold_spans_local = extract_contiguous_spans(t_local.tolist())
        pred_spans_local = extract_contiguous_spans(p_local.tolist())

        if not gold_spans_local:
            continue                          # NEI — no gold spans

        for gs, ge in gold_spans_local:
            score = _best_sba_for_gold(gs, ge, pred_spans_local)
            all_scores.append(score)

    return float(np.mean(all_scores)) if all_scores else 0.0


# ── Span F1 ───────────────────────────────────────────────────────────────────

def compute_span_f1(
    span_probs: np.ndarray,     # [B, L]
    span_labels: np.ndarray,    # [B, L]; -1 = ignore
    threshold: float = 0.5,
) -> float:
    """Binary token-level F1 over document tokens (span_labels ≠ −1)."""
    valid   = span_labels != -1
    preds   = (span_probs[valid] > threshold).astype(int)
    targets = span_labels[valid].astype(int)

    if targets.sum() == 0 and preds.sum() == 0:
        return 1.0
    return float(f1_score(targets, preds, average="binary", zero_division=0))


# ── Verdict metrics ───────────────────────────────────────────────────────────

_VERDICT_LABELS = [0, 1, 2]
_VERDICT_NAMES  = {0: "supports", 1: "refutes", 2: "nei"}


def compute_verdict_metrics(
    preds:  List[int],
    labels: List[int],
) -> Dict[str, float]:
    """
    Macro P / R / F1 and per-class F1.

    Parameters
    ----------
    preds  : predicted verdict ids (0/1/2)
    labels : ground-truth verdict ids

    Returns
    -------
    dict with keys:
        verdict_f1_macro, verdict_precision_macro, verdict_recall_macro
        verdict_f1_supports, verdict_f1_refutes, verdict_f1_nei
    """
    preds_a  = np.asarray(preds)
    labels_a = np.asarray(labels)

    metrics: Dict[str, float] = {
        "verdict_f1_macro":        float(f1_score(labels_a, preds_a, average="macro",  zero_division=0)),
        "verdict_precision_macro": float(precision_score(labels_a, preds_a, average="macro", zero_division=0)),
        "verdict_recall_macro":    float(recall_score(labels_a, preds_a, average="macro",    zero_division=0)),
    }
    per_class = f1_score(labels_a, preds_a, labels=_VERDICT_LABELS, average=None, zero_division=0)
    for label_id, name in _VERDICT_NAMES.items():
        metrics[f"verdict_f1_{name}"] = float(per_class[label_id])
    return metrics


# ── Combined ─────────────────────────────────────────────────────────────────

def compute_all_metrics(
    verdict_preds:  List[int],
    verdict_labels: List[int],
    span_probs:     np.ndarray,     # [B, L]
    span_labels:    np.ndarray,     # [B, L]; -1 = ignore
    threshold:      float = 0.5,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics in one call.

    Returns
    -------
    dict with all verdict, span_f1, CCS, and SBA metrics.
    """
    metrics = compute_verdict_metrics(verdict_preds, verdict_labels)
    metrics["span_f1"] = compute_span_f1(span_probs, span_labels, threshold)
    metrics["ccs"]     = compute_ccs(span_probs, span_labels, threshold)
    metrics["sba"]     = compute_sba(span_probs, span_labels, threshold)
    return metrics
