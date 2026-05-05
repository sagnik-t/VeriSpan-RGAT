"""
tests/evaluation/test_metrics.py

Unit tests for verispan.evaluation.metrics.

All functions are pure numpy/sklearn — no model, no GPU, no fixtures needed.
Every test runs in milliseconds.

Test inventory
--------------
    TestExtractContiguousSpans
        test_empty_sequence
        test_single_span
        test_multiple_spans
        test_span_at_boundaries
        test_all_positive
        test_all_negative

    TestCCS
        test_perfect_coverage      — P == T → CCS = 1.0
        test_zero_coverage         — P ∩ T = ∅ → CCS = 0.0
        test_partial_coverage      — partial overlap
        test_skips_nei_examples    — examples with |T|=0 excluded from average
        test_all_nei_returns_zero  — if no non-NEI examples, return 0.0
        test_over_prediction_ok    — CCS ignores false positives (by design)

    TestSBA
        test_exact_match           — SBA = 1.0 on perfect prediction
        test_full_boundary_error   — E ≥ L → SBA = 0.0 (floor at 0)
        test_partial_boundary_error — 0 < SBA < 1 on off-by-k boundaries
        test_no_predictions        — SBA = 0.0 when model predicts nothing
        test_multiple_gold_spans   — mean over all gold spans

    TestSpanF1
        test_perfect_f1            — identical pred and gold
        test_zero_f1               — no overlap
        test_ignore_index_excluded — tokens with label=-1 not counted

    TestVerdictMetrics
        test_perfect_classification
        test_all_wrong
        test_per_class_keys_present
        test_macro_averages_range

    TestComputeAllMetrics
        test_returns_all_keys
        test_values_in_range
"""

from __future__ import annotations

import numpy as np
import pytest

from verispan.evaluation.metrics import (
    compute_all_metrics,
    compute_ccs,
    compute_sba,
    compute_span_f1,
    compute_verdict_metrics,
    extract_contiguous_spans,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _probs_labels(
    B: int,
    L: int,
    gold_positions: list,   # list of lists of token indices that are gold positive
    pred_positions: list,   # list of lists of token indices predicted positive
    doc_start: int = 2,     # tokens before doc_start are claim tokens (label=-1)
) -> tuple:
    """
    Build (span_probs [B,L], span_labels [B,L]) arrays.
    Tokens before doc_start get label=-1 (claim/CLS/SEP).
    """
    labels = np.full((B, L), fill_value=-1.0, dtype=np.float32)
    probs  = np.zeros((B, L), dtype=np.float32)

    for b in range(B):
        labels[b, doc_start:] = 0.0
        for pos in gold_positions[b]:
            labels[b, pos] = 1.0
        for pos in pred_positions[b]:
            probs[b, pos] = 1.0   # probability 1.0 → always above threshold

    return probs, labels


# ── extract_contiguous_spans ─────────────────────────────────────────────────

class TestExtractContiguousSpans:

    def test_empty_sequence(self):
        assert extract_contiguous_spans([]) == []

    def test_single_span(self):
        assert extract_contiguous_spans([0, 1, 1, 0]) == [(1, 2)]

    def test_multiple_spans(self):
        assert extract_contiguous_spans([1, 0, 1, 1, 0]) == [(0, 0), (2, 3)]

    def test_span_at_boundaries(self):
        assert extract_contiguous_spans([1, 0, 0, 1]) == [(0, 0), (3, 3)]

    def test_all_positive(self):
        assert extract_contiguous_spans([1, 1, 1]) == [(0, 2)]

    def test_all_negative(self):
        assert extract_contiguous_spans([0, 0, 0]) == []


# ── CCS ───────────────────────────────────────────────────────────────────────

class TestCCS:

    def test_perfect_coverage(self):
        # Gold and pred identical → CCS = 1.0
        probs, labels = _probs_labels(
            B=1, L=6,
            gold_positions=[[3, 4]],
            pred_positions=[[3, 4]],
        )
        assert compute_ccs(probs, labels) == pytest.approx(1.0)

    def test_zero_coverage(self):
        probs, labels = _probs_labels(
            B=1, L=6,
            gold_positions=[[3, 4]],
            pred_positions=[[]],        # model predicts nothing
        )
        assert compute_ccs(probs, labels) == pytest.approx(0.0)

    def test_partial_coverage(self):
        # Gold = {3,4,5}, Pred = {3,4} → CCS = 2/3
        probs, labels = _probs_labels(
            B=1, L=7,
            gold_positions=[[3, 4, 5]],
            pred_positions=[[3, 4]],
        )
        assert compute_ccs(probs, labels) == pytest.approx(2 / 3, rel=1e-4)

    def test_skips_nei_examples(self):
        # Example 0: normal evidence. Example 1: NEI (no gold tokens).
        # CCS should average only over example 0.
        probs  = np.array([[0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0]], dtype=np.float32)
        labels = np.array([[-1, -1, 1, 1, 0],
                            [-1, -1, 0, 0, 0]], dtype=np.float32)
        ccs = compute_ccs(probs, labels)
        assert ccs == pytest.approx(1.0)

    def test_all_nei_returns_zero(self):
        # No example has any gold positive token
        labels = np.array([[-1, -1, 0, 0]], dtype=np.float32)
        probs  = np.zeros_like(labels)
        assert compute_ccs(probs, labels) == pytest.approx(0.0)

    def test_over_prediction_doesnt_hurt(self):
        # CCS = |T ∩ P| / |T|; extra predictions don't reduce the score
        probs, labels = _probs_labels(
            B=1, L=6,
            gold_positions=[[3]],
            pred_positions=[[2, 3, 4]],  # over-predict
        )
        assert compute_ccs(probs, labels) == pytest.approx(1.0)


# ── SBA ───────────────────────────────────────────────────────────────────────

class TestSBA:

    def test_exact_match(self):
        # Gold span (2,4), pred span (2,4) → E=0, L=3, SBA=1.0
        probs  = np.array([[0, 0, 1, 1, 1, 0]], dtype=np.float32)
        labels = np.array([[-1, -1, 1, 1, 1, 0]], dtype=np.float32)
        # doc_start=2 (first two tokens are claim)
        assert compute_sba(probs, labels) == pytest.approx(1.0)

    def test_full_boundary_error(self):
        # Gold span length 1, boundary error ≥ L → SBA = 0
        # Gold: token 2 only (L=1); Pred: token 5 (E=3 ≥ L=1)
        probs  = np.array([[0, 0, 0, 0, 0, 1]], dtype=np.float32)
        labels = np.array([[-1, -1, 1, 0, 0, 0]], dtype=np.float32)
        sba = compute_sba(probs, labels)
        assert sba == pytest.approx(0.0)

    def test_partial_boundary_error(self):
        # Gold (2,4), L=3; Pred (3,5), E=|3-2|+|5-4|=2; SBA=max(0,1-2/3)=1/3
        probs  = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.float32)
        labels = np.array([[-1, -1, 1, 1, 1, 0]], dtype=np.float32)
        sba = compute_sba(probs, labels)
        assert sba == pytest.approx(1 / 3, rel=1e-4)

    def test_no_predictions_returns_zero(self):
        probs  = np.zeros((1, 6), dtype=np.float32)
        labels = np.array([[-1, -1, 1, 1, 0, 0]], dtype=np.float32)
        assert compute_sba(probs, labels) == pytest.approx(0.0)

    def test_multiple_gold_spans_averaged(self):
        # Two gold spans, both exactly predicted → mean SBA = 1.0
        probs  = np.array([[0, 0, 1, 0, 1, 0]], dtype=np.float32)
        labels = np.array([[-1, -1, 1, 0, 1, 0]], dtype=np.float32)
        assert compute_sba(probs, labels) == pytest.approx(1.0)


# ── Span F1 ───────────────────────────────────────────────────────────────────

class TestSpanF1:

    def test_perfect_f1(self):
        probs  = np.array([[0, 0, 1, 1, 0]], dtype=np.float32)
        labels = np.array([[-1, -1, 1, 1, 0]], dtype=np.float32)
        assert compute_span_f1(probs, labels) == pytest.approx(1.0)

    def test_zero_f1(self):
        probs  = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
        labels = np.array([[-1, -1, 1, 1, 0]], dtype=np.float32)
        assert compute_span_f1(probs, labels) == pytest.approx(0.0)

    def test_ignore_index_excluded(self):
        # All gold positives are in the 'claim' region (label=-1); should be ignored
        probs  = np.array([[1, 1, 0, 0]], dtype=np.float32)
        labels = np.array([[-1, -1, 0, 0]], dtype=np.float32)
        # After masking, no gold positives and no pred positives → F1 = 1.0
        assert compute_span_f1(probs, labels) == pytest.approx(1.0)


# ── Verdict metrics ───────────────────────────────────────────────────────────

class TestVerdictMetrics:

    def test_perfect_classification(self):
        labels = [0, 1, 2, 0, 1]
        preds  = [0, 1, 2, 0, 1]
        m = compute_verdict_metrics(preds, labels)
        assert m["verdict_f1_macro"]    == pytest.approx(1.0)
        assert m["verdict_f1_supports"] == pytest.approx(1.0)
        assert m["verdict_f1_refutes"]  == pytest.approx(1.0)
        assert m["verdict_f1_nei"]      == pytest.approx(1.0)

    def test_all_wrong_macro_below_one(self):
        labels = [0, 1, 2]
        preds  = [1, 2, 0]     # every prediction is wrong
        m = compute_verdict_metrics(preds, labels)
        assert m["verdict_f1_macro"] < 0.1

    def test_per_class_keys_present(self):
        m = compute_verdict_metrics([0, 1, 2], [0, 1, 2])
        expected_keys = {
            "verdict_f1_macro",
            "verdict_precision_macro",
            "verdict_recall_macro",
            "verdict_f1_supports",
            "verdict_f1_refutes",
            "verdict_f1_nei",
        }
        assert expected_keys.issubset(m.keys())

    def test_macro_averages_in_range(self):
        m = compute_verdict_metrics([0, 1, 2, 0], [0, 1, 0, 2])
        for k in ("verdict_f1_macro", "verdict_precision_macro", "verdict_recall_macro"):
            assert 0.0 <= m[k] <= 1.0


# ── compute_all_metrics ───────────────────────────────────────────────────────

class TestComputeAllMetrics:

    def _make_inputs(self):
        B, L = 2, 6
        probs  = np.array([[0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0]], dtype=np.float32)
        labels = np.array([[-1, -1, 1, 1, 0, 0],
                            [-1, -1, 0, 1, 0, 0]], dtype=np.float32)
        return probs, labels

    def test_returns_all_keys(self):
        probs, labels = self._make_inputs()
        m = compute_all_metrics(
            verdict_preds=[0, 1], verdict_labels=[0, 1],
            span_probs=probs, span_labels=labels,
        )
        for key in ("verdict_f1_macro", "span_f1", "ccs", "sba"):
            assert key in m, f"Missing key: {key}"

    def test_values_in_valid_range(self):
        probs, labels = self._make_inputs()
        m = compute_all_metrics(
            verdict_preds=[0, 1], verdict_labels=[0, 1],
            span_probs=probs, span_labels=labels,
        )
        for k, v in m.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v} out of [0,1]"
