"""
span_head.py — Token-level span extraction head (Stage 2).

Applies a linear classifier over document-side token embeddings to produce
per-token span membership probabilities.  A separate utility function
(`extract_candidate_spans`) converts those probabilities into discrete
(start, end) span positions used as graph nodes in Stage 3.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class SpanExtractionHead(nn.Module):
    """
    Token-level binary classifier over document tokens.

    For each document-side token h_i, computes:
        p_i = σ(W_s h_i + b_s)

    Non-document positions are zeroed in `probs` so they never influence
    span extraction; they are also masked in the BCE loss via span_labels == -1.

    Parameters
    ----------
    hidden_dim : int
        Encoder hidden size (768 for DeBERTa-v3-small).
    dropout : float
        Applied to token embeddings before projection.
    """

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        H: torch.Tensor,               # [B, L, d]
        doc_token_mask: torch.Tensor,  # [B, L] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : Tensor [B, L]
            Raw pre-sigmoid scores (used directly in BCE loss).
        probs  : Tensor [B, L]
            Sigmoid probabilities.  Non-document positions are zeroed.
        """
        logits = self.linear(self.dropout(H)).squeeze(-1)  # [B, L]
        probs = torch.sigmoid(logits) * doc_token_mask.float()
        return logits, probs


# ─────────────────────────────────────────────────────────────────────────────
# Span extraction from token probabilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_candidate_spans(
    probs: torch.Tensor,       # [seq_len] — single example
    mask: torch.Tensor,        # [seq_len] bool — claim or doc tokens
    threshold: float = 0.5,
    min_span_len: int = 1,
    max_span_len: int = 30,
    min_spans: int = 1,
) -> List[Tuple[int, int]]:
    """
    Convert per-token probabilities into a list of candidate (start, end) spans.

    Algorithm
    ---------
    1. Identify contiguous runs of valid tokens with p_i > threshold.
    2. Split runs longer than max_span_len.
    3. Drop runs shorter than min_span_len.
    4. Fallback: if fewer than min_spans remain, insert a single span
       centred on the peak-scoring valid token.

    Parameters
    ----------
    probs : Tensor [seq_len]
        Per-token probabilities from SpanExtractionHead.
    mask : Tensor [seq_len] bool
        Restricts extraction to valid positions (doc or claim tokens only).
    threshold : float
        Minimum probability to include a token.
    min_span_len : int
        Minimum span length in tokens.
    max_span_len : int
        Maximum span length before splitting.
    min_spans : int
        Minimum number of spans to return.

    Returns
    -------
    List of (start, end) **inclusive** token index pairs.
    """
    valid = mask.bool().cpu()
    p = probs.detach().cpu()

    above = (p > threshold) & valid
    spans = _group_contiguous(above, min_span_len, max_span_len)

    if len(spans) < min_spans:
        fb = _fallback_span(p, valid, max_span_len // 4)
        if fb is not None and fb not in spans:
            spans.insert(0, fb)

    return spans


def _group_contiguous(
    above: torch.Tensor,   # [seq_len] bool
    min_len: int,
    max_len: int,
) -> List[Tuple[int, int]]:
    """Group consecutive True positions; split at max_len; drop below min_len."""
    spans: List[Tuple[int, int]] = []
    n = above.size(0)
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        # Walk to end of run
        j = i
        while j < n and above[j]:
            j += 1
        run_end = j - 1  # inclusive

        # Split the run into max_len-sized chunks
        start = i
        while start <= run_end:
            end = min(start + max_len - 1, run_end)
            if (end - start + 1) >= min_len:
                spans.append((start, end))
            start = end + 1

        i = j
    return spans


def _fallback_span(
    probs: torch.Tensor,
    valid: torch.Tensor,
    half_window: int,
) -> Optional[Tuple[int, int]]:
    """Return a span centred on the highest-scoring valid token."""
    if not valid.any():
        return None
    masked = probs.clone()
    masked[~valid] = -1.0
    peak = int(masked.argmax().item())
    start = max(0, peak - half_window)
    end = min(probs.size(0) - 1, peak + half_window)
    # Trim to valid boundary
    while start < end and not valid[start]:
        start += 1
    while end > start and not valid[end]:
        end -= 1
    return (start, end) if start <= end else None
