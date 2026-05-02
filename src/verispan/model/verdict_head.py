"""
verdict_head.py — Span-conditioned verdict prediction (Stage 5).

The verdict head reads exclusively from post-GNN evidence span
representations — never from the encoder's [CLS] token.  This is the
central structural novelty: the model cannot assign a verdict without
implicitly selecting and reasoning over evidence spans.

Forward pass
------------
    z_ev     : [N_es_total, d]  — post-GNN evidence span features for ALL
                                   examples in the batch, concatenated.
    batch_ev : [N_es_total]     — maps each row of z_ev to its example index.

    1. Compute a scalar attention score per evidence span:
           β_k = softmax over spans within example b of  (w^T z_k)
    2. Attention-weighted pool per example:
           v_rep_b = Σ_k β_k · z_k         (shape: [d])
    3. Linear classifier:
           logits = W_v v_rep + b_v         (shape: [B, 3])
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.utils import scatter


class VerdictHead(nn.Module):
    """
    Span-conditioned verdict prediction.

    Parameters
    ----------
    hidden_dim : int
        Evidence span representation dimension (= RGAT hidden_channels).
    num_classes : int
        Number of verdict classes (3: SUPPORTS / REFUTES / NEI).
    dropout : float
        Applied to the pooled evidence representation before classification.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Learnable attention query vector w ∈ R^d
        self.attn_w = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, z_ev: Tensor, batch_ev: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z_ev     : Tensor [N_es_total, d]
            Post-GNN evidence span representations for the entire batch,
            concatenated along the node dimension (PyG Batch convention).
        batch_ev : Tensor [N_es_total] (long)
            Example index for each evidence span node.

        Returns
        -------
        logits : Tensor [B, num_classes]
            Raw (pre-softmax) verdict scores.
        """
        B = int(batch_ev.max().item()) + 1

        # ── Attention-weighted pooling ────────────────────────────────────
        # Compute raw scores [N_es_total, 1] → [N_es_total]
        raw_scores = self.attn_w(z_ev).squeeze(-1)   # [N_es_total]

        # Softmax normalised *within* each example's evidence spans
        attn_weights = pyg_softmax(raw_scores, batch_ev, num_nodes=B)  # [N_es_total]

        # Weighted sum per example: scatter add → [B, d]
        v_rep = scatter(
            attn_weights.unsqueeze(-1) * z_ev,  # [N_es_total, d]
            batch_ev,
            dim=0,
            dim_size=B,
            reduce="sum",
        )   # [B, d]

        # ── Classification ────────────────────────────────────────────────
        return self.classifier(self.dropout(v_rep))   # [B, num_classes]
