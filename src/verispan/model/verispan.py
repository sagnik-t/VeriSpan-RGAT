"""
verispan.py — Top-level VeriSpan-RGAT model (all 5 stages + joint loss).

Architecture summary
--------------------
    Stage 1  DeBERTaEncoder       → H ∈ R^(B × L × 768)
    Stage 2  SpanExtractionHead   → span_logits, span_probs
    Stage 3  GraphBuilder         → PyG Batch (HeteroData)
    Stage 4  HeteroRGAT           → updated node reps in-place
    Stage 5  VerdictHead          → verdict_logits ∈ R^(B × 3)

Loss
----
    L = L_span  +  λ1 · L_verdict  +  λ2 · L_contrast
    where:
        L_span    = BCE over document tokens (span_labels ≠ −1)
        L_verdict = CE over verdict labels
        L_contrast= max(0, δ − ‖z̄_sup − z̄_ref‖₂)
                    (only when both SUPPORTS and REFUTES examples are present)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch

from .encoder import DeBERTaEncoder
from .graph import GRAPH_METADATA, GraphBuilder
from .rgat import HeteroRGAT
from .span_head import SpanExtractionHead
from .verdict_head import VerdictHead


# ── Output dataclasses ────────────────────────────────────────────────────────

@dataclass
class VeriSpanOutput:
    """Raw outputs from a forward pass."""
    span_logits:     Tensor           # [B, L]       pre-sigmoid span scores
    span_probs:      Tensor           # [B, L]       sigmoid span probabilities
    verdict_logits:  Tensor           # [B, 3]       pre-softmax verdict scores
    z_evidence:      Tensor           # [N_es, d]    post-GNN evidence span reps
    evidence_batch:  Tensor           # [N_es]       example index per span
    graph_batch:     Optional[Batch]  # full PyG Batch (for inspection / eval)


@dataclass
class LossOutput:
    """Scalar losses returned by compute_loss()."""
    total:    Tensor
    span:     Tensor
    verdict:  Tensor
    contrast: Tensor

    def as_dict(self) -> Dict[str, float]:
        return {
            "loss":          self.total.item(),
            "loss_span":     self.span.item(),
            "loss_verdict":  self.verdict.item(),
            "loss_contrast": self.contrast.item(),
        }


# ── VeriSpanConfig ────────────────────────────────────────────────────────────

@dataclass
class VeriSpanConfig:
    # Encoder
    encoder_name:     str   = "microsoft/deberta-v3-small"
    freeze_layers:    int   = 0

    # Span head
    span_dropout:     float = 0.1
    span_threshold:   float = 0.5
    min_span_len:     int   = 1
    max_span_len:     int   = 30
    min_spans:        int   = 1

    # Graph construction
    lex_tau:          float = 0.3
    sem_tau:          float = 0.5

    # RGAT
    rgat_layers:      int   = 2
    rgat_heads:       int   = 4
    hidden_channels:  int   = 768

    # Verdict head
    verdict_dropout:  float = 0.1
    num_classes:      int   = 3

    # Loss weights
    lambda1:          float = 1.0   # verdict loss weight
    lambda2:          float = 0.5   # contrastive loss weight
    contrast_margin:  float = 1.0   # δ in the margin loss


# ── VeriSpanModel ─────────────────────────────────────────────────────────────

class VeriSpanModel(nn.Module):
    """
    VeriSpan-RGAT: Span-Level Heterogeneous Graph Reasoning for Claim Verification.

    Parameters
    ----------
    config : VeriSpanConfig
        All architectural and training hyperparameters.
    """

    def __init__(self, config: VeriSpanConfig = VeriSpanConfig()) -> None:
        super().__init__()
        self.config = config

        # Stage 1
        self.encoder = DeBERTaEncoder(
            model_name=config.encoder_name,
            freeze_layers=config.freeze_layers,
        )

        # Stage 2
        self.span_head = SpanExtractionHead(
            hidden_dim=self.encoder.hidden_size,
            dropout=config.span_dropout,
        )

        # Stage 3 (stateless builder — no learnable params)
        self.graph_builder = GraphBuilder(
            span_threshold=config.span_threshold,
            lex_tau=config.lex_tau,
            sem_tau=config.sem_tau,
            min_span_len=config.min_span_len,
            max_span_len=config.max_span_len,
            min_spans=config.min_spans,
        )

        # Stage 4
        self.rgat = HeteroRGAT(
            in_channels=self.encoder.hidden_size,
            hidden_channels=config.hidden_channels,
            num_layers=config.rgat_layers,
            heads=config.rgat_heads,
            metadata=GRAPH_METADATA,
        )

        # Stage 5
        self.verdict_head = VerdictHead(
            hidden_dim=config.hidden_channels,
            num_classes=config.num_classes,
            dropout=config.verdict_dropout,
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:        Tensor,                              # [B, L]
        attention_mask:   Tensor,                              # [B, L]
        doc_token_mask:   Tensor,                              # [B, L]
        claim_token_mask: Tensor,                              # [B, L]
        entity_token_spans: Optional[List[List[Tuple[int, int]]]] = None,
        **kwargs,  # absorbs unused batch keys (e.g. span_labels, verdict_labels)
    ) -> VeriSpanOutput:
        """
        Full forward pass through all 5 stages.

        Parameters
        ----------
        input_ids, attention_mask : [B, L]
            Tokenizer outputs.
        doc_token_mask, claim_token_mask : [B, L] bool
            Segment masks from the collator.
        entity_token_spans : List[List[(start, end)]], optional
            Token-level entity mention spans per example.  Pass None to omit
            entity mention nodes from the graph.

        Returns
        -------
        VeriSpanOutput
        """

        # ── Stage 1: encode ──────────────────────────────────────────────
        H = self.encoder(input_ids, attention_mask)    # [B, L, d]

        # ── Stage 2: span extraction ─────────────────────────────────────
        span_logits, span_probs = self.span_head(H, doc_token_mask)

        # ── Stage 3: graph construction ──────────────────────────────────
        # detach span_probs from graph: topology is non-differentiable (threshold)
        # but node features (mean-pool of H) remain in the compute graph.
        graph_batch: Batch = self.graph_builder.build_batch(
            H=H,
            input_ids=input_ids,
            span_probs=span_probs.detach(),
            claim_token_mask=claim_token_mask,
            doc_token_mask=doc_token_mask,
            entity_token_spans=entity_token_spans,
        )

        # Move graph to the same device as the encoder output
        graph_batch = graph_batch.to(H.device)

        # ── Stage 4: RGAT message passing ────────────────────────────────
        import warnings
        with warnings.catch_warnings():
            # Entity nodes are intentional send-only nodes (coref edges push
            # signal to span nodes; entities never receive messages).
            warnings.filterwarnings("ignore", message=".*entity.*")
            graph_batch = self.rgat(graph_batch)

        # ── Stage 5: span-conditioned verdict prediction ──────────────────
        # Explicitly move to H.device — PyG creates batch index tensors on
        # CPU inside Batch.from_data_list(), and .to() does not always catch
        # all of them before we reach the verdict head.
        dev = H.device
        z_ev     = graph_batch["evidence_span"].x.to(dev)      # [N_es_total, d]
        batch_ev = graph_batch["evidence_span"].batch.to(dev)  # [N_es_total]

        verdict_logits = self.verdict_head(z_ev, batch_ev)  # [B, 3]

        return VeriSpanOutput(
            span_logits=span_logits,
            span_probs=span_probs,
            verdict_logits=verdict_logits,
            z_evidence=z_ev,
            evidence_batch=batch_ev,
            graph_batch=graph_batch,
        )

    # ── Loss ─────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        output: VeriSpanOutput,
        span_labels:    Tensor,    # [B, L]  1.0 / 0.0 / -1.0 (ignore)
        verdict_labels: Tensor,    # [B]     int 0/1/2
    ) -> LossOutput:
        """
        Compute the joint training objective:
            L = L_span  +  λ1 · L_verdict  +  λ2 · L_contrast
        """

        # ── Span extraction loss (BCE, masked) ────────────────────────────
        valid = span_labels != -1.0     # [B, L] — doc tokens only
        L_span = F.binary_cross_entropy_with_logits(
            output.span_logits[valid],
            span_labels[valid],
        )

        # ── Verdict classification loss (cross-entropy) ───────────────────
        L_verdict = F.cross_entropy(output.verdict_logits, verdict_labels)

        # ── Contrastive span loss ─────────────────────────────────────────
        L_contrast = self._contrastive_loss(
            output.z_evidence,
            output.evidence_batch,
            verdict_labels,
        )

        total = (
            L_span
            + self.config.lambda1 * L_verdict
            + self.config.lambda2 * L_contrast
        )

        return LossOutput(total=total, span=L_span, verdict=L_verdict, contrast=L_contrast)

    def _contrastive_loss(
        self,
        z_ev:           Tensor,    # [N_es_total, d]
        batch_ev:       Tensor,    # [N_es_total]
        verdict_labels: Tensor,    # [B]
    ) -> Tensor:
        """
        Push mean evidence representations of SUPPORTS and REFUTES examples
        apart by at least `contrast_margin` in L2 distance.

            L_contrast = max(0, δ − ‖z̄_sup − z̄_ref‖₂)

        Returns 0 if the batch does not contain both verdict types.
        """
        B = verdict_labels.size(0)
        device = z_ev.device

        # Mean-pool evidence spans per example
        z_per = []
        for b in range(B):
            mask = batch_ev == b
            z_per.append(z_ev[mask].mean(0) if mask.any() else torch.zeros(z_ev.size(-1), device=device))
        z_per = torch.stack(z_per)   # [B, d]

        sup_mask = verdict_labels == 0   # SUPPORTS
        ref_mask = verdict_labels == 1   # REFUTES

        if not sup_mask.any() or not ref_mask.any():
            # Contrastive loss undefined — both classes needed
            return torch.tensor(0.0, device=device, requires_grad=False)

        z_sup = z_per[sup_mask].mean(0)   # [d]
        z_ref = z_per[ref_mask].mean(0)   # [d]

        dist = (z_sup - z_ref).norm(p=2)
        return F.relu(self.config.contrast_margin - dist)

    # ── Convenience ──────────────────────────────────────────────────────────

    def forward_and_loss(
        self,
        batch: Dict[str, Tensor],
    ) -> Tuple[VeriSpanOutput, LossOutput]:
        """
        One-liner for the training loop: forward pass + loss computation.

        Parameters
        ----------
        batch : dict from VerificationCollator
            Must contain: input_ids, attention_mask, doc_token_mask,
            claim_token_mask, span_labels, verdict_labels.
        """
        output = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            doc_token_mask=batch["doc_token_mask"],
            claim_token_mask=batch["claim_token_mask"],
        )
        loss = self.compute_loss(
            output=output,
            span_labels=batch["span_labels"],
            verdict_labels=batch["verdict_labels"],
        )
        return output, loss

    @classmethod
    def from_config(cls, config: VeriSpanConfig) -> "VeriSpanModel":
        return cls(config)
