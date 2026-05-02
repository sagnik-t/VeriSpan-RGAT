"""
graph.py — Heterogeneous graph construction (Stage 3).

Converts token-level encoder outputs into one PyG HeteroData graph per example,
then collates them into a Batch for the RGAT.

Node types
----------
    claim_span    : candidate spans from the claim
    evidence_span : candidate spans from the document
    entity        : named entity mentions (optional; skipped if not provided)

Edge types
----------
    (X, lexical, X)             : Jaccard(token sets) > lex_tau, within-type
    (X, semantic, X)            : cosine(span reps) > sem_tau, within-type
    (claim_span, cross_align, evidence_span)  : all claim × evidence pairs
    (evidence_span, cross_align, claim_span)  : reverse
    (entity, coref, claim_span)   : entity mention contained in claim span
    (entity, coref, evidence_span): entity mention contained in evidence span

GRAPH_METADATA
--------------
Exported constant used by HeteroRGAT to pre-allocate weight matrices for
every possible edge type.  Edge types that are absent from an individual
example's graph are silently skipped by HeteroConv.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData

from .span_head import extract_candidate_spans

# ── Node / edge type constants ────────────────────────────────────────────────

NT_CLAIM   = "claim_span"
NT_EVID    = "evidence_span"
NT_ENTITY  = "entity"

ET_LEX     = "lexical"
ET_SEM     = "semantic"
ET_CROSS   = "cross_align"
ET_COREF   = "coref"

# Full metadata: (node_types, edge_types)
# HeteroRGAT uses this to register one RGATConv per edge type.
GRAPH_METADATA: Tuple[List[str], List[Tuple[str, str, str]]] = (
    [NT_CLAIM, NT_EVID, NT_ENTITY],
    [
        (NT_CLAIM,  ET_LEX,   NT_CLAIM),
        (NT_CLAIM,  ET_SEM,   NT_CLAIM),
        (NT_EVID,   ET_LEX,   NT_EVID),
        (NT_EVID,   ET_SEM,   NT_EVID),
        (NT_CLAIM,  ET_CROSS, NT_EVID),
        (NT_EVID,   ET_CROSS, NT_CLAIM),
        (NT_ENTITY, ET_COREF, NT_CLAIM),
        (NT_ENTITY, ET_COREF, NT_EVID),
    ],
)


# ── GraphBuilder ──────────────────────────────────────────────────────────────

class GraphBuilder:
    """
    Stateless builder: call build() per example, build_batch() for a full
    mini-batch.

    Parameters
    ----------
    span_threshold : float
        Minimum span probability for a token to be included in a span.
    lex_tau : float
        Minimum Jaccard similarity for a lexical edge.
    sem_tau : float
        Minimum cosine similarity for a semantic edge.
    min_span_len, max_span_len : int
        Span length bounds passed to extract_candidate_spans.
    min_spans : int
        Minimum spans per node type (fallback extraction).
    """

    def __init__(
        self,
        span_threshold: float = 0.5,
        lex_tau: float = 0.3,
        sem_tau: float = 0.5,
        min_span_len: int = 1,
        max_span_len: int = 30,
        min_spans: int = 1,
    ) -> None:
        self.span_threshold = span_threshold
        self.lex_tau = lex_tau
        self.sem_tau = sem_tau
        self.min_span_len = min_span_len
        self.max_span_len = max_span_len
        self.min_spans = min_spans

    # ── public ───────────────────────────────────────────────────────────────

    def build(
        self,
        H: torch.Tensor,                                  # [seq_len, d]
        input_ids: torch.Tensor,                          # [seq_len]
        span_probs: torch.Tensor,                         # [seq_len]
        claim_token_mask: torch.Tensor,                   # [seq_len] bool
        doc_token_mask: torch.Tensor,                     # [seq_len] bool
        entity_token_spans: Optional[List[Tuple[int, int]]] = None,
    ) -> HeteroData:
        """Build one HeteroData graph for a single (claim, document) example."""

        entity_token_spans = entity_token_spans or []

        # 1. Candidate spans → graph nodes
        claim_spans = extract_candidate_spans(
            span_probs, claim_token_mask,
            threshold=self.span_threshold,
            min_span_len=self.min_span_len,
            max_span_len=self.max_span_len,
            min_spans=self.min_spans,
        )
        evid_spans = extract_candidate_spans(
            span_probs, doc_token_mask,
            threshold=self.span_threshold,
            min_span_len=self.min_span_len,
            max_span_len=self.max_span_len,
            min_spans=self.min_spans,
        )

        d = H.size(-1)
        dev = H.device

        # 2. Node features: mean-pool H over each span's token range
        claim_feats  = _pool_spans(H, claim_spans)         # [N_cs, d]
        evid_feats   = _pool_spans(H, evid_spans)          # [N_es, d]
        entity_feats = _pool_spans(H, entity_token_spans)  # [N_em, d]

        # 3. Token id sets (for lexical overlap)
        claim_tok_sets = [set(input_ids[s:e+1].tolist()) for s, e in claim_spans]
        evid_tok_sets  = [set(input_ids[s:e+1].tolist()) for s, e in evid_spans]

        # 4. Assemble HeteroData
        data = HeteroData()

        data[NT_CLAIM].x              = claim_feats
        data[NT_EVID].x               = evid_feats
        data[NT_ENTITY].x             = entity_feats  # empty [0, d] if none

        # Span positions as LongTensors — stored for evaluation / loss alignment.
        # Shape: [N_spans, 2]  (each row is [start, end] inclusive token indices)
        def _spans_to_tensor(spans, dev):
            if not spans:
                return torch.empty(0, 2, dtype=torch.long, device=dev)
            return torch.tensor(spans, dtype=torch.long, device=dev)

        data[NT_CLAIM].span_positions = _spans_to_tensor(claim_spans, dev)
        data[NT_EVID].span_positions  = _spans_to_tensor(evid_spans,  dev)

        # 5. Intra-type edges
        self._add_intra_edges(data, NT_CLAIM,  claim_feats,  claim_tok_sets,  dev)
        self._add_intra_edges(data, NT_EVID,   evid_feats,   evid_tok_sets,   dev)

        # 6. Cross-alignment edges
        self._add_cross_edges(data, len(claim_spans), len(evid_spans), dev)

        # 7. Entity co-reference edges (skipped if no entities)
        if entity_feats.size(0) > 0:
            self._add_coref_edges(data, claim_spans, evid_spans, entity_token_spans, dev)

        return data

    def build_batch(
        self,
        H: torch.Tensor,                                           # [B, L, d]
        input_ids: torch.Tensor,                                   # [B, L]
        span_probs: torch.Tensor,                                  # [B, L]
        claim_token_mask: torch.Tensor,                            # [B, L]
        doc_token_mask: torch.Tensor,                              # [B, L]
        entity_token_spans: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> Batch:
        """Build and collate one graph per example in the batch."""
        graphs = [
            self.build(
                H=H[b],
                input_ids=input_ids[b],
                span_probs=span_probs[b],
                claim_token_mask=claim_token_mask[b],
                doc_token_mask=doc_token_mask[b],
                entity_token_spans=entity_token_spans[b] if entity_token_spans else None,
            )
            for b in range(H.size(0))
        ]
        return Batch.from_data_list(graphs)

    # ── internals ────────────────────────────────────────────────────────────

    def _add_intra_edges(
        self,
        data: HeteroData,
        ntype: str,
        feats: torch.Tensor,    # [N, d]
        tok_sets: List[set],
        dev: torch.device,
    ) -> None:
        """Lexical + semantic edges between nodes of the same type."""
        N = feats.size(0)
        if N < 2:
            return

        lex_src, lex_dst, lex_w = [], [], []
        sem_src, sem_dst, sem_w = [], [], []

        for i in range(N):
            for j in range(i + 1, N):
                # Lexical overlap (Jaccard on token id sets)
                jac = _jaccard(tok_sets[i], tok_sets[j])
                if jac > self.lex_tau:
                    lex_src += [i, j]; lex_dst += [j, i]; lex_w += [jac, jac]

                # Semantic similarity (cosine on mean-pooled reps)
                cos = float(_cosine(feats[i], feats[j]).detach())
                if cos > self.sem_tau:
                    sem_src += [i, j]; sem_dst += [j, i]; sem_w += [cos, cos]

        if lex_src:
            key = (ntype, ET_LEX, ntype)
            data[key].edge_index  = torch.tensor([lex_src, lex_dst], dtype=torch.long, device=dev)
            data[key].edge_weight = torch.tensor(lex_w, dtype=torch.float, device=dev)

        if sem_src:
            key = (ntype, ET_SEM, ntype)
            data[key].edge_index  = torch.tensor([sem_src, sem_dst], dtype=torch.long, device=dev)
            data[key].edge_weight = torch.tensor(sem_w, dtype=torch.float, device=dev)

    def _add_cross_edges(
        self,
        data: HeteroData,
        n_claim: int,
        n_evid: int,
        dev: torch.device,
    ) -> None:
        """All-pairs bidirectional cross-alignment edges (claim ↔ evidence)."""
        if n_claim == 0 or n_evid == 0:
            return

        cs = [i for i in range(n_claim) for _ in range(n_evid)]
        es = [j for _ in range(n_claim) for j in range(n_evid)]

        ei_fwd = torch.tensor([cs, es], dtype=torch.long, device=dev)
        ei_rev = torch.tensor([es, cs], dtype=torch.long, device=dev)

        data[NT_CLAIM, ET_CROSS, NT_EVID].edge_index = ei_fwd
        data[NT_EVID,  ET_CROSS, NT_CLAIM].edge_index = ei_rev

    def _add_coref_edges(
        self,
        data: HeteroData,
        claim_spans: List[Tuple[int, int]],
        evid_spans: List[Tuple[int, int]],
        entity_spans: List[Tuple[int, int]],
        dev: torch.device,
    ) -> None:
        """
        Connect entity mention nodes to span nodes that contain them.
        Containment = token-range overlap.
        """
        def _build_coref(target_spans, target_type):
            src, dst = [], []
            for eid, (es, ee) in enumerate(entity_spans):
                for sid, (ss, se) in enumerate(target_spans):
                    if es <= se and ee >= ss:   # ranges overlap
                        src.append(eid); dst.append(sid)
            if src:
                key = (NT_ENTITY, ET_COREF, target_type)
                data[key].edge_index = torch.tensor([src, dst], dtype=torch.long, device=dev)

        _build_coref(claim_spans, NT_CLAIM)
        _build_coref(evid_spans,  NT_EVID)


# ── Utility functions ─────────────────────────────────────────────────────────

def _pool_spans(H: torch.Tensor, spans: List[Tuple[int, int]]) -> torch.Tensor:
    """Mean-pool H over each span's token range. Returns [N_spans, d]."""
    if not spans:
        return torch.empty(0, H.size(-1), device=H.device, dtype=H.dtype)
    return torch.stack([H[s:e+1].mean(0) for s, e in spans])


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.normalize(u, dim=0) @ F.normalize(v, dim=0)
