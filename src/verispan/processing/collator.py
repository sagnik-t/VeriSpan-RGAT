"""
collator.py — Batching collator for VeriSpan-RGAT (verispan.processing).

Pads variable-length tensors to batch-max length and stacks them.
Entity spans are passed through as Python lists (not tensors) because
they are variable-length per example and consumed by GraphBuilder
before any tensor operation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from .tokenization import SPAN_IGNORE_INDEX


class VerificationCollator:
    """
    Collate a list of encoded examples into a padded batch.

    Parameters
    ----------
    pad_token_id : int
        Token id used to pad input_ids.
    span_ignore_index : float
        Padding value for span_labels (must match BCE loss mask).
    """

    def __init__(
        self,
        pad_token_id: int,
        span_ignore_index: float = SPAN_IGNORE_INDEX,
    ) -> None:
        self.pad_token_id      = pad_token_id
        self.span_ignore_index = span_ignore_index

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns
        -------
        Batched dict:
            input_ids             LongTensor    [B, L]
            attention_mask        LongTensor    [B, L]
            span_labels           FloatTensor   [B, L]
            doc_token_mask        BoolTensor    [B, L]
            claim_token_mask      BoolTensor    [B, L]
            verdict_labels        LongTensor    [B]
            example_ids           List[str]
            claim_entity_spans    List[List[Tuple[int,int]]]  — per example
            doc_entity_spans      List[List[Tuple[int,int]]]  — per example
        """
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids_list    = []
        attn_mask_list    = []
        span_labels_list  = []
        doc_mask_list     = []
        claim_mask_list   = []
        verdict_labels    = []
        example_ids       = []
        claim_ent_spans   = []
        doc_ent_spans     = []

        for item in batch:
            pad_len = max_len - item["input_ids"].size(0)

            input_ids_list.append(
                F.pad(item["input_ids"], (0, pad_len), value=self.pad_token_id)
            )
            attn_mask_list.append(
                F.pad(item["attention_mask"], (0, pad_len), value=0)
            )
            span_labels_list.append(
                F.pad(item["span_labels"], (0, pad_len), value=self.span_ignore_index)
            )
            doc_mask_list.append(
                F.pad(item["doc_token_mask"].long(), (0, pad_len), value=0).bool()
            )
            claim_mask_list.append(
                F.pad(item["claim_token_mask"].long(), (0, pad_len), value=0).bool()
            )
            verdict_labels.append(item["verdict_label"])
            example_ids.append(item["example_id"])

            # Entity spans: pass through as-is (variable length, used by GraphBuilder)
            claim_ent_spans.append(item.get("claim_entity_spans", []))
            doc_ent_spans.append(item.get("doc_entity_spans", []))

        return {
            "input_ids":           torch.stack(input_ids_list),
            "attention_mask":      torch.stack(attn_mask_list),
            "span_labels":         torch.stack(span_labels_list),
            "doc_token_mask":      torch.stack(doc_mask_list),
            "claim_token_mask":    torch.stack(claim_mask_list),
            "verdict_labels":      torch.stack(verdict_labels),
            "example_ids":         example_ids,
            "claim_entity_spans":  claim_ent_spans,
            "doc_entity_spans":    doc_ent_spans,
        }


def build_collator(model_name: str = "microsoft/deberta-v3-small") -> VerificationCollator:
    """Create a VerificationCollator by resolving the pad token id from the tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return VerificationCollator(pad_token_id=tok.pad_token_id or 0)
