"""
collator.py — Batching collator for VeriSpan-RGAT.

VerificationCollator pads sequences to the longest in each batch (dynamic
padding) and stacks all tensors.  It is designed to be passed directly to
torch.utils.data.DataLoader as the `collate_fn`.

Padding strategy:
    input_ids       → pad with tokenizer.pad_token_id
    attention_mask  → pad with 0
    span_labels     → pad with SPAN_IGNORE_INDEX (-1.0)  ← ignored by BCE loss
    doc_token_mask  → pad with False
    claim_token_mask→ pad with False

The variable-length graph construction (Stage 3) is handled during the
forward pass from these padded tensors; the collator only deals with the
token-level encoder outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .tokenization import SPAN_IGNORE_INDEX


class VerificationCollator:
    """
    Collate a list of encoded examples into a padded batch.

    Parameters
    ----------
    pad_token_id : int
        Token id used to pad input_ids.  Typically tokenizer.pad_token_id.
    span_ignore_index : float
        Value used to pad span_labels (default: -1.0).  Must match the
        value used by the BCE loss to identify positions to ignore.
    """

    def __init__(
        self,
        pad_token_id: int,
        span_ignore_index: float = SPAN_IGNORE_INDEX,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.span_ignore_index = span_ignore_index

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Pad all variable-length tensors to the batch's maximum sequence
        length and stack them.

        Parameters
        ----------
        batch : list of dicts, each from ClaimVerificationDataset.__getitem__

        Returns
        -------
        Batched dict with the following keys:
            input_ids         LongTensor   [B, L]
            attention_mask    LongTensor   [B, L]
            span_labels       FloatTensor  [B, L]
            doc_token_mask    BoolTensor   [B, L]
            claim_token_mask  BoolTensor   [B, L]
            verdict_labels    LongTensor   [B]
            example_ids       List[str]    (not a tensor)
        """
        max_len: int = max(item["input_ids"].size(0) for item in batch)

        input_ids_list: List[torch.Tensor] = []
        attn_mask_list: List[torch.Tensor] = []
        span_labels_list: List[torch.Tensor] = []
        doc_mask_list: List[torch.Tensor] = []
        claim_mask_list: List[torch.Tensor] = []
        verdict_labels: List[torch.Tensor] = []
        example_ids: List[str] = []

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
            # BoolTensors → cast to int for F.pad, then back to bool
            doc_mask_list.append(
                F.pad(item["doc_token_mask"].long(), (0, pad_len), value=0).bool()
            )
            claim_mask_list.append(
                F.pad(item["claim_token_mask"].long(), (0, pad_len), value=0).bool()
            )
            verdict_labels.append(item["verdict_label"])
            example_ids.append(item["example_id"])

        return {
            "input_ids":         torch.stack(input_ids_list),         # [B, L]
            "attention_mask":    torch.stack(attn_mask_list),         # [B, L]
            "span_labels":       torch.stack(span_labels_list),       # [B, L]
            "doc_token_mask":    torch.stack(doc_mask_list),          # [B, L]
            "claim_token_mask":  torch.stack(claim_mask_list),        # [B, L]
            "verdict_labels":    torch.stack(verdict_labels),         # [B]
            "example_ids":       example_ids,                         # List[str]
        }


# ── Convenience factory ───────────────────────────────────────────────────────

def build_collator(model_name: str = "microsoft/deberta-v3-small") -> VerificationCollator:
    """
    Create a VerificationCollator by loading the tokenizer pad_token_id.
    Avoids having to pass the tokenizer object separately.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return VerificationCollator(pad_token_id=tok.pad_token_id or 0)
