"""
tokenization.py — DeBERTa tokenizer wrapper (verispan.processing).

Encodes a VerificationExample into the tensor dict consumed by
ClaimVerificationDataset.  The key operation is aligning character-level
evidence spans to subword token positions via the tokenizer's offset mapping.

Output tensors (un-padded, per example):
    input_ids         LongTensor   [seq_len]
    attention_mask    LongTensor   [seq_len]
    span_labels       FloatTensor  [seq_len]
                        1.0  = evidence token
                        0.0  = non-evidence doc token
                       -1.0  = claim / special token (ignored in BCE loss)
    doc_token_mask    BoolTensor   [seq_len]
    claim_token_mask  BoolTensor   [seq_len]
    verdict_label     LongTensor   scalar
    example_id        str
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ..data.schema import VerificationExample

logger = logging.getLogger(__name__)

SPAN_IGNORE_INDEX: float = -1.0


class VerificationTokenizer:
    """
    Wraps a HuggingFace fast tokenizer and produces the tensor dict
    required by ClaimVerificationDataset.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Must be a fast tokenizer.
    max_length : int
        Hard token budget.  Sequences longer than this are truncated on
        the document side only (claim is never truncated).
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        max_length: int = 512,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            model_name
        )
        if not self.tokenizer.is_fast:
            raise ValueError(
                f"VerificationTokenizer requires a HuggingFace *fast* tokenizer. "
                f"{model_name!r} returned a slow tokenizer."
            )
        self.max_length    = max_length
        self.pad_token_id  = self.tokenizer.pad_token_id or 0

    # ── public ───────────────────────────────────────────────────────────────

    def encode(self, example: VerificationExample) -> Dict[str, Any]:
        """
        Tokenize one VerificationExample and return the tensor dict.

        Sequence layout:
            [CLS] <claim tokens> [SEP] <document tokens> [SEP]
        """
        document = example.document.strip() if example.document.strip() else "[EMPTY]"

        encoding = self.tokenizer(
            example.claim,
            document,
            max_length=self.max_length,
            truncation="only_second",   # claim is never truncated
            padding=False,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors=None,
        )

        input_ids:      List[int]              = encoding["input_ids"]
        attention_mask: List[int]              = encoding["attention_mask"]
        offset_mapping: List[Tuple[int, int]]  = encoding["offset_mapping"]
        seq_ids:        List[Optional[int]]    = encoding.sequence_ids()
        seq_len = len(input_ids)

        claim_token_mask = [sid == 0 for sid in seq_ids]
        doc_token_mask   = [sid == 1 for sid in seq_ids]

        span_labels: List[float] = [SPAN_IGNORE_INDEX] * seq_len
        for i, is_doc in enumerate(doc_token_mask):
            if not is_doc:
                continue
            tok_start, tok_end = offset_mapping[i]
            span_labels[i] = self._token_in_evidence(
                tok_start, tok_end, example.evidence_char_spans
            )

        return {
            "input_ids":         torch.tensor(input_ids, dtype=torch.long),
            "attention_mask":    torch.tensor(attention_mask, dtype=torch.long),
            "span_labels":       torch.tensor(span_labels, dtype=torch.float),
            "claim_token_mask":  torch.tensor(claim_token_mask, dtype=torch.bool),
            "doc_token_mask":    torch.tensor(doc_token_mask, dtype=torch.bool),
            "verdict_label":     torch.tensor(example.verdict, dtype=torch.long),
            "example_id":        example.example_id,
        }

    def batch_encode(
        self, examples: List[VerificationExample]
    ) -> List[Dict[str, Any]]:
        return [self.encode(ex) for ex in examples]

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _token_in_evidence(
        tok_start: int,
        tok_end: int,
        evidence_spans: List[Tuple[int, int]],
    ) -> float:
        """Return 1.0 if the token overlaps any evidence span, else 0.0."""
        for ev_start, ev_end in evidence_spans:
            if tok_start < ev_end and tok_end > ev_start:
                return 1.0
        return 0.0
