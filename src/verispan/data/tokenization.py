"""
tokenization.py — DeBERTa tokenizer wrapper for VeriSpan-RGAT.

Encodes a VerificationExample into the tensor dict consumed by
ClaimVerificationDataset.  The critical operation is aligning character-level
evidence spans (from VerificationExample.evidence_char_spans) to subword
token positions using the tokenizer's offset mapping.

Output tensors (all of length seq_len, un-padded):
    input_ids         LongTensor   — token ids
    attention_mask    LongTensor   — 1 for real tokens, 0 for padding
    span_labels       FloatTensor  — 1.0 for evidence tokens,
                                     0.0 for non-evidence doc tokens,
                                    -1.0 for claim / special tokens (ignored
                                          in BCE loss)
    doc_token_mask    BoolTensor   — True for document-side tokens
    claim_token_mask  BoolTensor   — True for claim-side tokens
    verdict_label     LongTensor   — scalar, 0/1/2
    example_id        str
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .schema import VerificationExample

logger = logging.getLogger(__name__)

# Sentinel used for claim / special tokens in span_labels.
# BCE loss must mask positions where span_labels == IGNORE_INDEX.
SPAN_IGNORE_INDEX: float = -1.0


class VerificationTokenizer:
    """
    Wraps a HuggingFace fast tokenizer (DeBERTa-v3-small by default) and
    produces the tensor dict required by ClaimVerificationDataset.

    Parameters
    ----------
    model_name : str
        HuggingFace model id.  Must be a *fast* tokenizer (returns
        offset_mapping and sequence_ids).
    max_length : int
        Hard token budget.  Sequences longer than this are truncated on the
        document side only (claim is never truncated).
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
        self.max_length = max_length
        self.pad_token_id: int = self.tokenizer.pad_token_id or 0

    # ── public API ───────────────────────────────────────────────────────────

    def encode(self, example: VerificationExample) -> Dict[str, Any]:
        """
        Tokenize a VerificationExample and return the tensor dict.

        Sequence layout:
            [CLS] <claim tokens> [SEP] <document tokens> [SEP]

        For NEI examples with an empty document, the document side is
        replaced with the single sentinel token '[EMPTY]' so the input
        is never degenerate.
        """
        document = example.document.strip() if example.document.strip() else "[EMPTY]"

        encoding = self.tokenizer(
            example.claim,
            document,
            max_length=self.max_length,
            truncation="only_second",  # claim is never truncated
            padding=False,             # collator handles padding
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors=None,       # return Python lists for now
        )

        input_ids: List[int] = encoding["input_ids"]
        attention_mask: List[int] = encoding["attention_mask"]
        offset_mapping: List[Tuple[int, int]] = encoding["offset_mapping"]
        # sequence_ids() → None (special), 0 (claim), 1 (document)
        seq_ids: List[Optional[int]] = encoding.sequence_ids()
        seq_len = len(input_ids)

        # ── per-token classification masks ────────────────────────────────
        claim_token_mask = [sid == 0 for sid in seq_ids]
        doc_token_mask   = [sid == 1 for sid in seq_ids]

        # ── span label alignment ──────────────────────────────────────────
        # Default: IGNORE for claim tokens and special tokens.
        span_labels: List[float] = [SPAN_IGNORE_INDEX] * seq_len

        for i, is_doc in enumerate(doc_token_mask):
            if not is_doc:
                continue
            tok_start, tok_end = offset_mapping[i]
            # Degenerate zero-width offset → special token inside the doc
            # segment; treat as non-evidence but supervise anyway.
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
        self,
        examples: List[VerificationExample],
    ) -> List[Dict[str, Any]]:
        """Encode a list of examples.  Thin convenience wrapper."""
        return [self.encode(ex) for ex in examples]

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _token_in_evidence(
        tok_start: int,
        tok_end: int,
        evidence_spans: List[Tuple[int, int]],
    ) -> float:
        """
        Return 1.0 if the token [tok_start, tok_end) overlaps with any
        evidence span, else 0.0.

        Overlap condition (half-open intervals):
            tok_start < ev_end  AND  tok_end > ev_start

        This is deliberately inclusive: a token that partially overlaps an
        evidence boundary is counted as evidence.
        """
        for ev_start, ev_end in evidence_spans:
            if tok_start < ev_end and tok_end > ev_start:
                return 1.0
        return 0.0


# ── Standalone test ──────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """Quick span alignment check — run with `python -m verispan.data.tokenization`."""
    from .schema import VerificationExample

    ex = VerificationExample(
        example_id="test-0",
        claim="The Eiffel Tower is located in Berlin.",
        document="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        verdict=1,  # REFUTES
        evidence_char_spans=[(71, 84)],  # "Paris, France"
        evidence_sentence_texts=["The Eiffel Tower is ... Paris, France."],
        source="test",
    )

    tok = VerificationTokenizer("microsoft/deberta-v3-small")
    enc = tok.encode(ex)

    print("input_ids shape  :", enc["input_ids"].shape)
    print("span_labels shape:", enc["span_labels"].shape)

    # Verify at least one token is labelled 1.0
    evidence_token_count = (enc["span_labels"] == 1.0).sum().item()
    print(f"Evidence tokens  : {evidence_token_count}")
    assert evidence_token_count > 0, "No evidence tokens found — alignment bug!"

    # Show labelled tokens
    ids = enc["input_ids"].tolist()
    labels = enc["span_labels"].tolist()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    tokens = tokenizer.convert_ids_to_tokens(ids)
    for tok_str, lbl in zip(tokens, labels):
        if lbl == 1.0:
            print(f"  [EVIDENCE] {tok_str!r}")

    print("Smoke test passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()
