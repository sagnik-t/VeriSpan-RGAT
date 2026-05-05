"""
tests/processing/test_tokenization.py

Unit tests for VerificationTokenizer.
Fast — no HuggingFace dataset downloads, no GPU.
The tokenizer itself downloads the DeBERTa vocab on first run (cached).
"""

import pytest
import torch

from verispan.processing.tokenization import SPAN_IGNORE_INDEX
from factories import make_example


class TestVerificationTokenizer:

    def test_output_keys(self, tokenizer, examples):
        enc = tokenizer.encode(examples[0])
        expected = {
            "input_ids", "attention_mask", "span_labels",
            "claim_token_mask", "doc_token_mask",
            "verdict_label", "example_id",
        }
        assert expected == set(enc.keys())

    def test_all_tensors_same_length(self, tokenizer, examples):
        enc = tokenizer.encode(examples[0])
        L   = enc["input_ids"].size(0)
        for key in ["attention_mask", "span_labels",
                    "claim_token_mask", "doc_token_mask"]:
            assert enc[key].size(0) == L, f"{key} has wrong length"

    def test_span_label_values_are_valid(self, tokenizer, examples):
        enc    = tokenizer.encode(examples[0])
        unique = set(enc["span_labels"].tolist())
        assert unique.issubset({-1.0, 0.0, 1.0}), \
            f"Unexpected span_label values: {unique}"

    def test_evidence_tokens_are_labelled_one(self, tokenizer, examples):
        """SUPPORTS example with char_spans must have at least one 1.0 label."""
        enc = tokenizer.encode(examples[0])   # verdict=SUPPORTS, has spans
        assert (enc["span_labels"] == 1.0).any(), \
            "No evidence tokens found — span alignment is broken"

    def test_claim_tokens_are_ignored(self, tokenizer, examples):
        enc         = tokenizer.encode(examples[0])
        claim_mask  = enc["claim_token_mask"]
        span_labels = enc["span_labels"]
        assert (span_labels[claim_mask] == SPAN_IGNORE_INDEX).all(), \
            "Claim tokens must always carry SPAN_IGNORE_INDEX"

    def test_masks_mutually_exclusive(self, tokenizer, examples):
        enc     = tokenizer.encode(examples[0])
        overlap = enc["claim_token_mask"] & enc["doc_token_mask"]
        assert not overlap.any(), \
            "claim_token_mask and doc_token_mask must not overlap"

    def test_at_least_one_doc_token(self, tokenizer, examples):
        enc = tokenizer.encode(examples[0])
        assert enc["doc_token_mask"].any(), "Expected at least one document token"

    def test_at_least_one_claim_token(self, tokenizer, examples):
        enc = tokenizer.encode(examples[0])
        assert enc["claim_token_mask"].any(), "Expected at least one claim token"

    @pytest.mark.parametrize("verdict_id", [0, 1, 2])
    def test_verdict_label_correct(self, tokenizer, verdict_id):
        ex  = make_example(verdict=verdict_id)
        enc = tokenizer.encode(ex)
        assert enc["verdict_label"].item() == verdict_id

    def test_nei_empty_document_does_not_crash(self, tokenizer):
        ex  = make_example(verdict=2, document="", char_spans=[])
        enc = tokenizer.encode(ex)
        assert enc["input_ids"].size(0) > 0

    def test_nei_no_evidence_tokens(self, tokenizer):
        ex  = make_example(verdict=2, document="Some text here.", char_spans=[])
        enc = tokenizer.encode(ex)
        assert not (enc["span_labels"] == 1.0).any(), \
            "NEI example with no char_spans should have no evidence tokens"

    def test_truncation_respects_max_length(self, tokenizer):
        long_doc = "The quick brown fox jumps over the lazy dog. " * 50
        ex       = make_example(document=long_doc, char_spans=[(0, 10)])
        enc      = tokenizer.encode(ex)
        assert enc["input_ids"].size(0) <= tokenizer.max_length

    def test_claim_is_never_truncated(self, tokenizer):
        """
        With truncation='only_second', the claim should always be fully
        present even when the document is very long.
        """
        long_doc  = "evidence sentence. " * 100
        short_claim = "Short claim."
        ex  = make_example(claim=short_claim, document=long_doc, char_spans=[(0, 8)])
        enc = tokenizer.encode(ex)

        # Decode and verify claim tokens are intact
        decoded = tokenizer.tokenizer.decode(
            enc["input_ids"][enc["claim_token_mask"]].tolist(),
            skip_special_tokens=True,
        )
        assert short_claim.lower().rstrip(".") in decoded.lower()

    def test_example_id_preserved(self, tokenizer):
        ex  = make_example(example_id="unique-id-99")
        enc = tokenizer.encode(ex)
        assert enc["example_id"] == "unique-id-99"

    def test_span_overlap_detection(self, tokenizer):
        """
        A char span that partially overlaps a token boundary should still
        label that token as evidence (inclusive overlap policy).
        """
        # "Paris" starts at char 36 in "The Eiffel Tower is located in Paris, France."
        # We provide a span that only covers the first two characters of "Paris"
        ex  = make_example(char_spans=[(36, 38)])
        enc = tokenizer.encode(ex)
        assert (enc["span_labels"] == 1.0).any(), \
            "Partial token overlap should still mark the token as evidence"
