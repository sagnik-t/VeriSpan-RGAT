"""
tests/processing/test_collator.py

Unit tests for VerificationCollator.
Fast — uses synthetic encoded examples from conftest.
"""

import pytest
import torch

from verispan.processing.tokenization import SPAN_IGNORE_INDEX
from factories import make_example


class TestVerificationCollator:

    def test_output_keys(self, collator, encoded_examples):
        batch    = collator(encoded_examples)
        expected = {
            "input_ids", "attention_mask", "span_labels",
            "doc_token_mask", "claim_token_mask",
            "verdict_labels", "example_ids",
            "claim_entity_spans", "doc_entity_spans",
        }
        assert expected == set(batch.keys())

    def test_batch_size_first_dim(self, collator, encoded_examples):
        batch = collator(encoded_examples)
        B     = len(encoded_examples)
        assert batch["input_ids"].shape[0]      == B
        assert batch["verdict_labels"].shape[0] == B
        assert len(batch["example_ids"])        == B

    def test_all_tensors_same_seq_length(self, collator, encoded_examples):
        batch = collator(encoded_examples)
        B, L  = batch["input_ids"].shape
        for key in ["attention_mask", "span_labels",
                    "doc_token_mask", "claim_token_mask"]:
            assert batch[key].shape == (B, L), \
                f"{key} has unexpected shape {batch[key].shape}"

    def test_padding_fills_to_max_length(self, tokenizer, collator):
        """Batch length must equal the longest sequence in the batch."""
        short_enc = tokenizer.encode(make_example(
            claim="Short.", document="Short doc.", char_spans=[(0, 5)],
            example_id="short",
        ))
        long_enc  = tokenizer.encode(make_example(
            claim="A much longer claim about a complex topic.",
            document="A much longer document. " * 10,
            char_spans=[(0, 10)],
            example_id="long",
        ))
        long_len = long_enc["input_ids"].size(0)
        batch    = collator([short_enc, long_enc])
        assert batch["input_ids"].shape[1] == long_len

    def test_padded_input_ids_use_pad_token(self, tokenizer, collator):
        short_enc = tokenizer.encode(make_example(
            claim="Hi.", document="Doc.", char_spans=[(0, 3)], example_id="s"
        ))
        long_enc  = tokenizer.encode(make_example(
            claim="A longer claim text here.",
            document="A longer document text. " * 5,
            char_spans=[(0, 10)],
            example_id="l",
        ))
        short_len = short_enc["input_ids"].size(0)
        long_len  = long_enc["input_ids"].size(0)
        batch     = collator([short_enc, long_enc])

        if long_len > short_len:
            padded = batch["input_ids"][0, short_len:]
            assert (padded == tokenizer.pad_token_id).all(), \
                "Padded input_ids must use pad_token_id"

    def test_padded_attention_mask_is_zero(self, tokenizer, collator):
        short_enc = tokenizer.encode(make_example(
            claim="Hi.", document="Doc.", char_spans=[(0, 3)], example_id="s"
        ))
        long_enc  = tokenizer.encode(make_example(
            claim="A longer claim.",
            document="A longer document. " * 5,
            char_spans=[(0, 8)],
            example_id="l",
        ))
        short_len = short_enc["input_ids"].size(0)
        long_len  = long_enc["input_ids"].size(0)
        batch     = collator([short_enc, long_enc])

        if long_len > short_len:
            assert (batch["attention_mask"][0, short_len:] == 0).all()

    def test_padded_span_labels_use_ignore_index(self, tokenizer, collator):
        short_enc = tokenizer.encode(make_example(
            claim="Hi.", document="Doc.", char_spans=[(0, 3)], example_id="s"
        ))
        long_enc  = tokenizer.encode(make_example(
            claim="A longer claim.",
            document="A longer document. " * 5,
            char_spans=[(0, 8)],
            example_id="l",
        ))
        short_len = short_enc["input_ids"].size(0)
        long_len  = long_enc["input_ids"].size(0)
        batch     = collator([short_enc, long_enc])

        if long_len > short_len:
            assert (batch["span_labels"][0, short_len:] == SPAN_IGNORE_INDEX).all()

    def test_verdict_labels_dtype(self, collator, encoded_examples):
        batch = collator(encoded_examples)
        assert batch["verdict_labels"].dtype == torch.long

    def test_entity_spans_pass_through_as_lists(self, tokenizer, collator):
        enc = tokenizer.encode(make_example(example_id="ent"))
        enc["claim_entity_spans"] = [(1, 3), (5, 7)]
        enc["doc_entity_spans"]   = [(2, 4)]
        batch = collator([enc])

        assert isinstance(batch["claim_entity_spans"],    list)
        assert isinstance(batch["claim_entity_spans"][0], list)
        assert batch["claim_entity_spans"][0] == [(1, 3), (5, 7)]
        assert batch["doc_entity_spans"][0]   == [(2, 4)]

    def test_missing_entity_spans_default_empty(self, tokenizer, collator):
        enc   = tokenizer.encode(make_example(example_id="no-ent"))
        batch = collator([enc])
        assert batch["claim_entity_spans"][0] == []
        assert batch["doc_entity_spans"][0]   == []

    def test_single_example_does_not_crash(self, collator, encoded_examples):
        batch = collator([encoded_examples[0]])
        assert batch["input_ids"].shape[0] == 1

    def test_span_label_values_remain_valid_after_padding(self, collator, encoded_examples):
        batch  = collator(encoded_examples)
        unique = set(batch["span_labels"].flatten().tolist())
        assert unique.issubset({-1.0, 0.0, 1.0})
