"""
tests/processing/test_dataset.py

Unit tests for ClaimVerificationDataset.
Fast — synthetic data only.
"""

import json

import pytest
import torch

from verispan.processing.dataset import ClaimVerificationDataset
from factories import make_example, make_examples


class TestClaimVerificationDataset:

    def test_length(self, tokenizer, examples):
        ds = ClaimVerificationDataset(examples, tokenizer, precompute=False)
        assert len(ds) == len(examples)

    def test_getitem_returns_dict(self, tokenizer, examples):
        ds   = ClaimVerificationDataset(examples, tokenizer, precompute=False)
        item = ds[0]
        assert isinstance(item, dict)
        assert "input_ids" in item

    def test_precompute_and_lazy_give_identical_output(self, tokenizer, examples):
        ds_pre  = ClaimVerificationDataset(examples, tokenizer, precompute=True)
        ds_lazy = ClaimVerificationDataset(examples, tokenizer, precompute=False)
        for i in range(len(examples)):
            assert torch.equal(ds_pre[i]["input_ids"], ds_lazy[i]["input_ids"]), \
                f"Mismatch at index {i}"

    def test_label_distribution_sums_to_total(self, tokenizer, examples):
        ds   = ClaimVerificationDataset(examples, tokenizer, precompute=False)
        dist = ds.label_distribution()
        assert sum(dist.values()) == len(examples)

    def test_label_distribution_has_all_classes(self, tokenizer, examples):
        ds   = ClaimVerificationDataset(examples, tokenizer, precompute=False)
        dist = ds.label_distribution()
        assert "SUPPORTS"       in dist
        assert "REFUTES"        in dist
        assert "NOT ENOUGH INFO" in dist

    def test_entity_spans_loaded_from_map(self, tokenizer, examples):
        entity_map = {
            "sup-0": {
                "claim_entity_spans": [(1, 3)],
                "doc_entity_spans":   [(2, 5)],
            }
        }
        ds   = ClaimVerificationDataset(
            examples, tokenizer,
            entity_span_map=entity_map,
            precompute=False,
        )
        item = ds[0]   # sup-0
        assert len(item["claim_entity_spans"]) == 1
        assert len(item["doc_entity_spans"])   == 1

    def test_missing_entity_spans_default_empty(self, tokenizer, examples):
        ds   = ClaimVerificationDataset(examples, tokenizer, precompute=False)
        item = ds[0]
        assert item["claim_entity_spans"] == []
        assert item["doc_entity_spans"]   == []

    def test_repr_does_not_crash(self, tokenizer, examples):
        ds = ClaimVerificationDataset(examples, tokenizer, precompute=False)
        r  = repr(ds)
        assert "ClaimVerificationDataset" in r

    def test_works_inside_dataloader(self, tokenizer, collator, examples):
        from torch.utils.data import DataLoader
        ds     = ClaimVerificationDataset(examples, tokenizer, precompute=True)
        loader = DataLoader(ds, batch_size=2, collate_fn=collator, shuffle=False)
        batches = list(loader)
        assert len(batches) == len(examples) // 2
        for batch in batches:
            assert "input_ids"      in batch
            assert "verdict_labels" in batch
            assert batch["input_ids"].shape[0] == 2
