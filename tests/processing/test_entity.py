"""
tests/processing/test_entity.py

Tests for EntityPreprocessor and load_entity_spans.

Unit tests (fast, no SpaCy):
    TestLoadEntitySpans — roundtrip JSON serialisation

Slow tests (require SpaCy + en_core_web_sm):
    TestEntityPreprocessor — full NER + token alignment

Run fast only:
    pytest tests/processing/test_entity.py -v -m "not slow"

Run all:
    pytest tests/processing/test_entity.py -v
"""

import json

import pytest

from verispan.processing.entity import load_entity_spans
from factories import make_example


class TestLoadEntitySpans:
    """Unit tests — no SpaCy required."""

    def test_roundtrip(self, tmp_path):
        span_map = {
            "ex-0": {
                "claim_entity_spans": [[1, 3], [5, 7]],
                "doc_entity_spans":   [[2, 4]],
            },
            "ex-1": {
                "claim_entity_spans": [],
                "doc_entity_spans":   [[0, 2]],
            },
        }
        path = tmp_path / "entities.json"
        path.write_text(json.dumps(span_map))

        loaded = load_entity_spans(str(path))
        assert set(loaded.keys()) == {"ex-0", "ex-1"}
        assert loaded["ex-0"]["claim_entity_spans"] == [(1, 3), (5, 7)]
        assert loaded["ex-1"]["doc_entity_spans"]   == [(0, 2)]

    def test_empty_spans_preserved(self, tmp_path):
        span_map = {
            "ex-0": {"claim_entity_spans": [], "doc_entity_spans": []}
        }
        path = tmp_path / "entities.json"
        path.write_text(json.dumps(span_map))
        loaded = load_entity_spans(str(path))
        assert loaded["ex-0"]["claim_entity_spans"] == []

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_entity_spans("/nonexistent/path/entities.json")


class TestEntityPreprocessor:
    """Slow tests — require SpaCy en_core_web_sm."""

    @pytest.fixture(autouse=True)
    def require_spacy(self):
        pytest.importorskip("spacy", reason="SpaCy not installed")
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("SpaCy model 'en_core_web_sm' not downloaded. "
                        "Run: python -m spacy download en_core_web_sm")

    @pytest.mark.slow
    def test_process_and_save_creates_file(self, tokenizer, tmp_path):
        from verispan.processing.entity import EntityPreprocessor
        proc = EntityPreprocessor(spacy_model="en_core_web_sm")
        examples = [make_example(
            claim="Apple is based in California.",
            document="Apple Inc. was founded by Steve Jobs in Cupertino.",
            char_spans=[(0, 5)],
            example_id="ent-0",
        )]
        out = tmp_path / "entities.json"
        proc.process_and_save(examples, tokenizer, str(out))
        assert out.exists()

    @pytest.mark.slow
    def test_output_keys_present(self, tokenizer, tmp_path):
        from verispan.processing.entity import EntityPreprocessor
        proc = EntityPreprocessor(spacy_model="en_core_web_sm")
        examples = [make_example(
            claim="Apple is based in California.",
            document="Apple Inc. was founded in Cupertino.",
            char_spans=[(0, 5)],
            example_id="ent-0",
        )]
        out      = tmp_path / "entities.json"
        span_map = proc.process_and_save(examples, tokenizer, str(out))
        assert "ent-0" in span_map
        assert "claim_entity_spans" in span_map["ent-0"]
        assert "doc_entity_spans"   in span_map["ent-0"]

    @pytest.mark.slow
    def test_entities_detected(self, tokenizer, tmp_path):
        """Apple and California are standard NER entities — should be found."""
        from verispan.processing.entity import EntityPreprocessor
        proc = EntityPreprocessor(spacy_model="en_core_web_sm")
        examples = [make_example(
            claim="Apple is based in California.",
            document="Apple Inc. was founded in Cupertino, California.",
            char_spans=[(0, 5)],
            example_id="ent-0",
        )]
        out      = tmp_path / "entities.json"
        span_map = proc.process_and_save(examples, tokenizer, str(out))
        claim_spans = span_map["ent-0"]["claim_entity_spans"]
        assert len(claim_spans) > 0, "Expected at least one named entity in claim"

    @pytest.mark.slow
    def test_token_spans_are_valid_indices(self, tokenizer, tmp_path):
        """All returned token spans must be within the tokenized sequence length."""
        from verispan.processing.entity import EntityPreprocessor
        proc = EntityPreprocessor(spacy_model="en_core_web_sm")
        ex   = make_example(
            claim="Google was founded in Menlo Park.",
            document="Google LLC is an American company.",
            char_spans=[(0, 6)],
            example_id="ent-0",
        )
        out      = tmp_path / "entities.json"
        span_map = proc.process_and_save([ex], tokenizer, str(out))

        enc      = tokenizer.encode(ex)
        seq_len  = enc["input_ids"].size(0)

        for start, end in span_map["ent-0"]["claim_entity_spans"]:
            assert 0 <= start <= end < seq_len

        for start, end in span_map["ent-0"]["doc_entity_spans"]:
            assert 0 <= start <= end < seq_len

    @pytest.mark.slow
    def test_empty_document_does_not_crash(self, tokenizer, tmp_path):
        from verispan.processing.entity import EntityPreprocessor
        proc = EntityPreprocessor(spacy_model="en_core_web_sm")
        ex   = make_example(verdict=2, document="", char_spans=[], example_id="nei-0")
        out  = tmp_path / "entities.json"
        span_map = proc.process_and_save([ex], tokenizer, str(out))
        assert "nei-0" in span_map
        assert span_map["nei-0"]["doc_entity_spans"] == []
