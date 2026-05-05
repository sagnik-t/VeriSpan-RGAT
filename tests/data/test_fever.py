"""
tests/data/test_fever.py

Integration tests for FEVERProcessor.
All tests are marked slow — they download from HuggingFace on first run.

Run with:
    pytest tests/data/test_fever.py -v -m slow
"""

import pytest

from verispan.data.schema import VerificationExample
from verispan.data.fever import FEVERProcessor


pytestmark = pytest.mark.slow   # mark every test in this file as slow


@pytest.fixture(scope="module")
def fever_dev(tmp_path_factory):
    cache = tmp_path_factory.mktemp("hf_cache")
    proc  = FEVERProcessor(
        cache_dir=str(cache),
        max_doc_sentences=3,
    )
    return proc.load("dev")


class TestFEVERProcessor:

    def test_returns_nonempty_list(self, fever_dev):
        assert len(fever_dev) > 0

    def test_all_examples_are_correct_type(self, fever_dev):
        for ex in fever_dev[:10]:
            assert isinstance(ex, VerificationExample)

    def test_source_field(self, fever_dev):
        for ex in fever_dev[:10]:
            assert ex.source == "fever"

    def test_all_verdicts_valid(self, fever_dev):
        for ex in fever_dev:
            assert ex.verdict in {0, 1, 2}

    def test_all_three_verdict_classes_present(self, fever_dev):
        verdicts = {ex.verdict for ex in fever_dev}
        assert verdicts == {0, 1, 2}, f"Missing classes: {verdicts}"

    def test_supports_refutes_have_document(self, fever_dev):
        for ex in fever_dev[:100]:
            if ex.verdict in {0, 1}:
                assert ex.document, (
                    f"Empty document for {ex.verdict_name} example {ex.example_id}"
                )

    def test_char_spans_within_document(self, fever_dev):
        for ex in fever_dev[:100]:
            for start, end in ex.evidence_char_spans:
                assert start >= 0
                assert end > start
                assert end <= len(ex.document), (
                    f"Span ({start},{end}) out of bounds for doc len {len(ex.document)}"
                )

    def test_nei_examples_have_empty_spans(self, fever_dev):
        nei = [ex for ex in fever_dev if ex.verdict == 2]
        assert len(nei) > 0
        for ex in nei[:10]:
            assert ex.evidence_char_spans == []

    def test_skip_nei_flag(self, tmp_path):
        proc     = FEVERProcessor(
            cache_dir=str(tmp_path / "hf_cache"),
            max_doc_sentences=3,
            skip_nei=True,
        )
        examples = proc.load("dev")
        verdicts = {ex.verdict for ex in examples}
        assert 2 not in verdicts, "NEI examples should be excluded when skip_nei=True"
