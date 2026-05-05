"""
tests/data/test_scifact.py

Integration tests for SciFatProcessor.
All tests marked slow — download from HuggingFace on first run.

Run with:
    pytest tests/data/test_scifact.py -v -m slow
"""

import pytest

from verispan.data.schema import VerificationExample
from verispan.data.scifact import SciFatProcessor


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def scifact_test():
    proc = SciFatProcessor(data_dir="data/raw/scifact")
    return proc.load("dev")


class TestSciFatProcessor:

    def test_returns_nonempty_list(self, scifact_test):
        assert len(scifact_test) > 0

    def test_all_correct_type(self, scifact_test):
        for ex in scifact_test[:10]:
            assert isinstance(ex, VerificationExample)

    def test_source_field(self, scifact_test):
        for ex in scifact_test[:10]:
            assert ex.source == "scifact"

    def test_all_verdicts_valid(self, scifact_test):
        for ex in scifact_test:
            assert ex.verdict in {0, 1, 2}

    def test_supports_refutes_have_document(self, scifact_test):
        for ex in scifact_test:
            if ex.verdict in {0, 1}:
                assert ex.document

    def test_char_spans_within_document(self, scifact_test):
        for ex in scifact_test:
            for start, end in ex.evidence_char_spans:
                assert 0 <= start < end <= len(ex.document), (
                    f"Invalid span ({start},{end}) for doc len {len(ex.document)}"
                )

    def test_evidence_spans_nonempty_for_labelled(self, scifact_test):
        """SUPPORTS/REFUTES examples in SciFact always have rationale annotations."""
        for ex in scifact_test:
            if ex.verdict in {0, 1}:
                assert ex.has_evidence, (
                    f"Expected evidence for {ex.verdict_name} example {ex.example_id}"
                )
