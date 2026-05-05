"""
tests/data/test_wice.py

Integration tests for WiCEProcessor.
All tests marked slow — download from HuggingFace on first run.

Run with:
    pytest tests/data/test_wice.py -v -m slow
"""

import pytest

from verispan.data.schema import VerificationExample
from verispan.data.wice import WiCEProcessor


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def wice_test(tmp_path_factory):
    cache = tmp_path_factory.mktemp("hf_cache")
    proc  = WiCEProcessor(cache_dir=str(cache))
    return proc.load("train")


class TestWiCEProcessor:

    def test_returns_nonempty_list(self, wice_test):
        assert len(wice_test) > 0

    def test_all_correct_type(self, wice_test):
        for ex in wice_test[:10]:
            assert isinstance(ex, VerificationExample)

    def test_source_field(self, wice_test):
        for ex in wice_test[:10]:
            assert ex.source == "wice"

    def test_all_verdicts_valid(self, wice_test):
        for ex in wice_test:
            assert ex.verdict in {0, 1, 2}

    def test_all_three_classes_present(self, wice_test):
        verdicts = {ex.verdict for ex in wice_test}
        assert verdicts == {0, 1, 2}, f"Missing classes: {verdicts}"

    def test_char_spans_within_document(self, wice_test):
        for ex in wice_test:
            for start, end in ex.evidence_char_spans:
                assert 0 <= start < end <= len(ex.document)
