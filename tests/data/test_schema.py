"""
tests/data/test_schema.py

Unit tests for verispan.data.schema.
No network, no GPU, no HuggingFace — pure Python.
"""

import pytest

from verispan.data.schema import ID2LABEL, LABEL2ID, VerificationExample, normalise_label
from factories import make_example


class TestLabelMaps:

    def test_label_maps_are_inverse(self):
        for label_id, name in ID2LABEL.items():
            assert LABEL2ID[name] == label_id

    def test_all_three_classes_present(self):
        assert set(LABEL2ID.values()) == {0, 1, 2}

    @pytest.mark.parametrize("raw,expected", [
        ("SUPPORTS",          0),
        ("Supported",         0),
        ("SUPPORT",           0),
        ("REFUTES",           1),
        ("REFUTED",           1),
        ("CONTRADICT",        1),
        ("NOT ENOUGH INFO",   2),
        ("NEI",               2),
        ("not_enough_info",   2),
        ("NOT_ENOUGH_INFO",   2),
        ("notenoughinfo",     2),
    ])
    def test_normalise_label(self, raw, expected):
        assert normalise_label(raw) == expected

    def test_normalise_label_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown label"):
            normalise_label("MAYBE")

    def test_normalise_label_strips_whitespace(self):
        assert normalise_label("  SUPPORTS  ") == 0


class TestVerificationExample:

    def test_construction(self):
        ex = make_example()
        assert ex.verdict == 0
        assert ex.verdict_name == "SUPPORTS"
        assert ex.has_evidence is True
        assert ex.source == "test"

    def test_nei_has_no_evidence(self):
        ex = make_example(verdict=2, document="", char_spans=[])
        assert ex.has_evidence is False

    def test_verdict_name_all_classes(self):
        for verdict_id, name in ID2LABEL.items():
            ex = make_example(verdict=verdict_id)
            assert ex.verdict_name == name

    def test_repr_contains_key_info(self):
        ex = make_example(example_id="test-42", verdict=1)
        r  = repr(ex)
        assert "test-42"  in r
        assert "REFUTES"  in r

    def test_repr_truncates_long_claim(self):
        ex = make_example(claim="x" * 200)
        r  = repr(ex)
        assert "..." in r

    def test_empty_document_allowed(self):
        ex = make_example(verdict=2, document="", char_spans=[])
        assert ex.document == ""

    def test_multiple_char_spans(self):
        ex = make_example(char_spans=[(0, 5), (10, 20)])
        assert len(ex.evidence_char_spans) == 2
        assert ex.has_evidence is True
