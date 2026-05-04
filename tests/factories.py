"""
tests/factories.py

Factory functions for building synthetic VerificationExample objects.
Imported directly by test files that need fine-grained control over
example construction.

This is separate from conftest.py because conftest is a pytest-internal
file and should not be imported directly.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from verispan.data.schema import VerificationExample


def make_example(
    verdict: int = 0,
    claim: str = "The Eiffel Tower is in Berlin.",
    document: str = "The Eiffel Tower is located in Paris, France.",
    char_spans: Optional[List[Tuple[int, int]]] = None,
    source: str = "test",
    example_id: str = "ex-0",
) -> VerificationExample:
    """Factory for a minimal VerificationExample."""
    return VerificationExample(
        example_id=example_id,
        claim=claim,
        document=document,
        verdict=verdict,
        evidence_char_spans=char_spans if char_spans is not None else [(36, 49)],
        evidence_sentence_texts=[document] if document else [],
        source=source,
    )


def make_examples(n: int = 4) -> List[VerificationExample]:
    """Return n examples with mixed verdicts (SUPPORTS, REFUTES, NEI, SUPPORTS)."""
    pool = [
        make_example(verdict=0, example_id="sup-0"),
        make_example(
            verdict=1, example_id="ref-0",
            claim="The Eiffel Tower is in Berlin.",
            char_spans=[(36, 49)],
        ),
        make_example(
            verdict=2, example_id="nei-0",
            document="", char_spans=[],
        ),
        make_example(verdict=0, example_id="sup-1"),
    ]
    return pool[:n]
