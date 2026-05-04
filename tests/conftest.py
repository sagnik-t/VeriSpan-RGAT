"""
tests/conftest.py

Shared pytest fixtures available to every test file automatically.
Pytest discovers this file and makes all fixtures here available
without any import statement in the test files.

Factory functions (make_example, make_examples) live in factories.py
and are imported explicitly by test files that need them.
"""

from __future__ import annotations

from typing import List

import pytest

from verispan.data.schema import VerificationExample
from verispan.processing.collator import VerificationCollator
from verispan.processing.tokenization import VerificationTokenizer
from factories import make_examples

MODEL_NAME = "microsoft/deberta-v3-small"


@pytest.fixture(scope="session")
def tokenizer() -> VerificationTokenizer:
    """
    Session-scoped DeBERTa tokenizer.
    Created once for the entire test run; subsequent tests reuse it.
    Downloads the DeBERTa vocab on first call, then uses the HF cache.
    """
    return VerificationTokenizer(model_name=MODEL_NAME, max_length=128)


@pytest.fixture(scope="session")
def collator(tokenizer) -> VerificationCollator:
    """Session-scoped collator derived from the shared tokenizer."""
    return VerificationCollator(pad_token_id=tokenizer.pad_token_id)


@pytest.fixture(scope="session")
def examples() -> List[VerificationExample]:
    """Four synthetic examples with mixed verdicts."""
    return make_examples(4)


@pytest.fixture(scope="session")
def encoded_examples(tokenizer, examples):
    """Pre-encoded tensor dicts for the four synthetic examples."""
    return [tokenizer.encode(ex) for ex in examples]
