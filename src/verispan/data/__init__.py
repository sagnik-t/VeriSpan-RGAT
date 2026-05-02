"""
verispan.data — Data pipeline for VeriSpan-RGAT.

Public API
----------
    from verispan.data import (
        VerificationExample,
        LABEL2ID, ID2LABEL, normalise_label,
        FEVERProcessor,
        WikiSentenceDB,
        VerificationTokenizer,
        SPAN_IGNORE_INDEX,
        ClaimVerificationDataset,
        VerificationCollator,
        build_collator,
    )
"""

from .schema import (
    VerificationExample,
    LABEL2ID,
    ID2LABEL,
    normalise_label,
)
from .fever import FEVERProcessor, WikiSentenceDB
from .tokenization import VerificationTokenizer, SPAN_IGNORE_INDEX
from .dataset import ClaimVerificationDataset
from .collator import VerificationCollator, build_collator

__all__ = [
    # schema
    "VerificationExample",
    "LABEL2ID",
    "ID2LABEL",
    "normalise_label",
    # loaders
    "FEVERProcessor",
    "WikiSentenceDB",
    # tokenization
    "VerificationTokenizer",
    "SPAN_IGNORE_INDEX",
    # dataset / collator
    "ClaimVerificationDataset",
    "VerificationCollator",
    "build_collator",
]
