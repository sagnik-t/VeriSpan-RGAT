"""
verispan.data — Raw data loading only.

Exports schema types and dataset processors.
All tensor transformations live in verispan.processing.

Public API
----------
    from verispan.data import (
        VerificationExample,
        LABEL2ID, ID2LABEL, normalise_label,
        FEVERProcessor,
        SciFatProcessor,
        WiCEProcessor,
    )
"""

from .schema import (
    VerificationExample,
    LABEL2ID,
    ID2LABEL,
    normalise_label,
)
from .fever import FEVERProcessor
from .scifact import SciFatProcessor
from .wice import WiCEProcessor

__all__ = [
    # Schema
    "VerificationExample",
    "LABEL2ID",
    "ID2LABEL",
    "normalise_label",
    # Loaders
    "FEVERProcessor",
    "SciFatProcessor",
    "WiCEProcessor",
]
