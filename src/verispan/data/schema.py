"""
schema.py — Canonical data structures for VeriSpan-RGAT.

All dataset loaders (FEVER, SciFact, WiCE) convert their raw format into
VerificationExample.  Everything downstream (tokenization, dataset, model)
works with this single representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Label maps
# ──────────────────────────────────────────────────────────────────────────────

LABEL2ID: dict[str, int] = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# Covers FEVER, SciFact ("SUPPORT"/"CONTRADICT"), and WiCE variant strings.
_CANONICAL: dict[str, str] = {
    "SUPPORTS": "SUPPORTS",
    "SUPPORTED": "SUPPORTS",
    "SUPPORT": "SUPPORTS",
    "REFUTES": "REFUTES",
    "REFUTED": "REFUTES",
    "CONTRADICT": "REFUTES",
    "NOT ENOUGH INFO": "NOT ENOUGH INFO",
    "NEI": "NOT ENOUGH INFO",
    "NOTENOUGHINFO": "NOT ENOUGH INFO",
    "NOT_ENOUGH_INFO": "NOT ENOUGH INFO",
}


def normalise_label(raw: str) -> int:
    """Map any dataset-specific label string → 0 / 1 / 2."""
    key = raw.strip().upper()
    canonical = _CANONICAL.get(key)
    if canonical is None:
        raise ValueError(
            f"Unknown label {raw!r}. "
            f"Expected one of: {list(_CANONICAL.keys())}"
        )
    return LABEL2ID[canonical]


# ──────────────────────────────────────────────────────────────────────────────
# Core example dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VerificationExample:
    """
    One (claim, document) pair with ground-truth labels.

    Attributes
    ----------
    example_id : str
        Unique identifier (stringified row id from the originating dataset).
    claim : str
        The factual claim to be verified.
    document : str
        Concatenation of the evidence sentences (space-joined).
        Empty string for NEI examples where no evidence is annotated.
    verdict : int
        0 = SUPPORTS, 1 = REFUTES, 2 = NOT ENOUGH INFO.
    evidence_char_spans : list of (start, end)
        Character-level [start, end) spans **within `document`** that
        constitute annotated evidence.  End is exclusive.
        Empty for NEI examples.
    evidence_sentence_texts : list of str
        The individual evidence sentences before joining.  Kept for
        qualitative inspection and error analysis.
    source : str
        Dataset identifier: 'fever' | 'scifact' | 'wice'.
    """

    example_id: str
    claim: str
    document: str
    verdict: int
    evidence_char_spans: List[Tuple[int, int]]
    evidence_sentence_texts: List[str] = field(default_factory=list)
    source: str = "fever"

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def verdict_name(self) -> str:
        return ID2LABEL[self.verdict]

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence_char_spans)

    def __repr__(self) -> str:
        return (
            f"VerificationExample("
            f"id={self.example_id!r}, "
            f"verdict={self.verdict_name!r}, "
            f"claim={self.claim[:60]!r}{'...' if len(self.claim) > 60 else ''})"
        )
