"""
dataset.py — PyTorch Dataset for VeriSpan-RGAT.

ClaimVerificationDataset wraps a List[VerificationExample] and a
VerificationTokenizer.  It supports optional pre-tokenization (recommended
for training) to avoid re-tokenizing on every __getitem__ call.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from .schema import VerificationExample
from .tokenization import VerificationTokenizer

logger = logging.getLogger(__name__)


class ClaimVerificationDataset(Dataset):
    """
    Maps a list of VerificationExample objects into a PyTorch Dataset.

    Parameters
    ----------
    examples : List[VerificationExample]
        Pre-built examples from any dataset processor (FEVER, SciFact, WiCE).
    tokenizer : VerificationTokenizer
        Handles claim/document encoding and span label alignment.
    precompute : bool
        If True (default), tokenize all examples in __init__ and cache
        the results.  This costs ~N × seq_len memory upfront but eliminates
        per-step tokenization overhead during training.
        Set to False for very large datasets or memory-constrained runs.
    """

    def __init__(
        self,
        examples: List[VerificationExample],
        tokenizer: VerificationTokenizer,
        precompute: bool = True,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.precompute = precompute

        self._cache: Optional[List[Dict[str, Any]]] = None

        if precompute:
            logger.info(
                f"Pre-tokenizing {len(examples):,} examples "
                f"(set precompute=False to skip) ..."
            )
            self._cache = [tokenizer.encode(ex) for ex in examples]
            logger.info("Pre-tokenization complete.")

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache[idx]
        return self.tokenizer.encode(self.examples[idx])

    # ── Convenience factories ─────────────────────────────────────────────────

    @classmethod
    def from_fever(
        cls,
        split: str = "train",
        model_name: str = "microsoft/deberta-v3-small",
        max_length: int = 512,
        max_doc_sentences: int = 5,
        cache_dir: Optional[str] = None,
        skip_nei: bool = False,
        precompute: bool = True,
    ) -> "ClaimVerificationDataset":
        """
        One-liner factory: load FEVER, tokenize, return dataset.

        Example
        -------
        >>> train_ds = ClaimVerificationDataset.from_fever("train")
        >>> dev_ds   = ClaimVerificationDataset.from_fever("dev")
        """
        from .fever import FEVERProcessor

        processor = FEVERProcessor(
            cache_dir=cache_dir,
            max_doc_sentences=max_doc_sentences,
            skip_nei=skip_nei,
        )
        examples = processor.load(split)
        tokenizer = VerificationTokenizer(model_name=model_name, max_length=max_length)
        return cls(examples, tokenizer, precompute=precompute)

    # ── Debug helpers ─────────────────────────────────────────────────────────

    def label_distribution(self) -> Dict[str, int]:
        """Return {verdict_name: count} for the dataset."""
        from collections import Counter
        from .schema import ID2LABEL

        counts: Counter = Counter(ex.verdict for ex in self.examples)
        return {ID2LABEL[k]: v for k, v in sorted(counts.items())}

    def __repr__(self) -> str:
        dist = self.label_distribution()
        return (
            f"ClaimVerificationDataset("
            f"n={len(self)}, "
            f"precomputed={self._cache is not None}, "
            f"dist={dist})"
        )
