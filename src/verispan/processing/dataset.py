"""
dataset.py — PyTorch Dataset for VeriSpan-RGAT (verispan.processing).

ClaimVerificationDataset wraps List[VerificationExample] + VerificationTokenizer.
Optionally loads pre-computed entity spans from an EntitySpanMap, making
entity mention nodes available to the graph builder in Stage 3.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from ..data.schema import VerificationExample
from .entity import EntitySpanMap
from .tokenization import VerificationTokenizer

logger = logging.getLogger(__name__)


class ClaimVerificationDataset(Dataset):
    """
    Maps VerificationExample objects to tokenized tensor dicts.

    Parameters
    ----------
    examples : List[VerificationExample]
    tokenizer : VerificationTokenizer
    entity_span_map : EntitySpanMap, optional
        Pre-computed entity spans from EntityPreprocessor.process_and_save().
        If None, entity mention nodes are omitted from the graph.
    precompute : bool
        If True (default), tokenize all examples in __init__.
        Set to False for very large datasets.
    """

    def __init__(
        self,
        examples: List[VerificationExample],
        tokenizer: VerificationTokenizer,
        entity_span_map: Optional[EntitySpanMap] = None,
        precompute: bool = True,
    ) -> None:
        self.examples        = examples
        self.tokenizer       = tokenizer
        self.entity_span_map = entity_span_map or {}
        self.precompute      = precompute
        self._cache: Optional[List[Dict[str, Any]]] = None

        if precompute:
            logger.info(f"Pre-tokenizing {len(examples):,} examples ...")
            self._cache = [self._encode(ex) for ex in examples]
            logger.info("Pre-tokenization complete.")

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache[idx]
        return self._encode(self.examples[idx])

    # ── internals ────────────────────────────────────────────────────────────

    def _encode(self, ex: VerificationExample) -> Dict[str, Any]:
        encoded = self.tokenizer.encode(ex)

        # Attach entity spans if available for this example
        entity_info = self.entity_span_map.get(ex.example_id, {})
        encoded["claim_entity_spans"] = entity_info.get("claim_entity_spans", [])
        encoded["doc_entity_spans"]   = entity_info.get("doc_entity_spans", [])

        return encoded

    # ── Convenience factories ─────────────────────────────────────────────────

    @classmethod
    def from_fever(
        cls,
        split: str = "train",
        model_name: str = "microsoft/deberta-v3-small",
        max_length: int = 512,
        max_doc_sentences: int = 5,
        data_dir: str = "data/raw/fever",
        skip_nei: bool = False,
        entity_span_path: Optional[str] = None,
        precompute: bool = True,
    ) -> "ClaimVerificationDataset":
        from ..data.fever import FEVERProcessor
        from .entity import load_entity_spans

        processor = FEVERProcessor(
            data_dir=data_dir,
            max_doc_sentences=max_doc_sentences,
            skip_nei=skip_nei,
        )
        examples  = processor.load(split)
        tokenizer = VerificationTokenizer(model_name=model_name, max_length=max_length)
        entity_span_map = load_entity_spans(entity_span_path) if entity_span_path else None
        return cls(examples, tokenizer, entity_span_map=entity_span_map, precompute=precompute)

    @classmethod
    def from_scifact(
        cls,
        split: str = "test",
        model_name: str = "microsoft/deberta-v3-small",
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        entity_span_path: Optional[str] = None,
        precompute: bool = True,
    ) -> "ClaimVerificationDataset":
        from ..data.scifact import SciFatProcessor
        from .entity import load_entity_spans

        processor = SciFatProcessor(cache_dir=cache_dir)
        examples  = processor.load(split)
        tokenizer = VerificationTokenizer(model_name=model_name, max_length=max_length)
        entity_span_map = load_entity_spans(entity_span_path) if entity_span_path else None
        return cls(examples, tokenizer, entity_span_map=entity_span_map, precompute=precompute)

    @classmethod
    def from_wice(
        cls,
        split: str = "test",
        model_name: str = "microsoft/deberta-v3-small",
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        entity_span_path: Optional[str] = None,
        precompute: bool = True,
    ) -> "ClaimVerificationDataset":
        from ..data.wice import WiCEProcessor
        from .entity import load_entity_spans

        processor = WiCEProcessor(cache_dir=cache_dir)
        examples  = processor.load(split)
        tokenizer = VerificationTokenizer(model_name=model_name, max_length=max_length)
        entity_span_map = load_entity_spans(entity_span_path) if entity_span_path else None
        return cls(examples, tokenizer, entity_span_map=entity_span_map, precompute=precompute)

    # ── Debug helpers ─────────────────────────────────────────────────────────

    def label_distribution(self) -> Dict[str, int]:
        from collections import Counter
        from ..data.schema import ID2LABEL
        counts: Counter = Counter(ex.verdict for ex in self.examples)
        return {ID2LABEL[k]: v for k, v in sorted(counts.items())}

    def __repr__(self) -> str:
        return (
            f"ClaimVerificationDataset("
            f"n={len(self)}, "
            f"precomputed={self._cache is not None}, "
            f"entities={'yes' if self.entity_span_map else 'no'}, "
            f"dist={self.label_distribution()})"
        )
