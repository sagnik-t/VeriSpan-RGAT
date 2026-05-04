"""
scifact.py — SciFact dataset loader for VeriSpan-RGAT.

Reads from locally downloaded files produced by scripts/setup_data.py.

Expected directory structure
-----------------------------
    data/raw/scifact/
    ├── claims_train.jsonl
    ├── claims_dev.jsonl
    ├── corpus.jsonl
    └── claims_test.jsonl   (optional, may not have labels)

AI2 SciFact format (claims_*.jsonl)
-------------------------------------
    {
        "id": int,
        "claim": str,
        "evidence": {
            "<corpus_id>": [
                {
                    "sentences": [int, ...],   <- rationale sentence indices
                    "label": "SUPPORT" | "CONTRADICT"
                }
            ]
        },
        "cited_doc_ids": [int, ...]
    }

AI2 SciFact format (corpus.jsonl)
-----------------------------------
    {
        "doc_id": int,
        "title": str,
        "abstract": [str, ...],   <- list of sentences
        "structured": bool
    }

Usage
-----
    from verispan.data.scifact import SciFatProcessor

    proc = SciFatProcessor(data_dir="data/raw/scifact")
    test = proc.load("dev")    # List[VerificationExample]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import VerificationExample, normalise_label

logger = logging.getLogger(__name__)


class SciFatProcessor:
    """
    Loads SciFact from locally downloaded AI2 JSONL files.

    Parameters
    ----------
    data_dir : str
        Directory containing claims_*.jsonl and corpus.jsonl.
        Default: "data/raw/scifact"
    max_sentences : int
        Maximum abstract sentences per document.  Default: 10.
    """

    _SPLIT_FILES: Dict[str, str] = {
        "train":      "claims_train.jsonl",
        "dev":        "claims_dev.jsonl",
        "test":       "claims_test.jsonl",
        "validation": "claims_dev.jsonl",
    }

    def __init__(
        self,
        data_dir: str = "data/raw/scifact",
        max_sentences: int = 10,
        # Legacy parameter — ignored, kept for backwards compatibility
        cache_dir: Optional[str] = None,
    ) -> None:
        self.data_dir      = Path(data_dir)
        self.max_sentences = max_sentences
        self._corpus: Optional[Dict[int, List[str]]] = None

    # ── public ───────────────────────────────────────────────────────────────

    def load(self, split: str = "dev") -> List[VerificationExample]:
        fname = self._SPLIT_FILES.get(split, "claims_dev.jsonl")
        path  = self.data_dir / fname

        if not path.exists():
            raise FileNotFoundError(
                f"SciFact {split} file not found: {path}\n"
                f"Run: python scripts/setup_data.py --only scifact"
            )

        self._ensure_corpus_loaded()
        logger.info(f"Loading SciFact split='{split}' from {path.name} ...")
        return self._build_examples(path, split)

    # ── internals ────────────────────────────────────────────────────────────

    def _ensure_corpus_loaded(self) -> None:
        if self._corpus is not None:
            return
        corpus_path = self.data_dir / "corpus.jsonl"
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"SciFact corpus not found: {corpus_path}\n"
                f"Run: python scripts/setup_data.py --only scifact"
            )
        logger.info("Loading SciFact corpus ...")
        self._corpus = {}
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                self._corpus[int(row["doc_id"])] = row["abstract"]
        logger.info(f"SciFact corpus: {len(self._corpus):,} abstracts loaded.")

    def _build_examples(
        self, path: Path, split_name: str
    ) -> List[VerificationExample]:
        examples: List[VerificationExample] = []
        n_skipped = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row      = json.loads(line)
                claim_id = str(row["id"])
                claim    = str(row["claim"])
                evidence = row.get("evidence") or {}

                if not evidence:
                    # NEI example
                    examples.append(VerificationExample(
                        example_id=claim_id,
                        claim=claim,
                        document="",
                        verdict=2,
                        evidence_char_spans=[],
                        source="scifact",
                    ))
                    continue

                doc_text, char_spans, sent_texts, verdict = self._resolve(evidence)
                if doc_text is None:
                    n_skipped += 1
                    continue

                examples.append(VerificationExample(
                    example_id=claim_id,
                    claim=claim,
                    document=doc_text,
                    verdict=verdict,
                    evidence_char_spans=char_spans,
                    evidence_sentence_texts=sent_texts,
                    source="scifact",
                ))

        logger.info(
            f"SciFact '{split_name}': {len(examples):,} examples "
            f"({n_skipped} skipped — missing corpus entry)."
        )
        return examples

    def _resolve(
        self, evidence: dict
    ) -> Tuple[Optional[str], List[Tuple[int, int]], List[str], int]:
        """Resolve evidence dict → (document, char_spans, sent_texts, verdict)."""
        for corpus_id_str, annotations in evidence.items():
            corpus_id = int(corpus_id_str)
            abstract  = self._corpus.get(corpus_id)
            if not abstract:
                continue

            ann            = annotations[0]
            label          = normalise_label(ann["label"])
            rationale_sids = set(ann.get("sentences", []))

            # Use rationale sentences first, fill with others up to max
            selected_sids = sorted(rationale_sids)[: self.max_sentences]
            extras = [
                i for i in range(len(abstract))
                if i not in rationale_sids
            ][: self.max_sentences - len(selected_sids)]
            all_sids = sorted(selected_sids + extras)

            sents = [abstract[i] for i in all_sids if i < len(abstract)]
            if not sents:
                continue

            rationale_positions = {
                pos for pos, sid in enumerate(all_sids) if sid in rationale_sids
            }

            document, char_spans, sent_texts = self._build_document(
                sents, rationale_positions
            )
            return document, char_spans, sent_texts, label

        return None, [], [], -1

    @staticmethod
    def _build_document(
        sentences: List[str], rationale_positions: set
    ) -> Tuple[str, List[Tuple[int, int]], List[str]]:
        document   = " ".join(sentences)
        char_spans: List[Tuple[int, int]] = []
        offset = 0
        for pos, sent in enumerate(sentences):
            end = offset + len(sent)
            if pos in rationale_positions:
                char_spans.append((offset, end))
            offset = end + 1
        return document, char_spans, sentences
