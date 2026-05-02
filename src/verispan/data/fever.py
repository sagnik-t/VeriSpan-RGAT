"""
fever.py — FEVER dataset loader for VeriSpan-RGAT.

Two-step process:
  1. Load wiki sentence text from HuggingFace 'fever' (wiki_pages config).
     This builds an in-memory lookup: page_title → {sent_id → text}.
  2. Load claims + evidence metadata from 'fever' v1.0 and assemble
     VerificationExample objects.

Both datasets are downloaded automatically by the HuggingFace datasets
library on first run and cached locally.

Usage
-----
    from verispan.data.fever import FEVERProcessor

    proc = FEVERProcessor(cache_dir="data/cache")
    train = proc.load("train")   # List[VerificationExample]
    dev   = proc.load("dev")
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

from .schema import VerificationExample, normalise_label

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Wiki sentence database
# ──────────────────────────────────────────────────────────────────────────────

class WikiSentenceDB:
    """
    In-memory lookup built from the HuggingFace 'fever' wiki_pages split.

    Structure after loading:
        _db[page_title][sent_id] = sentence_text

    The wiki_pages split ships ~5.4M sentences (~2GB download, ~4GB in RAM).
    Loading takes ≈ 2–3 min on first call; subsequent calls hit the HF cache
    and take ≈ 30–60 s.

    A module-level singleton (_WIKI_DB) means the database is built at most
    once per Python process regardless of how many FEVERProcessor instances
    exist.
    """

    def __init__(self) -> None:
        self._db: Dict[str, Dict[int, str]] = {}
        self._loaded: bool = False

    # ── public API ───────────────────────────────────────────────────────────

    def load(self, cache_dir: Optional[str] = None) -> None:
        """Download (or load from cache) and parse all FEVER wiki pages."""
        if self._loaded:
            return

        logger.info(
            "Loading FEVER wiki pages from HuggingFace "
            "(first run ~2–3 min; subsequent runs use cache) ..."
        )
        wiki_ds = load_dataset(
            "fever",
            "wiki_pages",
            split="wikipedia_pages",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        for row in wiki_ds:
            page_title: str = row["id"]
            lines_raw: str = row.get("lines", "") or ""
            self._db[page_title] = self._parse_lines(lines_raw)

        self._loaded = True
        logger.info(f"WikiSentenceDB ready — {len(self._db):,} pages loaded.")

    def get(self, page: str, sent_id: int) -> Optional[str]:
        """Return the sentence text for (page, sent_id), or None if absent."""
        return self._db.get(page, {}).get(sent_id)

    def __len__(self) -> int:
        return len(self._db)

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_lines(raw: str) -> Dict[int, str]:
        """
        Parse the tab-separated 'lines' field:
            '0\\tFirst sentence.\\n1\\tSecond sentence.\\n...'
        into {sent_id: text}.
        """
        result: Dict[int, str] = {}
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                try:
                    result[int(parts[0])] = parts[1].strip()
                except ValueError:
                    pass  # non-integer sentence id — skip
        return result


# Module-level singleton so the DB is built at most once per process.
_WIKI_DB: Optional[WikiSentenceDB] = None


def _get_wiki_db(cache_dir: Optional[str] = None) -> WikiSentenceDB:
    global _WIKI_DB
    if _WIKI_DB is None:
        _WIKI_DB = WikiSentenceDB()
        _WIKI_DB.load(cache_dir=cache_dir)
    return _WIKI_DB


# ──────────────────────────────────────────────────────────────────────────────
# Evidence parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_evidence_groups(evidence: dict) -> List[List[Tuple[str, int]]]:
    """
    Convert the FEVER evidence dict into a list of annotation groups.

    HuggingFace fever v1.0 evidence structure:
        {
            'wikipedia_url': [[str | None, ...], ...],  # outer = groups
            'sent_id':        [[int, ...], ...],
            ...
        }

    NEI examples have url=None and sent_id=-1; these are filtered out.

    Returns
    -------
    List of groups, where each group is a list of (page_title, sent_id).
    """
    groups: List[List[Tuple[str, int]]] = []
    urls = evidence.get("wikipedia_url", [])
    sids = evidence.get("sent_id", [])

    for url_group, sid_group in zip(urls, sids):
        group: List[Tuple[str, int]] = []
        for url, sid in zip(url_group, sid_group):
            if url is not None and sid is not None and int(sid) >= 0:
                group.append((str(url), int(sid)))
        if group:
            groups.append(group)

    return groups


def _build_document(
    evidence_groups: List[List[Tuple[str, int]]],
    wiki_db: WikiSentenceDB,
    max_sentences: int,
) -> Tuple[str, List[Tuple[int, int]], List[str]]:
    """
    Deduplicate evidence sentences across all annotation groups, look up their
    text, and assemble:
        document         — evidence sentences joined by single spaces
        evidence_char_spans — [(start, end), ...] char positions in document
        sentence_texts   — individual sentence strings (for debugging)

    We cap at `max_sentences` to stay within DeBERTa's 512-token budget.
    All sentences in the SUPPORTS / REFUTES evidence are marked as evidence
    (sentence-level supervision, consistent with FEVER annotation granularity).
    """
    seen: set = set()
    ordered: List[str] = []

    for group in evidence_groups:
        for (page, sid) in group:
            key = (page, sid)
            if key in seen:
                continue
            text = wiki_db.get(page, sid)
            if text:
                seen.add(key)
                ordered.append(text)
            if len(ordered) >= max_sentences:
                break
        if len(ordered) >= max_sentences:
            break

    if not ordered:
        return "", [], []

    document = " ".join(ordered)

    # Compute char spans (every sentence in the document IS evidence)
    char_spans: List[Tuple[int, int]] = []
    offset = 0
    for sent in ordered:
        start = offset
        end = offset + len(sent)
        char_spans.append((start, end))
        offset = end + 1  # +1 accounts for the joining space

    return document, char_spans, ordered


# ──────────────────────────────────────────────────────────────────────────────
# FEVERProcessor
# ──────────────────────────────────────────────────────────────────────────────

class FEVERProcessor:
    """
    Loads a FEVER split and returns List[VerificationExample].

    Parameters
    ----------
    cache_dir : str, optional
        HuggingFace datasets cache directory.  Defaults to ~/.cache/huggingface.
    max_doc_sentences : int
        Maximum number of evidence sentences to include in the document.
        Caps the sequence length for DeBERTa.  Default: 5.
    skip_nei : bool
        Drop NOT ENOUGH INFO examples.  Useful when computing the contrastive
        loss which requires both Supports and Refutes within a batch.
        Default: False.
    """

    _SPLIT_MAP: dict[str, str] = {
        "train": "train",
        "dev": "labelled_dev",
        "test": "paper_test",
    }

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_doc_sentences: int = 5,
        skip_nei: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.max_doc_sentences = max_doc_sentences
        self.skip_nei = skip_nei

    # ── public ───────────────────────────────────────────────────────────────

    def load(self, split: str = "train") -> List[VerificationExample]:
        """
        Load and return examples for a FEVER split.

        Parameters
        ----------
        split : 'train' | 'dev' | 'test'
        """
        hf_split = self._SPLIT_MAP.get(split, split)
        logger.info(f"Loading FEVER split='{hf_split}' ...")

        raw_ds = load_dataset(
            "fever",
            "v1.0",
            split=hf_split,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        wiki_db = _get_wiki_db(self.cache_dir)
        return self._build_examples(raw_ds, split_name=split, wiki_db=wiki_db)

    # ── internals ────────────────────────────────────────────────────────────

    def _build_examples(
        self,
        raw_ds,
        split_name: str,
        wiki_db: WikiSentenceDB,
    ) -> List[VerificationExample]:
        examples: List[VerificationExample] = []
        n_skipped_nei = 0
        n_skipped_no_doc = 0

        for row in raw_ds:
            verdict = normalise_label(row["label"])

            if self.skip_nei and verdict == 2:
                n_skipped_nei += 1
                continue

            evidence_groups = _parse_evidence_groups(row["evidence"])

            document, char_spans, sent_texts = _build_document(
                evidence_groups, wiki_db, self.max_doc_sentences
            )

            if not document:
                if verdict == 2:
                    # NEI — empty document is valid; no evidence to annotate
                    document = ""
                    char_spans = []
                    sent_texts = []
                else:
                    # SUPPORTS / REFUTES with unresolvable evidence — skip
                    n_skipped_no_doc += 1
                    continue

            examples.append(
                VerificationExample(
                    example_id=str(row["id"]),
                    claim=str(row["claim"]),
                    document=document,
                    verdict=verdict,
                    evidence_char_spans=char_spans,
                    evidence_sentence_texts=sent_texts,
                    source="fever",
                )
            )

        logger.info(
            f"FEVER '{split_name}': {len(examples):,} examples loaded "
            f"({n_skipped_no_doc} dropped — no evidence text; "
            f"{n_skipped_nei} dropped — NEI skip flag)."
        )
        return examples
