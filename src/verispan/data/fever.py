"""
fever.py — FEVER dataset loader for VeriSpan-RGAT.

Reads from locally downloaded JSONL files produced by scripts/download_fever.py.
Does NOT use HuggingFace loading scripts (which were deprecated in datasets v3).

Directory structure expected
----------------------------
    data/raw/fever/
    ├── train.jsonl
    ├── paper_dev.jsonl
    ├── paper_test.jsonl
    ├── shared_task_dev.jsonl
    └── wiki-pages/
        ├── wiki-001.jsonl
        ├── wiki-002.jsonl
        └── ...

Usage
-----
    from verispan.data.fever import FEVERProcessor

    proc = FEVERProcessor(data_dir="data/raw/fever")
    train = proc.load("train")   # List[VerificationExample]
    dev   = proc.load("dev")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import VerificationExample, normalise_label

logger = logging.getLogger(__name__)


# ── WikiSentenceDB ────────────────────────────────────────────────────────────

class WikiSentenceDB:
    """
    In-memory lookup built from the FEVER wiki-pages JSONL files.

        _db[page_title][sent_id] = sentence_text

    The wiki-pages directory contains ~109 JSONL files totalling ~5.4M sentences.
    Loading takes ~60-90s on first call; subsequent calls within the same
    process use the cached _db.

    A module-level singleton (_WIKI_DB) means the DB is built at most
    once per Python process.
    """

    def __init__(self) -> None:
        self._db: Dict[str, Dict[int, str]] = {}
        self._loaded: bool = False

    def load(self, wiki_dir: Path) -> None:
        if self._loaded:
            return
        if not wiki_dir.exists():
            raise FileNotFoundError(
                f"Wiki pages directory not found: {wiki_dir}\n"
                f"Run: python scripts/download_fever.py --data_dir data/raw/fever"
            )

        jsonl_files = sorted(wiki_dir.glob("wiki-*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(
                f"No wiki-*.jsonl files found in {wiki_dir}.\n"
                f"Run: python scripts/download_fever.py --data_dir data/raw/fever"
            )

        logger.info(
            f"Loading WikiSentenceDB from {len(jsonl_files)} files in {wiki_dir} ..."
        )
        for fpath in jsonl_files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    page_title: str    = row["id"]
                    lines_raw: str     = row.get("lines", "") or ""
                    self._db[page_title] = self._parse_lines(lines_raw)

        self._loaded = True
        logger.info(f"WikiSentenceDB ready — {len(self._db):,} pages loaded.")

    def get(self, page: str, sent_id: int) -> Optional[str]:
        return self._db.get(page, {}).get(sent_id)

    def __len__(self) -> int:
        return len(self._db)

    @staticmethod
    def _parse_lines(raw: str) -> Dict[int, str]:
        """
        Parse the tab-separated 'lines' field:
            '0\\tFirst sentence.\\n1\\tSecond sentence.\\n'
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
                    pass
        return result


# Module-level singleton
_WIKI_DB: Optional[WikiSentenceDB] = None


def _get_wiki_db(wiki_dir: Path) -> WikiSentenceDB:
    global _WIKI_DB
    if _WIKI_DB is None:
        _WIKI_DB = WikiSentenceDB()
        _WIKI_DB.load(wiki_dir)
    return _WIKI_DB


# ── Evidence parsing ──────────────────────────────────────────────────────────

def _parse_evidence_groups(
    evidence: list,
) -> List[List[Tuple[str, int]]]:
    """
    Parse FEVER evidence from the raw JSONL format.

    FEVER JSONL evidence structure (one claim):
        [
            [                          ← annotation group (annotator)
                [ann_id, ev_id, page_title, sent_id],
                ...
            ],
            ...
        ]

    Returns list of groups, each group is list of (page_title, sent_id).
    NEI entries have page_title=None or sent_id=-1 — filtered out.
    """
    groups: List[List[Tuple[str, int]]] = []
    for group in evidence:
        parsed: List[Tuple[str, int]] = []
        for ev_item in group:
            # ev_item = [ann_id, ev_id, page_title, sent_id]
            if len(ev_item) < 4:
                continue
            page = ev_item[2]
            sid  = ev_item[3]
            if page is not None and sid is not None and int(sid) >= 0:
                parsed.append((str(page), int(sid)))
        if parsed:
            groups.append(parsed)
    return groups


def _build_document(
    evidence_groups: List[List[Tuple[str, int]]],
    wiki_db: WikiSentenceDB,
    max_sentences: int,
) -> Tuple[str, List[Tuple[int, int]], List[str]]:
    """
    Resolve evidence (page, sent_id) pairs to text, deduplicate, and
    assemble the document string with char-level evidence spans.
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

    char_spans: List[Tuple[int, int]] = []
    offset = 0
    for sent in ordered:
        start = offset
        end   = offset + len(sent)
        char_spans.append((start, end))
        offset = end + 1

    return document, char_spans, ordered


# ── FEVERProcessor ────────────────────────────────────────────────────────────

class FEVERProcessor:
    """
    Loads FEVER splits from local JSONL files and returns List[VerificationExample].

    Parameters
    ----------
    data_dir : str
        Path to the directory containing FEVER JSONL files and wiki-pages/.
        Default: "data/raw/fever"
    max_doc_sentences : int
        Maximum evidence sentences per example.  Default: 5.
    skip_nei : bool
        Drop NOT ENOUGH INFO examples.  Default: False.
    """

    _SPLIT_MAP: Dict[str, str] = {
        "train": "train.jsonl",
        "dev":   "paper_dev.jsonl",
        "test":  "paper_test.jsonl",
        "shared_task_dev": "shared_task_dev.jsonl",
    }

    def __init__(
        self,
        data_dir: str = "data/raw/fever",
        max_doc_sentences: int = 5,
        skip_nei: bool = False,
        # Legacy parameter — ignored, kept for backwards compatibility
        cache_dir: Optional[str] = None,
    ) -> None:
        self.data_dir         = Path(data_dir)
        self.max_doc_sentences = max_doc_sentences
        self.skip_nei          = skip_nei

    # ── public ───────────────────────────────────────────────────────────────

    def load(self, split: str = "train") -> List[VerificationExample]:
        filename = self._SPLIT_MAP.get(split)
        if filename is None:
            raise ValueError(
                f"Unknown split {split!r}. "
                f"Valid splits: {list(self._SPLIT_MAP.keys())}"
            )

        claim_path = self.data_dir / filename
        if not claim_path.exists():
            raise FileNotFoundError(
                f"FEVER {split} file not found: {claim_path}\n"
                f"Run: python scripts/download_fever.py --data_dir {self.data_dir}"
            )

        # Load wiki DB (singleton — only built once per process)
        wiki_dir = self.data_dir / "wiki-pages"
        wiki_db  = _get_wiki_db(wiki_dir)

        logger.info(f"Loading FEVER split='{split}' from {claim_path} ...")
        return self._build_examples(claim_path, split_name=split, wiki_db=wiki_db)

    # ── internals ────────────────────────────────────────────────────────────

    def _build_examples(
        self,
        claim_path: Path,
        split_name: str,
        wiki_db: WikiSentenceDB,
    ) -> List[VerificationExample]:
        examples: List[VerificationExample] = []
        n_skipped_nei    = 0
        n_skipped_no_doc = 0

        with open(claim_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                verdict = normalise_label(row["label"])

                if self.skip_nei and verdict == 2:
                    n_skipped_nei += 1
                    continue

                evidence_groups = _parse_evidence_groups(row.get("evidence", []))

                document, char_spans, sent_texts = _build_document(
                    evidence_groups, wiki_db, self.max_doc_sentences
                )

                if not document:
                    if verdict == 2:
                        # NEI — empty document is valid
                        pass
                    else:
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
