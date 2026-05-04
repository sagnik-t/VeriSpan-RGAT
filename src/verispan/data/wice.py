"""
wice.py — WiCE dataset loader for VeriSpan-RGAT.

Reads from locally downloaded files produced by scripts/setup_data.py.
Handles the ryokamoi/wice GitHub format with automatic field detection.

Expected directory structure
-----------------------------
    data/raw/wice/
    └── *.jsonl   (train/dev/test files, exact names depend on the repo)

WiCE format (Kamoi et al., EMNLP 2023)
----------------------------------------
Each line is one claim-evidence pair:
    {
        "claim": str,
        "evidence": str,
        "label": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
        "supporting_sentences": [int, ...]   <- optional rationale indices
    }

Field names may vary slightly — this loader auto-detects them.

Usage
-----
    from verispan.data.wice import WiCEProcessor

    proc = WiCEProcessor(data_dir="data/raw/wice")
    test = proc.load("test")    # List[VerificationExample]
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import VerificationExample, normalise_label

logger = logging.getLogger(__name__)


class WiCEProcessor:
    """
    Loads WiCE from locally downloaded JSONL files.

    Parameters
    ----------
    data_dir : str
        Directory containing WiCE JSONL files.
        Default: "data/raw/wice"
    max_sentences : int
        Maximum sentences from the evidence passage.  Default: 10.
    """

    def __init__(
        self,
        data_dir: str = "data/raw/wice",
        max_sentences: int = 10,
        # Legacy parameter — ignored
        cache_dir: Optional[str] = None,
    ) -> None:
        self.data_dir      = Path(data_dir)
        self.max_sentences = max_sentences

    # ── public ───────────────────────────────────────────────────────────────

    def load(self, split: str = "test") -> List[VerificationExample]:
        """
        Load a WiCE split.

        Resolves the filename by looking for any JSONL file whose name
        contains the split name.  Falls back to the first available file
        if no match is found.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"WiCE data directory not found: {self.data_dir}\n"
                f"Run: python scripts/setup_data.py --only wice"
            )

        path = self._resolve_split_file(split)
        if path is None:
            raise FileNotFoundError(
                f"No WiCE file found for split='{split}' in {self.data_dir}.\n"
                f"Available: {[f.name for f in self.data_dir.glob('*.jsonl')]}"
            )

        logger.info(f"Loading WiCE split='{split}' from {path.name} ...")
        return self._build_examples(path, split)

    # ── internals ────────────────────────────────────────────────────────────

    def _resolve_split_file(self, split: str) -> Optional[Path]:
        """Find the JSONL file corresponding to this split."""
        jsonl_files = sorted(self.data_dir.glob("*.jsonl"))
        if not jsonl_files:
            return None

        # Prefer files whose name contains the split name
        for f in jsonl_files:
            if split in f.stem.lower():
                return f

        # Fallback: use the only file, or the first one
        if len(jsonl_files) == 1:
            logger.warning(
                f"No file matching split='{split}' found. "
                f"Using {jsonl_files[0].name}."
            )
            return jsonl_files[0]

        # Multiple files, none matching — prefer 'test' > 'dev' > first
        priority = {"test": 0, "dev": 1, "train": 2}
        jsonl_files.sort(key=lambda f: priority.get(f.stem.lower(), 3))
        logger.warning(
            f"No file matching split='{split}'. Using {jsonl_files[0].name}."
        )
        return jsonl_files[0]

    def _build_examples(
        self, path: Path, split_name: str
    ) -> List[VerificationExample]:
        examples: List[VerificationExample] = []
        n_skipped = 0

        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    n_skipped += 1
                    continue

                # Auto-detect field names
                claim    = self._get_field(row, ["claim", "statement", "text"])
                evidence = self._get_field(row, ["evidence", "passage", "context", "document"])
                label_raw = self._get_field(row, ["label", "verdict", "entailment_label"])

                if claim is None or label_raw is None:
                    n_skipped += 1
                    continue

                try:
                    verdict = normalise_label(str(label_raw))
                except ValueError:
                    n_skipped += 1
                    continue

                evidence_text = str(evidence) if evidence else ""
                sentences     = _split_sentences(evidence_text)[: self.max_sentences]

                support_sids = row.get("supporting_sentences") or \
                               row.get("rationale_sentences") or \
                               row.get("evidence_sentences") or []
                # Flatten in case support_sids is a list of lists (WiCE groups rationales)
                rationale_positions = set(
                    sid if isinstance(sid, int) else item
                    for sid in support_sids
                    for item in (sid if isinstance(sid, list) else [sid])
                )

                if sentences:
                    document, char_spans, sent_texts = _build_document(
                        sentences, rationale_positions
                    )
                else:
                    document    = ""
                    char_spans  = []
                    sent_texts  = []

                examples.append(VerificationExample(
                    example_id=str(row.get("id", idx)),
                    claim=str(claim),
                    document=document,
                    verdict=verdict,
                    evidence_char_spans=char_spans,
                    evidence_sentence_texts=sent_texts,
                    source="wice",
                ))

        logger.info(
            f"WiCE '{split_name}': {len(examples):,} examples "
            f"({n_skipped} skipped — bad format or label)."
        )
        return examples

    @staticmethod
    def _get_field(row: dict, candidates: List[str]):
        """Return the first matching field value from a dict."""
        for key in candidates:
            if key in row:
                return row[key]
        return None


# ── Utilities ─────────────────────────────────────────────────────────────────

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


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
