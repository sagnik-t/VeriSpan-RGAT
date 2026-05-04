"""
entity.py — SpaCy NER preprocessing pipeline (verispan.processing).

Extracts named entity mention token spans from claims and documents,
aligned to DeBERTa subword token positions.  These become the entity
mention nodes (NT_ENTITY) in the heterogeneous graph.

Design decisions
----------------
- Runs OFFLINE before training (not in the DataLoader hot path).
- Results are saved as a JSON file keyed by example_id.
- ClaimVerificationDataset loads this file at construction time and passes
  entity spans through to GraphBuilder via the collated batch.
- SpaCy is only imported in this file — no other module depends on it.

Output format (per example_id)
-------------------------------
    {
        "claim_entity_spans":    [[tok_start, tok_end], ...],
        "doc_entity_spans":      [[tok_start, tok_end], ...]
    }
    Spans are token-level (DeBERTa subword indices), inclusive.

Usage
-----
    # From scripts/preprocess_entities.py:
    from verispan.processing.entity import EntityPreprocessor

    proc = EntityPreprocessor()
    proc.process_and_save(examples, tokenizer, output_path="data/processed/fever_entities.json")

    # From ClaimVerificationDataset:
    from verispan.processing.entity import load_entity_spans
    entity_spans = load_entity_spans("data/processed/fever_entities.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..data.schema import VerificationExample
from .tokenization import VerificationTokenizer

logger = logging.getLogger(__name__)

# Type alias: maps example_id → {"claim_entity_spans": [...], "doc_entity_spans": [...]}
EntitySpanMap = Dict[str, Dict[str, List[List[int]]]]


class EntityPreprocessor:
    """
    Runs SpaCy NER over claims and documents and aligns entity spans to
    DeBERTa subword token positions.

    Parameters
    ----------
    spacy_model : str
        SpaCy model name.  'en_core_web_sm' is fast and sufficient for
        general NER (persons, organisations, locations, dates).
        For SciFact, consider 'en_core_sci_sm' from scispaCy.
    batch_size : int
        Number of texts per SpaCy pipe() call.  Larger = faster but
        more RAM.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        batch_size: int = 64,
    ) -> None:
        try:
            import spacy
            self.nlp = spacy.load(spacy_model, disable=["parser", "tagger", "lemmatizer"])
        except OSError:
            raise OSError(
                f"SpaCy model '{spacy_model}' not found. "
                f"Install it with:\n"
                f"    python -m spacy download {spacy_model}"
            )
        self.batch_size = batch_size

    # ── public ───────────────────────────────────────────────────────────────

    def process_and_save(
        self,
        examples: List[VerificationExample],
        tokenizer: VerificationTokenizer,
        output_path: str,
    ) -> EntitySpanMap:
        """
        Process all examples and save entity spans to a JSON file.

        Parameters
        ----------
        examples : List[VerificationExample]
        tokenizer : VerificationTokenizer
            Used to re-tokenize claim and document individually so we can
            align character-level SpaCy spans to subword token indices.
        output_path : str
            Where to write the JSON file.

        Returns
        -------
        EntitySpanMap — also written to output_path.
        """
        logger.info(
            f"Extracting entity spans for {len(examples):,} examples "
            f"using SpaCy model '{self.nlp.meta['name']}' ..."
        )

        span_map: EntitySpanMap = {}

        # Collect all texts for batch NER
        claim_texts = [ex.claim for ex in examples]
        doc_texts   = [ex.document if ex.document else "" for ex in examples]

        logger.info("Running SpaCy NER on claims ...")
        claim_ent_chars = self._batch_ner(claim_texts)

        logger.info("Running SpaCy NER on documents ...")
        doc_ent_chars   = self._batch_ner(doc_texts)

        logger.info("Aligning entity spans to subword token positions ...")
        for i, ex in enumerate(examples):
            claim_tok_spans = self._align_to_tokens(
                claim_ent_chars[i], ex.claim, tokenizer, is_claim=True
            )
            doc_tok_spans = self._align_to_tokens(
                doc_ent_chars[i], ex.document or "", tokenizer, is_claim=False
            )
            span_map[ex.example_id] = {
                "claim_entity_spans": [list(s) for s in claim_tok_spans],
                "doc_entity_spans":   [list(s) for s in doc_tok_spans],
            }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(span_map, f)

        logger.info(f"Entity spans saved → {output_path}")
        return span_map

    # ── internals ────────────────────────────────────────────────────────────

    def _batch_ner(
        self, texts: List[str]
    ) -> List[List[Tuple[int, int]]]:
        """
        Run SpaCy NER in batches.

        Returns a list (one per text) of (char_start, char_end) tuples
        for each named entity found.  char_end is exclusive.
        """
        results: List[List[Tuple[int, int]]] = []
        # Replace empty strings with a single space so SpaCy doesn't fail
        safe_texts = [t if t.strip() else " " for t in texts]

        for doc in self.nlp.pipe(safe_texts, batch_size=self.batch_size):
            results.append([(ent.start_char, ent.end_char) for ent in doc.ents])

        return results

    def _align_to_tokens(
        self,
        char_spans:  List[Tuple[int, int]],
        text:        str,
        tokenizer:   VerificationTokenizer,
        is_claim:    bool,
    ) -> List[Tuple[int, int]]:
        """
        Convert character-level entity spans to DeBERTa subword token indices.

        Tokenizes the text in isolation (not as part of the full
        claim+document pair) to get a clean offset mapping, then maps
        entity character offsets to token indices.

        Returns list of (tok_start, tok_end) inclusive pairs.
        """
        if not text.strip() or not char_spans:
            return []

        # Tokenize in isolation to get offset_mapping
        enc = tokenizer.tokenizer(
            text,
            max_length=tokenizer.max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        offsets: List[Tuple[int, int]] = enc["offset_mapping"]
        seq_ids = enc.sequence_ids() if hasattr(enc, "sequence_ids") else \
                  [None] + [0] * (len(offsets) - 2) + [None]

        tok_spans: List[Tuple[int, int]] = []
        for ent_start, ent_end in char_spans:
            first = last = None
            for tok_idx, (tok_s, tok_e) in enumerate(offsets):
                if seq_ids[tok_idx] is None:
                    continue
                if tok_s < ent_end and tok_e > ent_start:
                    if first is None:
                        first = tok_idx
                    last = tok_idx
            if first is not None and last is not None:
                tok_spans.append((first, last))

        return tok_spans


# ── Loading helper ────────────────────────────────────────────────────────────

def load_entity_spans(path: str) -> EntitySpanMap:
    """
    Load a previously computed entity span map from disk.

    Parameters
    ----------
    path : str
        Path to the JSON file produced by EntityPreprocessor.process_and_save().

    Returns
    -------
    EntitySpanMap: dict mapping example_id → span dict.
    """
    with open(path, "r") as f:
        raw: dict = json.load(f)
    # Convert lists back to tuples for consistency with the rest of the code
    result: EntitySpanMap = {}
    for ex_id, spans in raw.items():
        result[ex_id] = {
            "claim_entity_spans": [tuple(s) for s in spans.get("claim_entity_spans", [])],
            "doc_entity_spans":   [tuple(s) for s in spans.get("doc_entity_spans", [])],
        }
    return result
