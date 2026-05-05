"""
scripts/preprocess_entities.py — Offline SpaCy entity span preprocessing.

Runs SpaCy NER over every (claim, document) pair in a dataset split,
aligns entity character offsets to DeBERTa subword token positions, and
writes a JSON file that ClaimVerificationDataset can load via --entity_spans_path.

This script should be run ONCE before training, not during the training loop.

Output files
------------
    data/processed/fever_train_entities.json
    data/processed/fever_dev_entities.json
    data/processed/scifact_train_entities.json
    data/processed/scifact_dev_entities.json
    data/processed/wice_train_entities.json
    data/processed/wice_dev_entities.json
    data/processed/wice_test_entities.json

Usage
-----
    # Process a single split:
    python scripts/preprocess_entities.py --dataset fever --split train

    # Process all splits in sequence:
    python scripts/preprocess_entities.py --all

    # Use scispaCy for biomedical NER on SciFact (better entity coverage):
    python scripts/preprocess_entities.py --dataset scifact --split dev \\
        --spacy_model en_core_sci_sm

    # Dry-run — print what would be processed without doing anything:
    python scripts/preprocess_entities.py --all --dry_run

    # Resume — skip outputs that already exist on disk:
    python scripts/preprocess_entities.py --all --resume

    # Limit examples (useful for quick sanity checks):
    python scripts/preprocess_entities.py --dataset fever --split train \\
        --max_examples 1000

Notes
-----
- FEVER train is ~145k examples.  On CPU with en_core_web_sm this takes
  roughly 15-25 minutes depending on hardware.  The wiki DB load (~60s)
  is the main startup cost.
- SciFact and WiCE are small (< 5k examples each) and complete in < 2 min.
- Use --batch_size 128 or higher on machines with plenty of RAM to speed
  up the SpaCy pipe() calls.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from verispan.data.schema import VerificationExample
from verispan.processing.entity import EntityPreprocessor
from verispan.processing.tokenization import VerificationTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocess_entities")


# ── Job descriptor ────────────────────────────────────────────────────────────

@dataclass
class Job:
    """One (dataset, split) pair to process."""
    dataset:     str               # "fever" | "scifact" | "wice"
    split:       str               # "train" | "dev" | "test" | "shared_task_dev"
    output_path: Path
    spacy_model: str = "en_core_web_sm"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_fever(split: str, data_dir: str, max_examples: Optional[int]) -> List[VerificationExample]:
    from verispan.data.fever import FEVERProcessor
    proc = FEVERProcessor(data_dir=data_dir)
    examples = proc.load(split)
    if max_examples:
        examples = examples[:max_examples]
    return examples


def load_scifact(split: str, data_dir: str, max_examples: Optional[int]) -> List[VerificationExample]:
    from verispan.data.scifact import SciFatProcessor
    proc = SciFatProcessor(data_dir=data_dir)
    examples = proc.load(split)
    if max_examples:
        examples = examples[:max_examples]
    return examples


def load_wice(split: str, data_dir: str, max_examples: Optional[int]) -> List[VerificationExample]:
    from verispan.data.wice import WiCEProcessor
    proc = WiCEProcessor(data_dir=data_dir)
    examples = proc.load(split)
    if max_examples:
        examples = examples[:max_examples]
    return examples


LOADERS = {
    "fever":   load_fever,
    "scifact": load_scifact,
    "wice":    load_wice,
}

DATA_DIRS = {
    "fever":   "data/raw/fever",
    "scifact": "data/raw/scifact",
    "wice":    "data/raw/wice",
}

# All (dataset, split) pairs processed by --all
ALL_JOBS = [
    ("fever",   "train"),
    ("fever",   "dev"),
    ("scifact", "train"),
    ("scifact", "dev"),
    ("wice",    "train"),
    ("wice",    "dev"),
    ("wice",    "test"),
]


# ── Core processing ───────────────────────────────────────────────────────────

def run_job(
    job:          Job,
    args:         argparse.Namespace,
    tokenizer:    VerificationTokenizer,
    preprocessor: EntityPreprocessor,
) -> None:
    """Load examples for one job and run entity preprocessing."""
    if args.resume and job.output_path.exists():
        logger.info(
            f"[{job.dataset}/{job.split}] SKIPPED — output already exists: {job.output_path}"
        )
        return

    logger.info(f"[{job.dataset}/{job.split}] Loading examples ...")
    t0 = time.time()

    data_dir = getattr(args, f"{job.dataset}_data_dir", DATA_DIRS[job.dataset])
    loader   = LOADERS[job.dataset]
    examples = loader(job.split, data_dir, args.max_examples)

    if not examples:
        logger.warning(f"[{job.dataset}/{job.split}] No examples loaded — skipping.")
        return

    logger.info(f"[{job.dataset}/{job.split}] {len(examples):,} examples loaded in {time.time()-t0:.1f}s")
    logger.info(f"[{job.dataset}/{job.split}] Running entity preprocessing → {job.output_path}")

    preprocessor.process_and_save(
        examples=examples,
        tokenizer=tokenizer,
        output_path=str(job.output_path),
    )

    elapsed = time.time() - t0
    logger.info(f"[{job.dataset}/{job.split}] Done in {elapsed:.1f}s")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline SpaCy entity span preprocessing for VeriSpan-RGAT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Target selection
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all (dataset, split) pairs.",
    )
    group.add_argument(
        "--dataset",
        choices=["fever", "scifact", "wice"],
        help="Dataset to process (requires --split).",
    )
    p.add_argument(
        "--split",
        default=None,
        help="Split to process (train | dev | test | shared_task_dev). Required with --dataset.",
    )

    # Data paths
    p.add_argument("--fever_data_dir",   default="data/raw/fever",   metavar="DIR")
    p.add_argument("--scifact_data_dir", default="data/raw/scifact", metavar="DIR")
    p.add_argument("--wice_data_dir",    default="data/raw/wice",    metavar="DIR")
    p.add_argument("--output_dir",       default="data/processed",   metavar="DIR",
                   help="Directory to write entity span JSON files.")

    # SpaCy settings
    p.add_argument(
        "--spacy_model",
        default="en_core_web_sm",
        help="SpaCy model name.  Use 'en_core_sci_sm' for biomedical NER on SciFact.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="SpaCy pipe() batch size.  Increase for faster processing on large RAM machines.",
    )

    # Tokenizer settings
    p.add_argument(
        "--model_name",
        default="microsoft/deberta-v3-small",
        help="HuggingFace model name for span alignment.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=512,
    )

    # Execution flags
    p.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Truncate each split to this many examples.  Useful for quick sanity checks.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip output files that already exist on disk.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be processed without doing anything.",
    )

    args = p.parse_args()

    if args.dataset and not args.split:
        p.error("--split is required when --dataset is specified.")

    return args


def build_jobs(args: argparse.Namespace) -> List[Job]:
    output_dir = Path(args.output_dir)

    if args.all:
        pairs = ALL_JOBS
    else:
        pairs = [(args.dataset, args.split)]

    jobs: List[Job] = []
    for dataset, split in pairs:
        output_path = output_dir / f"{dataset}_{split}_entities.json"
        jobs.append(Job(
            dataset=dataset,
            split=split,
            output_path=output_path,
            spacy_model=args.spacy_model,
        ))
    return jobs


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("DRY RUN — no files will be written.")
        for job in jobs:
            exists = "EXISTS" if job.output_path.exists() else "missing"
            skip   = " (would skip)" if args.resume and job.output_path.exists() else ""
            logger.info(f"  {job.dataset:8s} / {job.split:16s} → {job.output_path}  [{exists}]{skip}")
        return

    # ── Build shared tokenizer and preprocessor ───────────────────────────────
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = VerificationTokenizer(
        model_name=args.model_name,
        max_length=args.max_length,
    )

    logger.info(f"Loading SpaCy model: {args.spacy_model}")
    preprocessor = EntityPreprocessor(
        spacy_model=args.spacy_model,
        batch_size=args.batch_size,
    )

    # ── Run jobs ──────────────────────────────────────────────────────────────
    total = len(jobs)
    t_total = time.time()

    for i, job in enumerate(jobs, 1):
        logger.info(f"── Job {i}/{total}: {job.dataset}/{job.split} {'─'*40}")
        run_job(job, args, tokenizer, preprocessor)

    elapsed = time.time() - t_total
    logger.info(f"{'='*60}")
    logger.info(f"All {total} job(s) complete in {elapsed:.1f}s")
    logger.info(f"Output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
