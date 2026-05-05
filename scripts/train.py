"""
scripts/train.py — Training entry point for VeriSpan-RGAT.

Usage
-----
    # Minimal — all defaults, single batch smoke test:
    python scripts/train.py --smoke_test

    # Full training run:
    python scripts/train.py \
        --output_dir checkpoints/run1 \
        --num_epochs 5 \
        --batch_size 16 \
        --grad_accum 4 \
        --lr 2e-5 \
        --cache_dir data/cache \
        --entity_spans_path data/processed/fever_entities.json

    # Resume from checkpoint:
    python scripts/train.py \
        --output_dir checkpoints/run1 \
        --resume checkpoints/run1/last_model.pt

Smoke test
----------
    --smoke_test loads 32 examples from FEVER dev (not train — faster),
    runs a single forward pass on CPU, prints tensor shapes and loss values,
    then exits. Use this to verify the full pipeline before committing to
    a real training run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── make sure src/ is on the path when running as a script ───────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader, Subset

from verispan.data.fever import FEVERProcessor
from verispan.model.verispan import VeriSpanConfig, VeriSpanModel
from verispan.processing.collator import VerificationCollator
from verispan.processing.dataset import ClaimVerificationDataset
from verispan.processing.tokenization import VerificationTokenizer
from verispan.training.trainer import Trainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VeriSpan-RGAT")

    # Paths
    p.add_argument("--output_dir",         default="checkpoints/run1")
    p.add_argument("--fever_dir",          default="data/raw/fever",
                   help="Path to local FEVER data directory (from setup_data.py).")
    p.add_argument("--entity_spans_path",  default=None,
                   help="Path to pre-computed entity spans JSON. "
                        "Omit to train without entity mention nodes.")

    # Training schedule
    p.add_argument("--num_epochs",  type=int,   default=5)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--warmup_steps",type=int,   default=500)
    p.add_argument("--max_length",  type=int,   default=512)
    p.add_argument("--max_doc_sents",type=int,  default=5)

    # Model
    p.add_argument("--encoder",       default="microsoft/deberta-v3-small")
    p.add_argument("--freeze_layers", type=int, default=0)
    p.add_argument("--rgat_layers",   type=int, default=2)
    p.add_argument("--rgat_heads",    type=int, default=4)
    p.add_argument("--hidden_channels", type=int, default=768)

    # Loss weights
    p.add_argument("--lambda1",          type=float, default=1.0)
    p.add_argument("--lambda2",          type=float, default=0.5)
    p.add_argument("--contrast_margin",  type=float, default=1.0)

    # W&B
    p.add_argument("--wandb_project",   default="verispan-rgat")
    p.add_argument("--wandb_run_name",  default=None)
    p.add_argument("--wandb_tags",      nargs="*", default=[])

    # Misc
    p.add_argument("--no_fp16",     action="store_true",
                   help="Disable fp16 mixed precision (use for CPU runs).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader worker processes. "
                        "0 = main process only (safer for debugging).")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint to resume training from.")
    p.add_argument("--smoke_test",  action="store_true",
                   help="Run a single forward pass and exit. "
                        "Does not train. Does not log to W&B.")

    return p.parse_args()


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_datasets(args) -> tuple[ClaimVerificationDataset, ClaimVerificationDataset]:
    entity_path = args.entity_spans_path

    logger.info("Building training dataset ...")
    train_ds = ClaimVerificationDataset.from_fever(
        split="train",
        model_name=args.encoder,
        max_length=args.max_length,
        max_doc_sentences=args.max_doc_sents,
        data_dir=args.fever_dir,
        entity_span_path=entity_path,
        precompute=False,   # FEVER train is large — lazy tokenization
    )

    logger.info("Building dev dataset ...")
    dev_ds = ClaimVerificationDataset.from_fever(
        split="dev",
        model_name=args.encoder,
        max_length=args.max_length,
        max_doc_sentences=args.max_doc_sents,
        data_dir=args.fever_dir,
        entity_span_path=entity_path,
        precompute=True,
    )

    logger.info(f"Train: {train_ds}")
    logger.info(f"Dev  : {dev_ds}")
    return train_ds, dev_ds


def build_dataloaders(
    train_ds: ClaimVerificationDataset,
    dev_ds:   ClaimVerificationDataset,
    args,
) -> tuple[DataLoader, DataLoader]:
    collator = VerificationCollator(
        pad_token_id=train_ds.tokenizer.pad_token_id
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size * 2,  # no grad → can double batch size
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, dev_loader


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(args) -> VeriSpanModel:
    config = VeriSpanConfig(
        encoder_name=args.encoder,
        freeze_layers=args.freeze_layers,
        rgat_layers=args.rgat_layers,
        rgat_heads=args.rgat_heads,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        contrast_margin=args.contrast_margin,
    )
    model = VeriSpanModel(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model built — {n_params:,} trainable parameters")
    return model


# ── Smoke test ────────────────────────────────────────────────────────────────

def run_smoke_test(args) -> None:
    """
    Single forward pass on a small synthetic batch.

    Uses VerificationExample objects constructed in-memory — no dataset
    download required.  This verifies the full model pipeline independently
    of any data access issues.

    Checks:
        1. All imports resolve correctly
        2. forward_and_loss() runs without crashing
        3. Output tensor shapes match expectations
        4. All loss terms are finite
        5. Gradients reach the encoder embeddings
    """
    logger.info("=" * 60)
    logger.info("SMOKE TEST — single forward pass on CPU (synthetic data)")
    logger.info("=" * 60)

    device = torch.device("cpu")

    # ── Synthetic examples — no data download required ────────────────────
    from verispan.data.schema import VerificationExample

    examples = [
        VerificationExample(
            example_id=f"smoke-{i}",
            claim="The Eiffel Tower is located in Berlin.",
            document="The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
            verdict=verdict,
            evidence_char_spans=[(49, 62)] if verdict != 2 else [],
            source="smoke_test",
        )
        for i, verdict in enumerate([0, 1, 2, 0])  # SUP, REF, NEI, SUP
    ]

    tokenizer = VerificationTokenizer(model_name=args.encoder, max_length=128)
    ds        = ClaimVerificationDataset(examples, tokenizer, precompute=True)
    collator  = VerificationCollator(pad_token_id=tokenizer.pad_token_id)
    loader    = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collator)

    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    logger.info(f"Batch input_ids shape  : {batch['input_ids'].shape}")
    logger.info(f"Batch span_labels shape: {batch['span_labels'].shape}")
    logger.info(f"Verdict labels         : {batch['verdict_labels'].tolist()}")

    # ── Build model ────────────────────────────────────────────────────────
    logger.info("Building model ...")
    model = build_model(args).to(device).float()
    model.train()

    # ── Forward + loss ─────────────────────────────────────────────────────
    logger.info("Running forward pass ...")
    output, loss_out = model.forward_and_loss(batch)

    # ── Shape assertions ───────────────────────────────────────────────────
    B, L = batch["input_ids"].shape
    assert output.span_logits.shape    == (B, L), \
        f"span_logits shape mismatch: {output.span_logits.shape}"
    assert output.span_probs.shape     == (B, L), \
        f"span_probs shape mismatch: {output.span_probs.shape}"
    assert output.verdict_logits.shape == (B, 3), \
        f"verdict_logits shape mismatch: {output.verdict_logits.shape}"

    # ── Loss assertions ────────────────────────────────────────────────────
    assert torch.isfinite(loss_out.total),    "Total loss is not finite"
    assert torch.isfinite(loss_out.span),     "Span loss is not finite"
    assert torch.isfinite(loss_out.verdict),  "Verdict loss is not finite"
    assert torch.isfinite(loss_out.contrast), "Contrast loss is not finite"

    # ── Backward pass ──────────────────────────────────────────────────────
    logger.info("Running backward pass ...")
    loss_out.total.backward()

    enc_grad = model.encoder.model.embeddings.word_embeddings.weight.grad
    assert enc_grad is not None, "No gradient reached the encoder embeddings"
    assert torch.isfinite(enc_grad).all(), "Encoder gradients contain inf/nan"

    logger.info("=" * 60)
    logger.info("SMOKE TEST PASSED ✓")
    logger.info(f"  span_logits    : {output.span_logits.shape}")
    logger.info(f"  verdict_logits : {output.verdict_logits.shape}")
    logger.info(f"  z_evidence     : {output.z_evidence.shape}")
    logger.info(f"  loss_total     : {loss_out.total.item():.4f}")
    logger.info(f"  loss_span      : {loss_out.span.item():.4f}")
    logger.info(f"  loss_verdict   : {loss_out.verdict.item():.4f}")
    logger.info(f"  loss_contrast  : {loss_out.contrast.item():.4f}")
    logger.info("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.smoke_test:
        run_smoke_test(args)
        return

    # ── Full training run ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_ds, dev_ds     = build_datasets(args)
    train_loader, dev_loader = build_dataloaders(train_ds, dev_ds, args)
    model                = build_model(args)

    training_config = TrainingConfig(
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags or [],
        num_epochs=args.num_epochs,
        per_device_batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        fp16=not args.no_fp16 and torch.cuda.is_available(),
        log_every_n_steps=50,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        config=training_config,
        device=device,
    )

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
