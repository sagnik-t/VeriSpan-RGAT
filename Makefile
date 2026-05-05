# =============================================================================
# VeriSpan-RGAT — Project Makefile
# =============================================================================
# Usage:
#   make setup          — install deps + download data
#   make preprocess     — run entity span preprocessing
#   make train          — full training run (run1)
#   make smoke          — single forward-pass sanity check (CPU-safe)
#   make test           — run pytest suite
#   make lint           — ruff + black check
#   make fmt            — auto-format with black
# =============================================================================

.PHONY: help setup preprocess train smoke test lint fmt clean

# ── Env ───────────────────────────────────────────────────────────────────────

WANDB_MODE            ?= offline
PYTORCH_ALLOC_CONF    := expandable_segments:True

# ── Paths ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR            := checkpoints/run1
FEVER_DIR             := data/raw/fever
ENTITY_SPANS          := data/processed/fever_train_entities.json
PROCESSED_DIR         := data/processed

# ── Hyperparameters ───────────────────────────────────────────────────────────

NUM_EPOCHS            := 5
BATCH_SIZE            := 4
GRAD_ACCUM            := 16
MAX_LENGTH            := 256
LR                    := 2e-5
WANDB_RUN_NAME        := run1-full

# =============================================================================

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Override any var inline:  make train BATCH_SIZE=8 NUM_EPOCHS=3"

# ── Setup ─────────────────────────────────────────────────────────────────────

setup: ## Install dependencies and download raw data
	poetry install
	poetry run python scripts/setup_data.py

# ── Preprocessing ─────────────────────────────────────────────────────────────

preprocess: ## Run SpaCy entity span preprocessing for all splits
	poetry run python scripts/preprocess_entities.py --all

preprocess-fever: ## Preprocess FEVER train split only
	poetry run python scripts/preprocess_entities.py --dataset fever --split train

preprocess-resume: ## Preprocess, skipping splits that already exist
	poetry run python scripts/preprocess_entities.py --all --resume

# ── Training ──────────────────────────────────────────────────────────────────

train: ## Full training run (run1)
	WANDB_MODE=$(WANDB_MODE) \
	PYTORCH_CUDA_ALLOC_CONF=$(PYTORCH_ALLOC_CONF) \
	poetry run python scripts/train.py \
		--output_dir      $(OUTPUT_DIR) \
		--fever_dir       $(FEVER_DIR) \
		--entity_spans_path $(ENTITY_SPANS) \
		--num_epochs      $(NUM_EPOCHS) \
		--batch_size      $(BATCH_SIZE) \
		--grad_accum      $(GRAD_ACCUM) \
		--max_length      $(MAX_LENGTH) \
		--lr              $(LR) \
		--wandb_run_name  $(WANDB_RUN_NAME)

train-resume: ## Resume training from last checkpoint
	WANDB_MODE=$(WANDB_MODE) \
	PYTORCH_CUDA_ALLOC_CONF=$(PYTORCH_ALLOC_CONF) \
	poetry run python scripts/train.py \
		--output_dir      $(OUTPUT_DIR) \
		--fever_dir       $(FEVER_DIR) \
		--entity_spans_path $(ENTITY_SPANS) \
		--num_epochs      $(NUM_EPOCHS) \
		--batch_size      $(BATCH_SIZE) \
		--grad_accum      $(GRAD_ACCUM) \
		--max_length      $(MAX_LENGTH) \
		--lr              $(LR) \
		--wandb_run_name  $(WANDB_RUN_NAME) \
		--resume          $(OUTPUT_DIR)/last_model.pt

smoke: ## Single forward-pass sanity check — no GPU required
	poetry run python scripts/train.py --smoke_test --no_fp16

# ── Evaluation ────────────────────────────────────────────────────────────────

eval-scifact: ## Zero-shot eval on SciFact
	poetry run python scripts/evaluate.py \
		--checkpoint $(OUTPUT_DIR)/best_model.pt \
		--dataset    scifact

eval-wice: ## Zero-shot eval on WiCE
	poetry run python scripts/evaluate.py \
		--checkpoint $(OUTPUT_DIR)/best_model.pt \
		--dataset    wice

# ── Quality ───────────────────────────────────────────────────────────────────

test: ## Run pytest suite
	poetry run pytest -v

test-fast: ## Skip slow tests (no downloads, no SpaCy)
	poetry run pytest -v -m "not slow"

test-last-failed: ## Re-run only previously failing tests
	poetry run pytest -v --lf

lint: ## Ruff lint + black format check
	poetry run ruff check src/ scripts/ tests/
	poetry run black --check src/ scripts/ tests/

fmt: ## Auto-format with black
	poetry run black src/ scripts/ tests/

# ── Housekeeping ──────────────────────────────────────────────────────────────

clean: ## Remove checkpoints and processed data
	rm -rf checkpoints/ $(PROCESSED_DIR)/
