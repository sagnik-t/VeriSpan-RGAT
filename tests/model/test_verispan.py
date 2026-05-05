"""
tests/model/test_verispan.py

Unit and integration tests for VeriSpanModel.

Test inventory
--------------
    TestVeriSpanConfig
        test_defaults             — default field values match thesis spec
        test_custom_values        — arbitrary overrides round-trip correctly

    TestModelInstantiation
        test_builds_without_error — model constructs, no exceptions
        test_parameter_count      — ~170 M trainable params (sanity check)
        test_config_attached      — model.config is the passed VeriSpanConfig

    TestForwardPass               (CPU, batch of 4 synthetic examples)
        test_span_logits_shape    — [B, L]
        test_span_probs_shape     — [B, L]
        test_span_probs_range     — all values in (0, 1)
        test_verdict_logits_shape — [B, 3]
        test_z_evidence_feature_dim — last dim == hidden_channels
        test_evidence_batch_shape — [N_es] with correct max index

    TestLoss
        test_all_loss_terms_finite     — total, span, verdict, contrast all finite
        test_loss_output_as_dict       — keys and finite float values
        test_contrast_zero_all_nei     — contrast = 0 when no support/refute in batch
        test_contrast_zero_one_class   — contrast = 0 when only SUPPORTS present
        test_contrast_nonneg_mixed     — contrast ≥ 0 with SUPPORTS + REFUTES present

    TestBackwardPass
        test_gradients_reach_encoder   — grad is not None on embedding weight
        test_encoder_grads_finite      — no inf/nan in encoder gradients
        test_span_head_grads_exist     — grad flows into span head

Fixtures (inherited from tests/conftest.py)
-------------------------------------------
    tokenizer, collator, examples, encoded_examples
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# ── path setup ────────────────────────────────────────────────────────────────
# Needed when running individual test files directly; pytest handles this
# automatically when invoked from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factories import make_example, make_examples
from verispan.model.verispan import LossOutput, VeriSpanConfig, VeriSpanModel
from verispan.processing.dataset import ClaimVerificationDataset
from verispan.processing.collator import VerificationCollator


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_batch(tokenizer, collator, n: int = 4) -> dict:
    """Build a real padded batch from n synthetic examples."""
    examples = make_examples(n)
    encoded  = [tokenizer.encode(ex) for ex in examples]
    return collator(encoded)


def _model_and_batch(tokenizer, collator):
    """Return (model, batch) on CPU for use in forward-pass tests."""
    model = VeriSpanModel(VeriSpanConfig())
    model.eval()
    batch = _build_batch(tokenizer, collator, n=4)
    return model, batch


# ── Config ────────────────────────────────────────────────────────────────────

class TestVeriSpanConfig:

    def test_defaults(self):
        cfg = VeriSpanConfig()
        assert cfg.encoder_name    == "microsoft/deberta-v3-small"
        assert cfg.rgat_layers     == 2
        assert cfg.rgat_heads      == 4
        assert cfg.hidden_channels == 768
        assert cfg.num_classes     == 3
        assert cfg.lambda1         == 1.0
        assert cfg.lambda2         == 0.5
        assert cfg.contrast_margin == 1.0
        assert cfg.span_threshold  == 0.5

    def test_custom_values(self):
        cfg = VeriSpanConfig(rgat_layers=3, hidden_channels=256, lambda2=0.1)
        assert cfg.rgat_layers     == 3
        assert cfg.hidden_channels == 256
        assert cfg.lambda2         == 0.1
        # Unrelated defaults untouched
        assert cfg.num_classes     == 3


# ── Instantiation ─────────────────────────────────────────────────────────────

class TestModelInstantiation:

    def test_builds_without_error(self):
        model = VeriSpanModel(VeriSpanConfig())
        assert model is not None

    def test_parameter_count(self):
        """Model should have ~170 M trainable parameters (DeBERTa-v3-small base)."""
        model  = VeriSpanModel(VeriSpanConfig())
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Accept anything between 100 M and 250 M — catches gross structural errors
        assert 100_000_000 < n_params < 250_000_000, \
            f"Unexpected parameter count: {n_params:,}"

    def test_config_attached(self):
        cfg   = VeriSpanConfig(rgat_heads=2)
        model = VeriSpanModel(cfg)
        assert model.config is cfg


# ── Forward pass ──────────────────────────────────────────────────────────────

class TestForwardPass:

    @pytest.fixture(scope="class")
    def output_and_batch(self, tokenizer, collator):
        """Run one forward pass and cache the result for all tests in this class."""
        model, batch = _model_and_batch(tokenizer, collator)
        with torch.no_grad():
            out, loss = model.forward_and_loss(batch)
        return out, loss, batch

    def test_span_logits_shape(self, output_and_batch):
        out, _, batch = output_and_batch
        B, L = batch["input_ids"].shape
        assert out.span_logits.shape == (B, L), \
            f"Expected ({B}, {L}), got {out.span_logits.shape}"

    def test_span_probs_shape(self, output_and_batch):
        out, _, batch = output_and_batch
        B, L = batch["input_ids"].shape
        assert out.span_probs.shape == (B, L)

    def test_span_probs_range(self, output_and_batch):
        out, _, _ = output_and_batch
        assert out.span_probs.min().item() >= 0.0
        assert out.span_probs.max().item() <= 1.0

    def test_verdict_logits_shape(self, output_and_batch):
        out, _, batch = output_and_batch
        B = batch["input_ids"].shape[0]
        assert out.verdict_logits.shape == (B, 3), \
            f"Expected ({B}, 3), got {out.verdict_logits.shape}"

    def test_z_evidence_feature_dim(self, output_and_batch):
        """Post-GNN evidence span representations must have d == hidden_channels."""
        out, _, _ = output_and_batch
        assert out.z_evidence.shape[-1] == 768, \
            f"Expected feature dim 768, got {out.z_evidence.shape[-1]}"

    def test_evidence_batch_indices_valid(self, output_and_batch):
        """evidence_batch must be a 1-D integer tensor with valid example indices."""
        out, _, batch = output_and_batch
        B = batch["input_ids"].shape[0]
        assert out.evidence_batch.dim() == 1
        assert out.evidence_batch.max().item() < B


# ── Loss ──────────────────────────────────────────────────────────────────────

class TestLoss:

    @pytest.fixture(scope="class")
    def loss_out(self, tokenizer, collator):
        """Run forward_and_loss with a mixed-verdict batch (sup + ref + nei)."""
        model, batch = _model_and_batch(tokenizer, collator)
        with torch.no_grad():
            _, loss = model.forward_and_loss(batch)
        return loss

    def test_all_loss_terms_finite(self, loss_out):
        assert torch.isfinite(loss_out.total),    "total loss is not finite"
        assert torch.isfinite(loss_out.span),     "span loss is not finite"
        assert torch.isfinite(loss_out.verdict),  "verdict loss is not finite"
        assert torch.isfinite(loss_out.contrast), "contrast loss is not finite"

    def test_loss_output_as_dict(self, loss_out):
        d = loss_out.as_dict()
        assert set(d.keys()) == {"loss", "loss_span", "loss_verdict", "loss_contrast"}
        for v in d.values():
            assert isinstance(v, float)
            assert torch.isfinite(torch.tensor(v))

    def test_contrast_zero_all_nei(self, tokenizer, collator):
        """Contrast loss must be 0 when the batch contains no SUPPORTS or REFUTES."""
        examples = [
            make_example(verdict=2, example_id=f"nei-{i}", document="", char_spans=[])
            for i in range(4)
        ]
        encoded = [tokenizer.encode(ex) for ex in examples]
        batch   = collator(encoded)

        model = VeriSpanModel(VeriSpanConfig())
        model.eval()
        with torch.no_grad():
            _, loss = model.forward_and_loss(batch)

        assert loss.contrast.item() == pytest.approx(0.0, abs=1e-6)

    def test_contrast_zero_only_supports(self, tokenizer, collator):
        """Contrast loss must be 0 when only one verdict class (SUPPORTS) is present."""
        examples = [make_example(verdict=0, example_id=f"sup-{i}") for i in range(4)]
        encoded  = [tokenizer.encode(ex) for ex in examples]
        batch    = collator(encoded)

        model = VeriSpanModel(VeriSpanConfig())
        model.eval()
        with torch.no_grad():
            _, loss = model.forward_and_loss(batch)

        assert loss.contrast.item() == pytest.approx(0.0, abs=1e-6)

    def test_contrast_nonneg_mixed_verdict(self, tokenizer, collator):
        """
        With at least one SUPPORTS and one REFUTES example, the contrast loss
        is computed (≥ 0).  We can't assert it's > 0 for a random-weight model
        (embeddings may already be separated), so we just check it's non-negative
        and finite.
        """
        examples = [
            make_example(verdict=0, example_id="sup-0"),
            make_example(verdict=1, example_id="ref-0"),
            make_example(verdict=0, example_id="sup-1"),
            make_example(verdict=1, example_id="ref-1"),
        ]
        encoded = [tokenizer.encode(ex) for ex in examples]
        batch   = collator(encoded)

        model = VeriSpanModel(VeriSpanConfig())
        model.eval()
        with torch.no_grad():
            _, loss = model.forward_and_loss(batch)

        assert torch.isfinite(loss.contrast)
        assert loss.contrast.item() >= 0.0


# ── Backward pass ─────────────────────────────────────────────────────────────

class TestBackwardPass:
    """
    These tests run a full backward pass.  They are intentionally NOT inside
    torch.no_grad() blocks.  Scope='function' so each test gets a fresh model
    and clean gradient state.
    """

    @pytest.fixture
    def model_and_batch(self, tokenizer, collator):
        model = VeriSpanModel(VeriSpanConfig())
        model.train()
        batch = _build_batch(tokenizer, collator)
        return model, batch

    def test_gradients_reach_encoder(self, model_and_batch):
        model, batch = model_and_batch
        _, loss = model.forward_and_loss(batch)
        loss.total.backward()

        emb_weight = model.encoder.model.embeddings.word_embeddings.weight
        assert emb_weight.grad is not None, \
            "No gradient reached the encoder word embeddings"

    def test_encoder_grads_finite(self, model_and_batch):
        model, batch = model_and_batch
        _, loss = model.forward_and_loss(batch)
        loss.total.backward()

        emb_grad = model.encoder.model.embeddings.word_embeddings.weight.grad
        assert emb_grad.isfinite().all(), \
            "Encoder embedding gradients contain inf or nan"

    def test_span_head_grads_exist(self, model_and_batch):
        model, batch = model_and_batch
        _, loss = model.forward_and_loss(batch)
        loss.total.backward()

        # SpanExtractionHead has a linear layer — check its weight gradient
        for name, param in model.span_head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"No gradient for span_head.{name}"
                break
