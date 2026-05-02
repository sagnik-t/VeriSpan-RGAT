"""
encoder.py — Contextualized encoding stage (Stage 1).

Wraps DeBERTa-v3-small and returns token-level embeddings
H ∈ R^(B × L × d) for consumption by all downstream stages.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class DeBERTaEncoder(nn.Module):
    """
    DeBERTa-v3-small encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    freeze_layers : int
        Number of bottom transformer layers to freeze (parameter-efficient
        fine-tuning).  0 = train all layers.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        freeze_layers: int = 0,
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.hidden_size: int = self.model.config.hidden_size  # 768

        if freeze_layers > 0:
            self._freeze_bottom_layers(freeze_layers)

    # ── public ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """
        Encode the concatenated [CLS] claim [SEP] document [SEP] sequence.

        Returns
        -------
        H : Tensor [B, L, d]
            Token-level contextual embeddings.
        """
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # [B, L, d]

    # ── internals ────────────────────────────────────────────────────────────

    def _freeze_bottom_layers(self, n: int) -> None:
        """Freeze the bottom n transformer layers (keeps embeddings trainable)."""
        encoder_layers = self.model.encoder.layer
        for i, layer in enumerate(encoder_layers):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False
