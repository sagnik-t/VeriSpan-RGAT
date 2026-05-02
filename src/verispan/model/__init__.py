"""
verispan.model — Model architecture for VeriSpan-RGAT.

Public API
----------
    from verispan.model import (
        VeriSpanModel,
        VeriSpanConfig,
        VeriSpanOutput,
        LossOutput,
    )
"""

from .verispan import VeriSpanModel, VeriSpanConfig, VeriSpanOutput, LossOutput
from .encoder import DeBERTaEncoder
from .span_head import SpanExtractionHead, extract_candidate_spans
from .graph import GraphBuilder, GRAPH_METADATA
from .rgat import HeteroRGAT, HeteroRGATLayer, RGATConv, SingleHeadRGATConv
from .verdict_head import VerdictHead

__all__ = [
    # Top-level
    "VeriSpanModel",
    "VeriSpanConfig",
    "VeriSpanOutput",
    "LossOutput",
    # Stages (for unit testing / ablations)
    "DeBERTaEncoder",
    "SpanExtractionHead",
    "extract_candidate_spans",
    "GraphBuilder",
    "GRAPH_METADATA",
    "HeteroRGAT",
    "HeteroRGATLayer",
    "RGATConv",
    "SingleHeadRGATConv",
    "VerdictHead",
]
