"""
rgat.py — Relational Graph Attention Network (Stage 4).

Class hierarchy
---------------
    SingleHeadRGATConv(MessagePassing)
        One relation type, one attention head.
        Implements the thesis attention formula with edge-weight modulation.

    RGATConv(nn.Module)
        One relation type, K parallel heads → concatenated output.

    HeteroRGATLayer(nn.Module)
        One RGAT layer over the full heterogeneous graph.
        Uses PyG's HeteroConv to dispatch per-relation convolutions and
        sum messages at shared target node types.

    HeteroRGAT(nn.Module)
        L stacked HeteroRGATLayers.  Main entry point for the model.

Attention formula (per relation r, head k, layer l)
----------------------------------------------------
    e_ij  = LeakyReLU( a_r^k · [W_r^{Q,k} h_i ∥ W_r^{K,k} h_j] )
    α_ij  = softmax_{j ∈ N_r(i)}( edge_weight_ij · e_ij )
    h_i'  = Σ_r Σ_j α_ij · W_r^{V,k} h_j        (per head k)
    h_i^{l+1} = ∥_{k=1}^K ELU(h_i'^k) + h_i^l   (residual, then LayerNorm)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.utils import softmax as pyg_softmax


# ── SingleHeadRGATConv ────────────────────────────────────────────────────────

class SingleHeadRGATConv(MessagePassing):
    """
    Single attention head for one relation type.

    Supports both homogeneous (x: Tensor) and bipartite
    (x: Tuple[Tensor, Tensor]) input, so it works for edges between
    nodes of the same type *and* edges between different node types
    (e.g. claim_span → evidence_span).
    """

    def __init__(self, in_channels: int, d_head: int) -> None:
        # aggr='add' sums weighted value vectors; pyg_softmax normalises α.
        # node_dim=0 so PyG indexes along the node dimension correctly.
        super().__init__(aggr="add", node_dim=0)
        self.d_head = d_head

        self.W_Q = nn.Linear(in_channels, d_head, bias=False)
        self.W_K = nn.Linear(in_channels, d_head, bias=False)
        self.W_V = nn.Linear(in_channels, d_head, bias=False)

        # Learnable attention vector: scalar attention score per edge
        self.a = nn.Parameter(torch.empty(2 * d_head))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.a.view(1, -1))

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,                       # [2, E]  row=src, col=dst
        edge_weight: Optional[Tensor] = None,     # [E]
    ) -> Tensor:                                  # [N_dst, d_head]

        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        N_src = x_src.size(0)
        N_dst = x_dst.size(0)
        row, col = edge_index   # row = source index, col = target index

        # Project all nodes
        Q = self.W_Q(x_dst)   # [N_dst, d_head]  — query: from target
        K = self.W_K(x_src)   # [N_src, d_head]  — key:   from source
        V = self.W_V(x_src)   # [N_src, d_head]  — value: from source

        # Attention score for each edge: a^T [ Q[dst] ∥ K[src] ]
        cat = torch.cat([Q[col], K[row]], dim=-1)          # [E, 2*d_head]
        alpha = F.leaky_relu((self.a * cat).sum(-1), negative_slope=0.2)  # [E]

        # Modulate by structural edge weight (e.g. Jaccard, cosine similarity)
        if edge_weight is not None:
            alpha = alpha * edge_weight

        # Softmax over the target node's incoming edges
        alpha = pyg_softmax(alpha, col, num_nodes=N_dst)   # [E]

        return self.propagate(
            edge_index,
            V=V,
            alpha=alpha,
            size=(N_src, N_dst),
        )   # [N_dst, d_head]

    def message(self, V_j: Tensor, alpha: Tensor) -> Tensor:
        """Weighted value vector for each (source → target) edge."""
        return alpha.unsqueeze(-1) * V_j   # [E, d_head]


# ── RGATConv ──────────────────────────────────────────────────────────────────

class RGATConv(nn.Module):
    """
    K-head RGAT convolution for a single relation type.

    Runs K independent SingleHeadRGATConv heads and concatenates their outputs
    so the output dimension equals `out_channels = K × d_head`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
    ) -> None:
        super().__init__()
        assert out_channels % heads == 0, (
            f"out_channels ({out_channels}) must be divisible by heads ({heads})"
        )
        d_head = out_channels // heads
        self.heads = nn.ModuleList(
            [SingleHeadRGATConv(in_channels, d_head) for _ in range(heads)]
        )

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:   # [N_dst, out_channels]
        return torch.cat(
            [h(x, edge_index, edge_weight) for h in self.heads], dim=-1
        )


# ── HeteroRGATLayer ───────────────────────────────────────────────────────────

class HeteroRGATLayer(nn.Module):
    """
    One layer of heterogeneous RGAT over the full typed graph.

    Uses PyG's HeteroConv to:
      1. Apply the appropriate RGATConv for each relation type.
      2. Sum contributions at each target node from all incoming relation types.

    Then applies: ELU → residual add (if dims match) → LayerNorm.

    Parameters
    ----------
    in_channels, out_channels : int
        We keep these equal (hidden_channels throughout) so residuals work.
    heads : int
        Number of attention heads in each RGATConv.
    metadata : (node_types, edge_types)
        Defines the complete graph schema (all possible edge types).
        Edge types absent from a particular batch are silently ignored.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata

        # One RGATConv per edge type in the schema
        self.conv = HeteroConv(
            {et: RGATConv(in_channels, out_channels, heads=heads) for et in edge_types},
            aggr="sum",
        )

        # Per-node-type layer normalisation
        self.norms = nn.ModuleDict(
            {nt: nn.LayerNorm(out_channels) for nt in node_types}
        )

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple, Tensor],
        edge_weight_dict: Optional[Dict[Tuple, Tensor]] = None,
    ) -> Dict[str, Tensor]:

        # HeteroConv dispatches per-relation convolutions and sums at targets
        # HeteroConv requires all kwargs to end with '_dict'.
        # It strips the suffix and passes the remainder to each per-edge conv:
        #   edge_weight_dict={et: tensor} → conv(..., edge_weight=tensor)
        out_dict = self.conv(
            x_dict,
            edge_index_dict,
            edge_weight_dict=edge_weight_dict or {},
        )

        result: Dict[str, Tensor] = {}
        for nt, out in out_dict.items():
            out = F.elu(out)
            # Residual: only if the source node type has features in this layer
            if nt in x_dict and x_dict[nt].size(-1) == out.size(-1):
                out = out + x_dict[nt]
            result[nt] = self.norms[nt](out)

        # Pass through node types that received no messages this layer
        for nt, x in x_dict.items():
            if nt not in result:
                result[nt] = x

        return result


# ── HeteroRGAT ────────────────────────────────────────────────────────────────

class HeteroRGAT(nn.Module):
    """
    L-layer Heterogeneous Relational Graph Attention Network.

    The hidden dimension is kept constant across all layers (no projection
    between layers) so residuals require no linear transform.

    An optional input projection aligns the encoder hidden dim to
    hidden_channels before the first layer (in our case both are 768,
    so the projection is skipped).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        heads: int,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
    ) -> None:
        super().__init__()
        node_types = metadata[0]

        # Input projection (identity mapping when dims match)
        if in_channels != hidden_channels:
            self.input_proj: Optional[nn.ModuleDict] = nn.ModuleDict(
                {nt: nn.Linear(in_channels, hidden_channels) for nt in node_types}
            )
        else:
            self.input_proj = None

        self.layers = nn.ModuleList([
            HeteroRGATLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                heads=heads,
                metadata=metadata,
            )
            for _ in range(num_layers)
        ])

    def forward(self, data: HeteroData) -> HeteroData:
        """
        Run L RGAT layers over the batched heterogeneous graph.

        Reads node features from data[node_type].x and writes updated
        features back in-place.  Returns the modified data for chaining.
        """
        # PyG's Batch.from_data_list() creates batch-index tensors on CPU
        # regardless of input device. Explicitly move every tensor we touch.
        device = next(self.parameters()).device

        x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
        edge_index_dict = {
            et: data[et].edge_index.to(device) for et in data.edge_types
        }
        edge_weight_dict: Dict = {
            et: data[et].edge_weight.to(device)
            for et in data.edge_types
            if hasattr(data[et], "edge_weight") and data[et].edge_weight is not None
        }

        # Optional input projection
        if self.input_proj is not None:
            x_dict = {
                nt: self.input_proj[nt](x) if nt in self.input_proj else x
                for nt, x in x_dict.items()
            }

        # L RGAT layers
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_weight_dict)

        # Write updated features back into the HeteroData object
        for nt, x in x_dict.items():
            data[nt].x = x

        return data
