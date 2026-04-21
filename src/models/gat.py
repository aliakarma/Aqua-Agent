"""
src/models/gat.py
-----------------
Graph Attention Network (GAT) — Stage 2 of the Anomaly Detection Agent.

Paper Section 3.7, Equations (10)-(11):
  "A Graph Attention Network (GAT) with 3 attention heads propagates latent
   features across the pipeline graph G_w, yielding anomaly logits:
   l̂_t = σ(GAT_{θ_gat}(h_t, G_w))"

Architecture follows Veličković et al. (2018) "Graph Attention Networks."

The GAT receives per-edge latent representations h_t from the TCN and
propagates information across the pipeline graph topology. The output is
a per-edge anomaly score in [0,1] after sigmoid activation.

Because PyTorch Geometric operates on node-level features, we use a line-graph
transformation: each pipe edge becomes a node in the line graph, and two
line-graph nodes are adjacent iff the corresponding pipes share a junction.
This allows the GAT to propagate anomaly evidence along connected pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from torch_geometric.nn import GATConv
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False


class GraphAnomalyScorer(nn.Module):
    """
    GAT-based per-edge anomaly scorer.

    Input:
        h_t:        TCN latent features  [batch, num_edges, latent_dim]
        edge_index: Line-graph adjacency [2, num_line_edges]

    Output:
        l̂_t:        Per-edge anomaly logits (pre-sigmoid) [batch, num_edges, 1]

    Falls back to a simple MLP scorer if PyTorch Geometric is not installed,
    to allow the repository to run end-to-end without the optional PYG dep.
    """

    def __init__(self,
                 latent_dim: int = 128,       # TCN output dim
                 num_heads: int = 3,          # Paper: 3 attention heads
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 dropout: float = 0.1,
                 concat_heads: bool = True):  # Assumption A6
        super().__init__()

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.concat_heads = concat_heads
        self.use_pyg = _PYG_AVAILABLE

        if self.use_pyg:
            # GAT layer: in_channels → num_heads × hidden_dim (concatenated)
            self.gat1 = GATConv(
                in_channels=latent_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=concat_heads,
            )
            gat_out = hidden_dim * num_heads if concat_heads else hidden_dim
            # Final projection to scalar anomaly logit per edge
            self.out_proj = nn.Linear(gat_out, output_dim)
        else:
            # Fallback MLP scorer (no graph message passing)
            self.mlp_scorer = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h:          TCN features [batch, num_edges, latent_dim]
            edge_index: Line-graph COO adjacency [2, num_line_edges].
                        If None, falls back to MLP per-edge scoring.

        Returns:
            logits: Pre-sigmoid anomaly scores [batch, num_edges, 1]
        """
        batch, num_edges, _ = h.shape

        if self.use_pyg and edge_index is not None:
            # Process each batch element independently through GAT
            # (batch loop is acceptable since batch size is usually 1 at inference)
            out_list = []
            for b in range(batch):
                # h_b: [num_edges, latent_dim]
                h_b = h[b]
                gat_out = self.gat1(h_b, edge_index)  # [num_edges, gat_out]
                gat_out = F.elu(gat_out)
                gat_out = self.dropout(gat_out)
                logit = self.out_proj(gat_out)         # [num_edges, 1]
                out_list.append(logit)
            logits = torch.stack(out_list, dim=0)      # [batch, num_edges, 1]
        else:
            # Fallback: per-edge MLP scoring (no graph context)
            h_flat = h.reshape(batch * num_edges, -1)
            logits_flat = self.mlp_scorer(h_flat)
            logits = logits_flat.reshape(batch, num_edges, 1)

        return logits  # Raw logits (apply sigmoid externally for probabilities)

    def score(self, h: torch.Tensor,
              edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convenience method returning sigmoid-activated scores in [0, 1].

        Returns:
            scores: Anomaly probabilities [batch, num_edges, 1]
        """
        return torch.sigmoid(self.forward(h, edge_index))


def build_line_graph_edge_index(pipe_from: torch.Tensor,
                                 pipe_to: torch.Tensor,
                                 num_nodes: int) -> torch.Tensor:
    """
    Construct the line graph edge_index for the pipeline graph.

    Two pipe edges e1 and e2 are adjacent in the line graph iff they share
    a junction node. This encodes the physical connectivity of the water network.

    Args:
        pipe_from:  Source node indices of pipes [num_edges]
        pipe_to:    Target node indices of pipes [num_edges]
        num_nodes:  Total number of junctions/nodes in the network

    Returns:
        line_edge_index: Line-graph adjacency [2, num_line_edges]
    """
    num_edges = len(pipe_from)

    # Build node-to-incident-edges mapping
    node_to_edges: dict = {n: [] for n in range(num_nodes)}
    for e_idx in range(num_edges):
        u = int(pipe_from[e_idx])
        v = int(pipe_to[e_idx])
        if u < num_nodes:
            node_to_edges[u].append(e_idx)
        if v < num_nodes:
            node_to_edges[v].append(e_idx)

    # Two edges sharing a node are adjacent in the line graph
    src_list, dst_list = [], []
    for node, edges in node_to_edges.items():
        for i in range(len(edges)):
            for j in range(len(edges)):
                if i != j:
                    src_list.append(edges[i])
                    dst_list.append(edges[j])

    if len(src_list) == 0:
        # Fallback: no adjacency (isolated edges)
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor([src_list, dst_list], dtype=torch.long)
