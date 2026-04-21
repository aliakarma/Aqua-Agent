"""
src/utils/graph_utils.py
------------------------
Utilities for converting the EPANET water network topology into graph
objects suitable for the GAT anomaly scorer (PyTorch Geometric).

The water distribution network is modelled as a directed graph G_w = (V, E)
where V = nodes (junctions, reservoirs, tanks) and E = pipe segments.
(Paper Section 3.2, Equation 2)
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def build_edge_index(num_nodes: int,
                     pipe_from: np.ndarray,
                     pipe_to: np.ndarray,
                     bidirectional: bool = True) -> Tensor:
    """
    Build a PyTorch Geometric edge_index tensor from EPANET pipe connectivity.

    Args:
        num_nodes:     Total number of nodes |V|.
        pipe_from:     Array of source node indices for each pipe.
        pipe_to:       Array of target node indices for each pipe.
        bidirectional: If True, add reverse edges (undirected graph treatment).

    Returns:
        edge_index: LongTensor of shape [2, num_edges] (or [2, 2*num_edges]).
    """
    src = list(pipe_from)
    dst = list(pipe_to)
    if bidirectional:
        src = src + list(pipe_to)
        dst = dst + list(pipe_from)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def build_edge_attr(pipe_lengths: np.ndarray,
                    pipe_diameters: np.ndarray,
                    pipe_roughness: np.ndarray,
                    bidirectional: bool = True) -> Tensor:
    """
    Build edge feature matrix from pipe hydraulic properties.

    Features (d_edge = 3, from configs/network.yaml):
      [0] length (m, log-normalised)
      [1] diameter (mm, normalised)
      [2] roughness (Hazen-Williams coefficient, normalised)

    Args:
        pipe_lengths:    Array of pipe lengths (m).
        pipe_diameters:  Array of pipe diameters (mm).
        pipe_roughness:  Array of HW roughness coefficients.
        bidirectional:   Mirror attributes for reverse edges.

    Returns:
        edge_attr: FloatTensor of shape [num_edges, 3].
    """
    def _safe_log_norm(x: np.ndarray) -> np.ndarray:
        x = np.log1p(x)
        rng = x.max() - x.min() + 1e-8
        return (x - x.min()) / rng

    def _norm(x: np.ndarray) -> np.ndarray:
        rng = x.max() - x.min() + 1e-8
        return (x - x.min()) / rng

    feats = np.stack([
        _safe_log_norm(pipe_lengths),
        _norm(pipe_diameters),
        _norm(pipe_roughness),
    ], axis=1).astype(np.float32)

    if bidirectional:
        feats = np.concatenate([feats, feats], axis=0)

    return torch.from_numpy(feats)


def build_node_features(node_types: np.ndarray,
                        base_demands: np.ndarray,
                        elevations: np.ndarray) -> Tensor:
    """
    Build node feature matrix.

    Features:
      [0-2] One-hot node type (junction=0, reservoir=1, tank=2)
      [3]   Base demand (normalised)
      [4]   Elevation (normalised)

    Args:
        node_types:   Integer array of node type codes.
        base_demands: Base demand values (L/s).
        elevations:   Node elevation (m).

    Returns:
        x: FloatTensor of shape [num_nodes, 5].
    """
    num_nodes = len(node_types)
    type_onehot = np.zeros((num_nodes, 3), dtype=np.float32)
    for i, t in enumerate(node_types):
        type_onehot[i, int(t)] = 1.0

    def _norm(x: np.ndarray) -> np.ndarray:
        rng = x.max() - x.min() + 1e-8
        return ((x - x.min()) / rng).astype(np.float32)

    node_feats = np.concatenate([
        type_onehot,
        _norm(base_demands).reshape(-1, 1),
        _norm(elevations).reshape(-1, 1),
    ], axis=1)

    return torch.from_numpy(node_feats)


def pipe_to_node_mapping(pipe_from: np.ndarray,
                         pipe_to: np.ndarray,
                         num_edges: int,
                         num_nodes: int) -> Tuple[Tensor, Tensor]:
    """
    Return adjacency info for mapping pipe-level features to node-level.
    Used by the GAT to aggregate pipe (edge) anomaly scores to nodes.

    Returns:
        incidence_src: source nodes of pipes [num_edges]
        incidence_dst: target nodes of pipes [num_edges]
    """
    return (
        torch.tensor(pipe_from, dtype=torch.long),
        torch.tensor(pipe_to, dtype=torch.long),
    )


def synthetic_network_topology(num_nodes: int = 261,
                                num_edges: int = 213,
                                seed: int = 0) -> dict:
    """
    Generate a synthetic connected network topology for testing when no
    real EPANET .inp file is available.

    This does NOT replicate the exact paper network (which is not released),
    but produces a graph with matching dimensions that EPANET can simulate.

    Returns:
        dict with keys: pipe_from, pipe_to, pipe_lengths, pipe_diameters,
                        pipe_roughness, node_types, base_demands, elevations
    """
    rng = np.random.default_rng(seed)

    # Build a random connected graph: start with a spanning tree, add extras
    # Spanning tree: node i connects to node i+1 (chain), then add random edges
    pipe_from_list = list(range(num_nodes - 1))
    pipe_to_list = list(range(1, num_nodes))

    extra = num_edges - (num_nodes - 1)
    added = 0
    while added < extra:
        u = rng.integers(0, num_nodes)
        v = rng.integers(0, num_nodes)
        if u != v and (u, v) not in zip(pipe_from_list, pipe_to_list):
            pipe_from_list.append(int(u))
            pipe_to_list.append(int(v))
            added += 1

    pipe_from = np.array(pipe_from_list[:num_edges])
    pipe_to = np.array(pipe_to_list[:num_edges])

    # Hydraulic properties
    pipe_lengths = rng.uniform(50, 500, size=num_edges)      # m
    pipe_diameters = rng.choice([100, 150, 200, 300], size=num_edges)  # mm
    pipe_roughness = rng.uniform(100, 150, size=num_edges)   # HW coefficient

    # Node properties
    node_types = np.zeros(num_nodes, dtype=int)
    node_types[:2] = 1        # First 2 = reservoirs
    node_types[2:6] = 2       # Next 4 = tanks

    base_demands = rng.exponential(scale=0.5, size=num_nodes)  # L/s
    base_demands[:6] = 0.0    # Reservoirs/tanks have no demand
    elevations = rng.uniform(0, 100, size=num_nodes)           # m

    return {
        "pipe_from": pipe_from,
        "pipe_to": pipe_to,
        "pipe_lengths": pipe_lengths,
        "pipe_diameters": pipe_diameters,
        "pipe_roughness": pipe_roughness,
        "node_types": node_types,
        "base_demands": base_demands,
        "elevations": elevations,
    }
