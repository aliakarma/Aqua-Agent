"""
src/data/dataset.py
--------------------
PyTorch Dataset for supervised pre-training of the Anomaly Detection Agent.

Paper Section 4.2:
  "ADA is pre-trained supervisedly over 100 epochs using binary cross-entropy
   loss. Each sample is a 30-step lookback window of sensor features with
   edge-level binary leak labels as targets."

Dataset structure:
  Input  x: [num_edges, T, d_feat]   (T = 30, d_feat = 12)
  Target y: [num_edges]               binary leak indicator at window end
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WaterLeakDataset(Dataset):
    """
    Sliding-window dataset for ADA pre-training.

    Each sample consists of a T-step lookback window of per-edge sensor
    features and the binary leak label at the final time step.

    Supports both HDF5 (memory-mapped for large datasets) and NumPy .npz.
    """

    def __init__(self, split: str = "train",
                 data_dir: str = "data/generated",
                 splits_dir: str = "data/splits",
                 lookback: int = 30,
                 stride: int = 60,
                 num_edges: int = 213,
                 d_feat: int = 12):
        """
        Args:
            split:      "train" | "val" | "test"
            data_dir:   Directory containing simulation.h5 or simulation.npz
            splits_dir: Directory containing {split}_idx.npy
            lookback:   TCN lookback window T.
            stride:     Step stride between windows (Assumption A10: 60s)
            num_edges:  Number of pipe edges.
            d_feat:     Feature dimension (d_feat = 12).
        """
        self.split    = split
        self.lookback = lookback
        self.stride   = stride
        self.num_edges = num_edges
        self.d_feat    = d_feat

        # Load raw data arrays (memory-mapped for HDF5)
        self._data = self._load_data(data_dir)

        # Load split indices
        idx_path = Path(splits_dir) / f"{split}_idx.npy"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Split index not found: {idx_path}. "
                f"Run: python -m src.data.simulate --config configs/default.yaml"
            )
        raw_idx = np.load(idx_path)

        # Filter: only indices where a full lookback window is available
        min_start = lookback
        valid = raw_idx[raw_idx >= min_start]

        # Apply stride to reduce dataset size (Assumption A10)
        self._window_ends = valid[::stride]

        # Pre-compute features from raw signals (flows + pressures → d_feat tensor)
        self._flows     = self._data["flows"]      # [T_total, num_edges]
        self._pressures = self._data["pressures"]  # [T_total, num_nodes]
        self._demands   = self._data["demands"]    # [T_total, num_nodes]
        self._labels    = self._data["leak_labels"]  # [T_total, num_edges]

    def __len__(self) -> int:
        return len(self._window_ends)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Sensor feature window [num_edges, T, d_feat]
            y: Binary leak labels at final step [num_edges]
        """
        t_end   = int(self._window_ends[idx])
        t_start = t_end - self.lookback

        # Build feature tensor: [T, num_edges, d_feat]
        x = self._build_features(t_start, t_end)  # [T, E, d_feat]
        # Rearrange to [E, T, d_feat] for TCN (edges processed independently)
        x = x.transpose(1, 0, 2)  # [E, T, d_feat]
        x_tensor = torch.from_numpy(x)

        # Labels at the final timestep
        y = self._labels[t_end].astype(np.float32)  # [num_edges]
        y_tensor = torch.from_numpy(y)

        return x_tensor, y_tensor

    def get_graph_data(self, edge_index: Optional[torch.Tensor] = None) -> dict:
        """
        Return graph-level metadata for the GAT.

        Returns:
            dict with edge_index and num_edges
        """
        return {
            "num_edges": self.num_edges,
            "edge_index": edge_index,
        }

    def get_class_weights(self) -> Tuple[float, float]:
        """Compute positive/negative class weights for weighted BCE loss."""
        labels_subset = self._labels[self._window_ends].astype(float)
        pos_rate = labels_subset.mean()
        neg_rate = 1.0 - pos_rate
        # Inverse frequency weighting
        pos_weight = neg_rate / max(pos_rate, 1e-8)
        return 1.0, float(pos_weight)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(self, t_start: int, t_end: int) -> np.ndarray:
        """
        Compute the 12-dimensional feature representation for each edge
        over the time window [t_start, t_end).

        Returns:
            feat: [T, num_edges, d_feat]
        """
        T = t_end - t_start
        E = min(self.num_edges, self._flows.shape[1])
        feat = np.zeros((T, E, self.d_feat), dtype=np.float32)

        flows = self._flows[t_start:t_end, :E]      # [T, E]
        pres  = self._pressures[t_start:t_end, :min(self._pressures.shape[1], E)]
        dem   = self._demands[t_start:t_end, :E]

        # Instantaneous values
        feat[:, :, 0] = flows                          # mean proxy (instant)
        feat[:, :, 4] = flows                          # min proxy
        feat[:, :, 3] = flows                          # max proxy

        # Pressure (mapped to edge axis if V < E)
        V = pres.shape[1]
        feat[:, :V, 5] = pres
        feat[:, :V, 8] = pres
        feat[:, :V, 9] = pres

        # Demand
        feat[:, :min(dem.shape[1], E), 10] = dem

        # Compute rolling stats for each step (using causal window)
        window = 60   # 1-minute rolling window (at 1Hz)
        for t in range(T):
            w_start = max(0, t - window)
            f_win = flows[w_start:t+1]
            if f_win.shape[0] > 1:
                feat[t, :, 0] = f_win.mean(axis=0)    # FEAT_MEAN_FLOW
                feat[t, :, 1] = f_win.std(axis=0)     # FEAT_STD_FLOW
                feat[t, :, 2] = (f_win[-1] - f_win[0]) / f_win.shape[0]  # FEAT_ROC_FLOW
                feat[t, :, 3] = f_win.max(axis=0)     # FEAT_MAX_FLOW
                feat[t, :, 4] = f_win.min(axis=0)     # FEAT_MIN_FLOW

            if V > 0:
                p_win = pres[w_start:t+1]
                if p_win.shape[0] > 1:
                    feat[t, :V, 5] = p_win.mean(axis=0)
                    feat[t, :V, 6] = p_win.std(axis=0)
                    feat[t, :V, 7] = (p_win[-1] - p_win[0]) / p_win.shape[0]
                    feat[t, :V, 8] = p_win.max(axis=0)
                    feat[t, :V, 9] = p_win.min(axis=0)

        # Cyclic time encoding
        abs_t = np.arange(t_start, t_end)
        time_sin = np.sin(2 * np.pi * (abs_t % 86400) / 86400).astype(np.float32)
        feat[:, :, 11] = time_sin[:, np.newaxis]  # Broadcast to all edges

        return feat

    def _load_data(self, data_dir: str) -> dict:
        """Load simulation arrays from HDF5 or .npz."""
        h5_path  = Path(data_dir) / "simulation.h5"
        npz_path = Path(data_dir) / "simulation.npz"

        if h5_path.exists():
            try:
                import h5py
                f = h5py.File(h5_path, "r")
                return {k: f[k] for k in
                        ["flows", "pressures", "demands", "leak_labels", "exogenous"]}
            except ImportError:
                pass

        if npz_path.exists():
            data = np.load(npz_path, mmap_mode="r")
            return dict(data)

        raise FileNotFoundError(
            f"No simulation data found in {data_dir}. "
            "Run: python -m src.data.simulate --config configs/default.yaml"
        )


def build_dataloaders(cfg: dict, num_edges: int = 213,
                      d_feat: int = 12) -> dict:
    """
    Build train/val/test DataLoaders for ADA pre-training.

    Args:
        cfg: Full merged config.
        num_edges: Number of pipe edges.
        d_feat:    Feature dimension.

    Returns:
        Dict with keys "train", "val", "test".
    """
    train_cfg = cfg.get("training", {})
    ada_cfg   = train_cfg.get("ada", {})
    ds_cfg    = train_cfg.get("dataset", {})
    paths     = cfg.get("paths", {})

    lookback  = ada_cfg.get("tcn", {}).get("lookback_window", 30)
    stride    = ds_cfg.get("stride", 60)
    batch     = ada_cfg.get("training", {}).get("batch_size", 256)

    loaders = {}
    for split in ["train", "val", "test"]:
        ds = WaterLeakDataset(
            split=split,
            data_dir=paths.get("generated_dir", "data/generated"),
            splits_dir=paths.get("splits_dir", "data/splits"),
            lookback=lookback,
            stride=stride,
            num_edges=num_edges,
            d_feat=d_feat,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch,
            shuffle=(split == "train"),
            num_workers=min(4, cfg.get("num_workers", 4)),
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
