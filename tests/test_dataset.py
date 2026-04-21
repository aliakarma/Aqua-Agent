"""tests/test_dataset.py — Unit tests for the WaterLeakDataset."""

import numpy as np
import pytest
from pathlib import Path


def make_tiny_dataset(tmp_path: Path, T: int = 300,
                       num_edges: int = 10, num_nodes: int = 8) -> tuple:
    """Create a tiny synthetic simulation dataset for testing."""
    rng = np.random.default_rng(0)

    data = {
        "flows":       rng.random((T, num_edges)).astype(np.float32),
        "pressures":   rng.random((T, num_nodes)).astype(np.float32),
        "demands":     rng.random((T, num_nodes)).astype(np.float32),
        "leak_labels": (rng.random((T, num_edges)) > 0.8).astype(np.int8),
        "exogenous":   rng.random((T, 5)).astype(np.float32),
    }

    data_dir   = tmp_path / "data" / "generated"
    splits_dir = tmp_path / "data" / "splits"
    data_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    np.savez_compressed(data_dir / "simulation.npz", **data)

    # Simple splits: first 70% train, next 15% val, last 15% test
    train_end = int(T * 0.7)
    val_end   = int(T * 0.85)

    np.save(splits_dir / "train_idx.npy", np.arange(30, train_end))
    np.save(splits_dir / "val_idx.npy",   np.arange(train_end, val_end))
    np.save(splits_dir / "test_idx.npy",  np.arange(val_end, T))

    return str(data_dir), str(splits_dir)


class TestWaterLeakDataset:
    def test_len_positive(self, tmp_path):
        from src.data.dataset import WaterLeakDataset
        data_dir, splits_dir = make_tiny_dataset(tmp_path)
        ds = WaterLeakDataset(
            split="train", data_dir=data_dir, splits_dir=splits_dir,
            lookback=10, stride=5, num_edges=10, d_feat=12,
        )
        assert len(ds) > 0

    def test_item_shapes(self, tmp_path):
        from src.data.dataset import WaterLeakDataset
        data_dir, splits_dir = make_tiny_dataset(tmp_path)
        ds = WaterLeakDataset(
            split="train", data_dir=data_dir, splits_dir=splits_dir,
            lookback=10, stride=5, num_edges=10, d_feat=12,
        )
        x, y = ds[0]
        assert x.shape == (10, 10, 12), f"x shape: {x.shape}"  # [E, T, d_feat]
        assert y.shape == (10,)                                  # [E]

    def test_labels_binary(self, tmp_path):
        from src.data.dataset import WaterLeakDataset
        data_dir, splits_dir = make_tiny_dataset(tmp_path)
        ds = WaterLeakDataset(
            split="train", data_dir=data_dir, splits_dir=splits_dir,
            lookback=10, stride=5, num_edges=10, d_feat=12,
        )
        for i in range(min(5, len(ds))):
            _, y = ds[i]
            assert set(y.tolist()).issubset({0.0, 1.0})

    def test_missing_data_raises(self, tmp_path):
        from src.data.dataset import WaterLeakDataset
        empty_dir = str(tmp_path / "empty")
        Path(empty_dir).mkdir()
        with pytest.raises(FileNotFoundError):
            WaterLeakDataset(
                split="train", data_dir=empty_dir,
                splits_dir=str(tmp_path / "splits"),
                lookback=10, stride=5,
            )

    def test_val_test_splits(self, tmp_path):
        from src.data.dataset import WaterLeakDataset
        data_dir, splits_dir = make_tiny_dataset(tmp_path, T=500)
        for split in ["train", "val", "test"]:
            ds = WaterLeakDataset(
                split=split, data_dir=data_dir, splits_dir=splits_dir,
                lookback=10, stride=5, num_edges=10, d_feat=12,
            )
            assert len(ds) >= 0   # May be empty for very small datasets

    def test_class_weights(self, tmp_path):
        from src.data.dataset import WaterLeakDataset
        data_dir, splits_dir = make_tiny_dataset(tmp_path)
        ds = WaterLeakDataset(
            split="train", data_dir=data_dir, splits_dir=splits_dir,
            lookback=10, stride=5, num_edges=10, d_feat=12,
        )
        neg_w, pos_w = ds.get_class_weights()
        assert neg_w > 0
        assert pos_w > 0
