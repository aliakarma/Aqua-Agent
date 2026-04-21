"""
src/data/simulate.py
---------------------
Generate the full 365-day simulation dataset from the EPANET digital twin.

Paper Section 4.1:
  "Simulation: 365 days, 1-Hz sampling, ≈3.15×10^7 samples.
   Split: 70% train / 15% val / 15% test by calendar month."

Outputs:
  data/generated/simulation.h5  — HDF5 file with datasets:
    /flows        [T, num_edges]   float32
    /pressures    [T, num_nodes]   float32
    /demands      [T, num_nodes]   float32
    /leak_labels  [T, num_edges]   int8  (0/1)
    /exogenous    [T, 5]           float32

  data/splits/train_idx.npy, val_idx.npy, test_idx.npy — sample index arrays

Usage:
    python -m src.data.simulate --config configs/default.yaml
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from src.env.digital_twin import DigitalTwin
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger("simulate")

try:
    import h5py
    _H5_AVAILABLE = True
except ImportError:
    _H5_AVAILABLE = False
    logger.warning("h5py not found — will save as .npz instead. Install: pip install h5py")


def load_config(config_path: str) -> dict:
    """Load and merge all YAML config files."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Merge sub-configs
    for sub_key, sub_file in [
        ("network", cfg.get("network_config", "configs/network.yaml")),
        ("training", cfg.get("training_config", "configs/training.yaml")),
        ("governance", cfg.get("governance_config", "configs/governance.yaml")),
    ]:
        sub_path = Path(sub_file)
        if sub_path.exists():
            with open(sub_path) as f:
                sub_cfg = yaml.safe_load(f)
            cfg[sub_key] = sub_cfg.get(sub_key, sub_cfg)

    return cfg


def run_simulation(cfg: dict, seed: int = 42) -> dict:
    """
    Run the full 365-day simulation and collect all arrays.

    Args:
        cfg:  Merged config dict.
        seed: Random seed.

    Returns:
        Dict with keys: flows, pressures, demands, leak_labels, exogenous
    """
    set_seed(seed)
    net_cfg = cfg.get("network", {})
    sim_cfg = net_cfg.get("simulation", {})

    duration_days = sim_cfg.get("duration_days", 365)
    duration_steps = duration_days * 86400

    dt = DigitalTwin(cfg, seed=seed)
    state = dt.reset()

    num_edges = dt.num_edges
    num_nodes = dt.num_nodes

    # Pre-allocate arrays
    # At 1Hz for 365 days: 31,536,000 steps → chunked storage
    chunk = 86400   # Process day by day to manage memory
    total_days = duration_days

    flows_list     = []
    pressures_list = []
    demands_list   = []
    labels_list    = []
    exog_list      = []

    logger.info(f"Starting simulation: {duration_days} days, {duration_steps:,} steps")
    t_start = time.time()

    for day in range(total_days):
        day_flows     = np.zeros((chunk, num_edges), dtype=np.float32)
        day_pressures = np.zeros((chunk, num_nodes), dtype=np.float32)
        day_demands   = np.zeros((chunk, num_nodes), dtype=np.float32)
        day_labels    = np.zeros((chunk, num_edges), dtype=np.int8)
        day_exog      = np.zeros((chunk, 5),         dtype=np.float32)

        for s in range(chunk):
            reading = dt.get_sensor_readings()
            day_flows[s]     = reading.noisy_flows
            day_pressures[s] = reading.noisy_pressures
            day_demands[s]   = reading.noisy_demands
            day_labels[s]    = state.leak_indicator.astype(np.int8)
            day_exog[s]      = state.exogenous

            action = {"type": 3, "edge": 0, "valve": 0, "delta": 0.0}  # no_op default
            state, done = dt.step(action)

        flows_list.append(day_flows)
        pressures_list.append(day_pressures)
        demands_list.append(day_demands)
        labels_list.append(day_labels)
        exog_list.append(day_exog)

        if (day + 1) % 30 == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"Day {day+1}/{total_days} — "
                f"leak_frac={day_labels.mean():.3f} — "
                f"elapsed={elapsed:.1f}s"
            )

    logger.info("Simulation complete. Concatenating arrays...")
    return {
        "flows":       np.concatenate(flows_list, axis=0),
        "pressures":   np.concatenate(pressures_list, axis=0),
        "demands":     np.concatenate(demands_list, axis=0),
        "leak_labels": np.concatenate(labels_list, axis=0),
        "exogenous":   np.concatenate(exog_list, axis=0),
    }


def save_dataset(arrays: dict, output_dir: str = "data/generated") -> None:
    """Save simulation arrays to HDF5 (or .npz fallback)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if _H5_AVAILABLE:
        h5_path = Path(output_dir) / "simulation.h5"
        with h5py.File(h5_path, "w") as f:
            for key, arr in arrays.items():
                f.create_dataset(key, data=arr, compression="gzip",
                                 compression_opts=4, chunks=True)
        logger.info(f"Dataset saved to {h5_path}")
    else:
        npz_path = Path(output_dir) / "simulation.npz"
        np.savez_compressed(npz_path, **arrays)
        logger.info(f"Dataset saved to {npz_path} (h5py fallback)")


def build_splits(total_steps: int, cfg: dict,
                 output_dir: str = "data/splits") -> None:
    """
    Build train/val/test splits by calendar month (paper: 70/15/15).
    Saves index arrays to disk.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_cfg = cfg.get("training", {}).get("dataset", {})
    train_months = train_cfg.get("train_months", list(range(1, 10)))
    val_months   = train_cfg.get("val_months", [10, 11])
    test_months  = train_cfg.get("test_months", [12])

    # Map step index → month (1-indexed)
    steps_per_month = total_steps // 12
    month_of_step = np.arange(total_steps) // steps_per_month + 1
    month_of_step = np.clip(month_of_step, 1, 12)

    train_idx = np.where(np.isin(month_of_step, train_months))[0]
    val_idx   = np.where(np.isin(month_of_step, val_months))[0]
    test_idx  = np.where(np.isin(month_of_step, test_months))[0]

    np.save(Path(output_dir) / "train_idx.npy", train_idx)
    np.save(Path(output_dir) / "val_idx.npy",   val_idx)
    np.save(Path(output_dir) / "test_idx.npy",  test_idx)

    logger.info(
        f"Splits: train={len(train_idx):,}  val={len(val_idx):,}  "
        f"test={len(test_idx):,}  (total={total_steps:,})"
    )


def main():
    parser = argparse.ArgumentParser(description="Run AquaAgent data simulation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--days", type=int, default=None,
                        help="Override duration_days (for quick testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.days:
        cfg.setdefault("network", {}).setdefault("simulation", {})["duration_days"] = args.days

    arrays = run_simulation(cfg, seed=args.seed)
    save_dataset(arrays, args.output_dir)
    build_splits(len(arrays["flows"]), cfg, args.splits_dir)
    logger.info("Data generation complete.")


if __name__ == "__main__":
    main()
