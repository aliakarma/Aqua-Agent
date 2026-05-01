"""
src/training/train_lstm.py
---------------------------
Supervised training of the B2 centralised LSTM baseline detector.

FIX-03 (Reviewer 1 Issue 3 / Reviewer 2 Moderate Issue 3):
  The original repository shipped evaluate.py expecting checkpoints/lstm_best.pt
  but provided no training script to produce it.  This script mirrors train_ada.py
  for LSTMDetector, producing a reproducible B2 checkpoint.

FIX-09 (Reviewer 1 Moderate Issue 2):
  pos_weight is applied to BCEWithLogitsLoss (matching ADA's training) so that
  class imbalance at low leak-prevalence rates does not systematically
  disadvantage B2 relative to ADA.

Paper Section 4.4:
  "B2: Centralised LSTM (128 units, 2 layers) trained supervisedly to detect
   leaks. Unlike AquaAgent, B2 is a passive detector with no actuation."

Usage:
    bash scripts/run_train_lstm.sh
    # or directly:
    python -m src.training.train_lstm --config configs/default.yaml --seed 42
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.lstm_centralised import LSTMDetector
from src.data.dataset import WaterLeakDataset
from src.data.simulate import load_config
from src.evaluation.metrics import compute_f1, compute_auc
from src.utils.seed import set_seed
from src.utils.logger import get_logger, TBLogger

logger = get_logger("train_lstm")


class LSTMDataset(torch.utils.data.Dataset):
    """
    Adapter that exposes WaterLeakDataset samples as flat sensor vectors
    suitable for the LSTM baseline.

    The LSTM receives a sequence of concatenated (flow, pressure) readings
    over the lookback window T.  Shape: [T, num_edges + num_nodes].
    Label: binary leak indicator at the final step [num_edges].
    """

    def __init__(self, water_ds: WaterLeakDataset, num_nodes: int):
        self.water_ds = water_ds
        self.num_nodes = num_nodes

    def __len__(self) -> int:
        return len(self.water_ds)

    def __getitem__(self, idx: int):
        x, y = self.water_ds[idx]  # x: [E, T, d_feat], y: [E]
        E, T, d = x.shape
        # Use feature 0 (flow mean) for each edge and feature 5 (pressure mean)
        # mapped back to node axis (first N_v edges carry pressure).
        flows = x[:, :, 0]          # [E, T]
        pres  = x[:min(E, self.num_nodes), :, 5]  # [V, T]

        # Pad pressure to num_edges if V < E
        if pres.shape[0] < E:
            pad = torch.zeros(E - pres.shape[0], T)
            pres = torch.cat([pres, pad], dim=0)  # [E, T]

        # Concat along feature axis → [T, 2E] then take only [T, E+num_nodes]
        seq = torch.cat([flows.T, pres.T], dim=1)  # [T, 2E]
        seq = seq[:, : E + self.num_nodes]          # [T, E+num_nodes]

        return seq.float(), y.float()


class LSTMTrainer:
    """Supervised training loop for the LSTMDetector (B2 baseline)."""

    def __init__(self, cfg: dict, loaders: dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = torch.device(device)

        # FIX-05: read from cfg.ada or cfg.training.ada (both are now available).
        ada_cfg   = cfg.get("ada", {})
        train_cfg = ada_cfg.get("training", {})
        lstm_cfg  = cfg.get("baselines", {}).get("lstm", {})
        net_cfg   = cfg.get("network", {})

        self.num_edges = net_cfg.get("num_edges", 213)
        self.num_nodes = net_cfg.get("num_nodes", 261)
        self.lookback  = ada_cfg.get("tcn", {}).get("lookback_window", 30)

        self.model = LSTMDetector(
            input_dim=self.num_edges + self.num_nodes,
            hidden_dim=lstm_cfg.get("hidden_dim", 128),
            num_layers=lstm_cfg.get("num_layers", 2),
            num_edges=self.num_edges,
        ).to(self.device)

        # FIX Class imbalance hardcoding: Use dataset's computed class weights instead of config constant.
        _, pos_weight_val = loaders["train"].dataset.water_ds.get_class_weights()
        logger.info(f"Using dynamically computed class weight: {pos_weight_val:.4f}")
        pos_weight = torch.tensor([pos_weight_val], device=self.device)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        lr  = lstm_cfg.get("learning_rate", train_cfg.get("learning_rate", 1e-3))
        wd  = lstm_cfg.get("weight_decay",  train_cfg.get("weight_decay",  1e-5))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        epochs = lstm_cfg.get("epochs", train_cfg.get("epochs", 100))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.max_epochs = epochs
        self.patience   = lstm_cfg.get("patience", train_cfg.get("early_stopping_patience", 15))

        ckpt_dir = Path(cfg.get("paths", {}).get("checkpoints_dir", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt = str(ckpt_dir / "lstm_best.pt")
        self.last_ckpt = str(ckpt_dir / "lstm_last.pt")

        log_dir = cfg.get("paths", {}).get("logs_dir", "logs")
        tb_on   = cfg.get("logging", {}).get("tensorboard", True)
        self.tb = TBLogger(str(Path(log_dir) / "lstm"), enabled=tb_on)

    def train(self, loaders: dict) -> None:
        best_val_f1  = 0.0
        patience_ctr = 0
        epoch = 1

        logger.info(
            f"B2 LSTM training: {self.max_epochs} epochs, device={self.device}, "
            f"train={len(loaders['train'].dataset):,} samples"
        )

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            tr = self._run_epoch(loaders["train"], train=True)
            va = self._run_epoch(loaders["val"],   train=False)
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"loss={tr['loss']:.4f}  val_F1={va['f1']:.4f}  "
                f"val_AUC={va['auc']:.4f}  ({time.time()-t0:.1f}s)"
            )
            self.tb.log_dict(tr, epoch, prefix="lstm/train")
            self.tb.log_dict(va, epoch, prefix="lstm/val")

            if va["f1"] > best_val_f1:
                best_val_f1 = va["f1"]
                patience_ctr = 0
                torch.save(self.model.state_dict(), self.best_ckpt)
                logger.info(f"  New best val F1: {best_val_f1:.4f} — saved to {self.best_ckpt}")
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        torch.save(self.model.state_dict(), self.last_ckpt)
        logger.info(f"B2 LSTM training complete. Best val F1: {best_val_f1:.4f}")
        self.tb.close()

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_loss = 0.0
        all_scores, all_labels = [], []

        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            for x_seq, y in loader:
                x_seq = x_seq.to(self.device)   # [B, T, input_dim]
                y     = y.to(self.device)        # [B, num_edges]

                logits = self.model(x_seq)       # [B, num_edges]
                loss   = self.criterion(logits, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * x_seq.size(0)
                scores = torch.sigmoid(logits).detach().cpu().numpy()
                all_scores.append(scores.reshape(-1))
                all_labels.append(y.cpu().numpy().reshape(-1))

        n  = sum(len(s) for s in all_scores)
        s  = np.concatenate(all_scores)
        lb = np.concatenate(all_labels)
        return {"loss": total_loss / max(n, 1), "f1": compute_f1(s, lb), "auc": compute_auc(s, lb)}


def build_lstm_loaders(cfg: dict, num_edges: int, num_nodes: int,
                        lookback: int) -> dict:
    """Build train/val DataLoaders wrapping WaterLeakDataset with LSTM adapter."""
    paths    = cfg.get("paths", {})
    ada_cfg  = cfg.get("ada", {})
    ds_cfg   = cfg.get("ada", {}).get("training", {})
    batch    = cfg.get("baselines", {}).get("lstm", {}).get(
        "batch_size", ds_cfg.get("batch_size", 256)
    )
    stride   = cfg.get("ada", {}).get("training", {}).get("stride", 60)

    loaders = {}
    for split in ("train", "val"):
        water_ds = WaterLeakDataset(
            split=split,
            data_dir=paths.get("generated_dir", "data/generated"),
            splits_dir=paths.get("splits_dir", "data/splits"),
            lookback=lookback,
            stride=stride,
            num_edges=num_edges,
        )
        lstm_ds = LSTMDataset(water_ds, num_nodes=num_nodes)
        loaders[split] = DataLoader(
            lstm_ds,
            batch_size=batch,
            shuffle=(split == "train"),
            num_workers=min(4, cfg.get("num_workers", 4)),
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders


def main():
    parser = argparse.ArgumentParser(description="Train B2 LSTM baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    net_cfg    = cfg.get("network", {})
    ada_cfg    = cfg.get("ada", {})
    num_edges  = net_cfg.get("num_edges", 213)
    num_nodes  = net_cfg.get("num_nodes", 261)
    lookback   = ada_cfg.get("tcn", {}).get("lookback_window", 30)

    loaders = build_lstm_loaders(cfg, num_edges=num_edges,
                                  num_nodes=num_nodes, lookback=lookback)
    trainer = LSTMTrainer(cfg, loaders, device=args.device)
    trainer.train(loaders)
    logger.info("B2 LSTM training complete.")


if __name__ == "__main__":
    main()
