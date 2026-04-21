"""
src/training/train_ada.py
--------------------------
Supervised pre-training of the Anomaly Detection Agent (ADA).

Paper Section 4.3:
  "ADA pre-training: 100 epochs, binary cross-entropy loss, Adam (lr=1e-3,
   batch=256). Early stopping with patience=15 on validation F1."

Training objective:
  L_BCE = -[y·log(σ(ŷ)) + (1−y)·log(1−σ(ŷ))]  (per edge, per sample)

After training, ADA weights are frozen and the threshold τ is calibrated
on the validation set before MAPPO training begins.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.agents.anomaly_agent import AnomalyDetectionAgent
from src.data.dataset import build_dataloaders
from src.data.simulate import load_config
from src.evaluation.metrics import compute_f1, compute_auc
from src.utils.seed import set_seed
from src.utils.logger import get_logger, TBLogger
from src.models.gat import build_line_graph_edge_index
from src.utils.graph_utils import synthetic_network_topology

logger = get_logger("train_ada")


class ADATrainer:
    """Manages the full ADA supervised pre-training loop."""

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = torch.device(device)
        self.ada_cfg = cfg.get("ada", {})
        self.train_cfg = self.ada_cfg.get("training", {})
        self.net_cfg = cfg.get("network", {})

        self.num_edges = self.net_cfg.get("num_edges", 213)
        self.num_nodes = self.net_cfg.get("num_nodes", 261)

        # Build line-graph edge_index for GAT
        topo = synthetic_network_topology(self.num_nodes, self.num_edges, seed=0)
        edge_index = build_line_graph_edge_index(
            torch.tensor(topo["pipe_from"]),
            torch.tensor(topo["pipe_to"]),
            self.num_nodes,
        ).to(self.device)

        self.model = AnomalyDetectionAgent(cfg, edge_index=edge_index, device=str(self.device))

        # Loss
        pos_weight = torch.tensor(
            self.train_cfg.get("pos_weight", 2.0), device=self.device
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        lr = self.train_cfg.get("learning_rate", 1e-3)
        wd = self.train_cfg.get("weight_decay", 1e-5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        # LR scheduler
        T_max = self.train_cfg.get("scheduler_T_max", 100)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max)

        # Training state
        self.best_val_f1  = 0.0
        self.best_epoch   = 0
        self.patience     = self.train_cfg.get("early_stopping_patience", 15)
        self.patience_ctr = 0

        # Paths
        self.ckpt_dir = Path(cfg.get("paths", {}).get("checkpoints_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt = str(self.ckpt_dir / "ada_best.pt")
        self.last_ckpt = str(self.ckpt_dir / "ada_last.pt")

        # TensorBoard
        log_dir = cfg.get("paths", {}).get("logs_dir", "logs")
        tb_enabled = cfg.get("logging", {}).get("tensorboard", True)
        self.tb = TBLogger(str(Path(log_dir) / "ada"), enabled=tb_enabled)

    def train(self, loaders: dict) -> None:
        """
        Run the full ADA training loop.

        Args:
            loaders: Dict with "train", "val" DataLoaders.
        """
        max_epochs = self.train_cfg.get("epochs", 100)
        logger.info(
            f"ADA training: {max_epochs} epochs, "
            f"device={self.device}, "
            f"train={len(loaders['train'].dataset):,} samples"
        )

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(loaders["train"], epoch)
            val_metrics   = self.validate(loaders["val"], epoch)
            self.scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Loss={train_metrics['loss']:.4f}  "
                f"Val F1={val_metrics['f1']:.4f}  "
                f"Val AUC={val_metrics['auc']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            # Log to TensorBoard
            self.tb.log_dict(train_metrics, epoch, prefix="train")
            self.tb.log_dict(val_metrics,   epoch, prefix="val")

            # Save checkpoint on improvement
            val_f1 = val_metrics["f1"]
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch  = epoch
                self.patience_ctr = 0
                self.save_checkpoint(self.best_ckpt, epoch, val_metrics)
                logger.info(f"  ↑ New best val F1: {val_f1:.4f} — checkpoint saved")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(best epoch={self.best_epoch}, F1={self.best_val_f1:.4f})"
                    )
                    break

        # Save final checkpoint
        self.save_checkpoint(self.last_ckpt, epoch, val_metrics)
        logger.info(f"ADA training complete. Best val F1: {self.best_val_f1:.4f}")

        # Calibrate threshold on validation set
        self._calibrate_threshold(loaders["val"])

        # Freeze weights for MAPPO deployment
        self.model.freeze()
        self.tb.close()

    def _train_epoch(self, loader, epoch: int) -> Dict[str, float]:
        """One training epoch."""
        self.model.train()
        total_loss = 0.0
        all_scores, all_labels = [], []
        grad_clip = self.train_cfg.get("gradient_clip_norm", 1.0)

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)  # [B, E, T, d_feat]
            batch_y = batch_y.to(self.device)  # [B, E]

            # ADA forward: TCN → GAT → logits
            logits = self.model.gat(
                self.model.tcn(batch_x),
                self.model.edge_index,
            )                              # [B, E, 1]
            logits = logits.squeeze(-1)    # [B, E]

            loss = self.criterion(logits, batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            with torch.no_grad():
                scores = torch.sigmoid(logits).detach().cpu().numpy()
            all_scores.append(scores.reshape(-1))
            all_labels.append(batch_y.cpu().numpy().reshape(-1))

        n = sum(len(s) for s in all_scores)
        all_s = np.concatenate(all_scores)
        all_l = np.concatenate(all_labels)
        return {
            "loss": total_loss / max(n, 1),
            "f1":   compute_f1(all_s, all_l, threshold=self.model.threshold),
            "auc":  compute_auc(all_s, all_l),
        }

    def validate(self, loader, epoch: int = 0) -> Dict[str, float]:
        """Evaluate on validation loader."""
        self.model.eval()
        total_loss = 0.0
        all_scores, all_labels = [], []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model.gat(
                    self.model.tcn(batch_x), self.model.edge_index
                ).squeeze(-1)
                loss = self.criterion(logits, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                scores = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(scores.reshape(-1))
                all_labels.append(batch_y.cpu().numpy().reshape(-1))

        n = sum(len(s) for s in all_scores)
        all_s = np.concatenate(all_scores)
        all_l = np.concatenate(all_labels)
        return {
            "loss": total_loss / max(n, 1),
            "f1":   compute_f1(all_s, all_l, threshold=self.model.threshold),
            "auc":  compute_auc(all_s, all_l),
        }

    def save_checkpoint(self, path: str, epoch: int,
                        metrics: Dict[str, float]) -> None:
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
            "threshold": self.model.threshold,
        }, path)

    def _calibrate_threshold(self, val_loader) -> None:
        """Calibrate τ on validation set after training."""
        self.model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                logits = self.model.gat(
                    self.model.tcn(batch_x), self.model.edge_index
                ).squeeze(-1)
                scores = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(scores.reshape(-1))
                all_labels.append(batch_y.numpy().reshape(-1))

        all_s = np.concatenate(all_scores)
        all_l = np.concatenate(all_labels)
        self.model.calibrate_threshold(all_s, all_l)


def main():
    parser = argparse.ArgumentParser(description="Pre-train AquaAgent ADA")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    loaders = build_dataloaders(
        cfg,
        num_edges=cfg.get("network", {}).get("num_edges", 213),
        d_feat=cfg.get("network", {}).get("monitoring", {}).get("feature_dim", 12),
    )

    trainer = ADATrainer(cfg, device=args.device)
    trainer.train(loaders)
    logger.info("ADA pre-training complete.")


if __name__ == "__main__":
    main()
