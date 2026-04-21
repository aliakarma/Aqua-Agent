"""
src/agents/anomaly_agent.py
----------------------------
Anomaly Detection Agent (ADA) — Two-stage TCN → GAT detection pipeline.

Paper Section 3.7, Equations (9)-(11):
  Stage 1 (TCN): h_t = TCN_{θ_enc}(o_{t-30:t}^(MA))
  Stage 2 (GAT): l̂_t = σ(GAT_{θ_gat}(h_t, G_w))

  Edge e is flagged as anomalous if l̂_{t,e} > τ = 0.5

The ADA is pre-trained supervisedly (100 epochs, BCE loss) and its weights
are frozen during MAPPO deployment. Detection accuracy depends on the
hydraulic context being stable — this is guaranteed by the Governance Agent
(paper Remark 3.4: governance as hydraulic stabiliser).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.tcn import TemporalConvNet
from src.models.gat import GraphAnomalyScorer, build_line_graph_edge_index
from src.utils.logger import get_logger

logger = get_logger("anomaly_agent")


class AnomalyDetectionAgent(nn.Module):
    """
    Wraps the TCN encoder and GAT scorer into one callable ADA module.

    Usage:
        ada = AnomalyDetectionAgent(cfg, edge_index)
        scores = ada.score(obs_tensor)       # [1, num_edges, 1]
        flags  = ada.flag_anomalies(scores)  # binary np.ndarray [num_edges]
    """

    def __init__(self, cfg: dict,
                 edge_index: Optional[torch.Tensor] = None,
                 device: str = "cpu"):
        """
        Args:
            cfg:        Full merged config dict.
            edge_index: Line-graph adjacency [2, L] for GAT.
                        If None, GAT falls back to per-edge MLP scoring.
            device:     Torch device string.
        """
        super().__init__()
        self.device = torch.device(device)
        self.edge_index = (
            edge_index.to(self.device) if edge_index is not None else None
        )

        ada_cfg = cfg.get("ada", {})
        net_cfg = cfg.get("network", {})

        self.num_edges: int = net_cfg.get("num_edges", 213)
        self.lookback: int = ada_cfg.get("tcn", {}).get("lookback_window", 30)
        self.threshold: float = ada_cfg.get("threshold", 0.5)

        # ── Stage 1: TCN temporal encoder ──
        tcn_cfg = ada_cfg.get("tcn", {})
        self.tcn = TemporalConvNet(
            input_dim=net_cfg.get("monitoring", {}).get("feature_dim", 12),
            num_channels=tcn_cfg.get("num_channels", 64),
            num_layers=tcn_cfg.get("num_layers", 4),
            kernel_size=tcn_cfg.get("kernel_size", 3),
            latent_dim=tcn_cfg.get("latent_dim", 128),
            dilations=tcn_cfg.get("dilations", [1, 2, 4, 8]),
            dropout=tcn_cfg.get("dropout", 0.1),
        )

        # ── Stage 2: GAT anomaly scorer ──
        gat_cfg = ada_cfg.get("gat", {})
        self.gat = GraphAnomalyScorer(
            latent_dim=tcn_cfg.get("latent_dim", 128),
            num_heads=gat_cfg.get("num_heads", 3),
            hidden_dim=gat_cfg.get("hidden_dim", 64),
            output_dim=gat_cfg.get("output_dim", 1),
            dropout=gat_cfg.get("dropout", 0.1),
            concat_heads=gat_cfg.get("concat_heads", True),
        )

        self.to(self.device)
        self._frozen = False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, obs_window: torch.Tensor) -> torch.Tensor:
        """
        Run full ADA forward pass.

        Args:
            obs_window: Sensor feature tensor [batch, num_edges, T, d_feat]

        Returns:
            scores: Per-edge anomaly probabilities [batch, num_edges, 1]
        """
        obs_window = obs_window.to(self.device)

        # Rearrange from [B, E, T, d] to expected [B, E, T, d]
        # (TCN expects: batch × edges × time × features)
        h = self.tcn(obs_window)           # [B, E, latent_dim]
        scores = self.gat.score(h, self.edge_index)  # [B, E, 1]
        return scores

    def flag_anomalies(self, scores: torch.Tensor) -> np.ndarray:
        """
        Apply threshold τ to produce binary edge-level anomaly flags.

        Paper: "An edge e is flagged as anomalous if l̂_{t,e} > τ, τ = 0.5"

        Args:
            scores: Anomaly probabilities [batch, num_edges, 1]

        Returns:
            flags: Binary indicator [num_edges] (numpy)
        """
        probs = scores.squeeze(-1).squeeze(0)   # [num_edges]
        return (probs.detach().cpu().numpy() > self.threshold).astype(np.float32)

    def forward(self, obs_window: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Convenience forward: returns (scores tensor, binary flag array).
        """
        scores = self.score(obs_window)
        flags = self.flag_anomalies(scores)
        return scores, flags

    # ------------------------------------------------------------------
    # Training / checkpointing
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze all ADA weights (called after pre-training, before MAPPO)."""
        for param in self.parameters():
            param.requires_grad = False
        self._frozen = True
        logger.info("ADA weights frozen for MAPPO deployment.")

    def unfreeze(self) -> None:
        """Unfreeze weights (for fine-tuning or continued ADA training)."""
        for param in self.parameters():
            param.requires_grad = True
        self._frozen = False

    def save(self, path: str) -> None:
        """Save ADA state dict to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "tcn_state": self.tcn.state_dict(),
            "gat_state": self.gat.state_dict(),
            "threshold": self.threshold,
        }, path)
        logger.info(f"ADA checkpoint saved to {path}")

    def load(self, path: str, strict: bool = True) -> None:
        """Load ADA state dict from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.tcn.load_state_dict(checkpoint["tcn_state"], strict=strict)
        self.gat.load_state_dict(checkpoint["gat_state"], strict=strict)
        self.threshold = checkpoint.get("threshold", self.threshold)
        logger.info(f"ADA checkpoint loaded from {path}")

    def calibrate_threshold(self, val_scores: np.ndarray,
                             val_labels: np.ndarray) -> float:
        """
        Calibrate detection threshold τ on a held-out validation set.
        Selects τ maximising F1-score (paper Section 4.3).

        Args:
            val_scores: Predicted probabilities [N]
            val_labels: Binary ground-truth labels [N]

        Returns:
            Best threshold τ.
        """
        from sklearn.metrics import f1_score
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.9, 81):
            preds = (val_scores > t).astype(int)
            f1 = f1_score(val_labels.astype(int), preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        self.threshold = float(best_t)
        logger.info(f"ADA threshold calibrated: τ = {self.threshold:.3f} (val F1={best_f1:.4f})")
        return self.threshold
