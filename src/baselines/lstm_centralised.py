"""
src/baselines/lstm_centralised.py
----------------------------------
B2: Centralised LSTM-based anomaly detector.

Paper Section 4.4:
  "B2: Centralised LSTM (128 units, 2 layers) trained supervisedly to detect
   leaks. Unlike AquaAgent, B2 is a passive detector with no actuation."

This baseline encodes the global sensor vector (flows + pressures) with a
shared LSTM and produces per-edge leak probability scores. It is trained
end-to-end with BCE loss in the same way as ADA (but without the graph component).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional


class LSTMDetector(nn.Module):
    """
    Two-layer bidirectional LSTM passive leak detector.
    Paper: 128 hidden units, 2 layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_edges: int = 213,
                 dropout: float = 0.1):
        """
        Args:
            input_dim:  Dimension of concatenated sensor vector (flows + pressures).
            hidden_dim: LSTM hidden state dimension (paper: 128).
            num_layers: Number of LSTM layers (paper: 2).
            num_edges:  Number of output edge scores.
            dropout:    Dropout between LSTM layers.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_edges  = num_edges

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_edges),
        )
        self._device = "cpu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sensor sequence [batch, T, input_dim]

        Returns:
            logits: Per-edge anomaly logits at final step [batch, num_edges]
        """
        out, _ = self.lstm(x)          # [B, T, hidden_dim]
        logits = self.output_head(out[:, -1, :])  # [B, num_edges]
        return logits

    def score(self, obs: np.ndarray, history: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Produce per-edge anomaly scores from current + optional history.

        Args:
            obs:     Current sensor vector [input_dim].
            history: Optional recent history [T-1, input_dim]. If None,
                     uses only the current reading as a 1-step sequence.

        Returns:
            scores: Anomaly probabilities [num_edges].
        """
        self.eval()
        if history is not None:
            seq = np.concatenate([history, obs[np.newaxis, :]], axis=0)
        else:
            seq = obs[np.newaxis, :]   # [1, input_dim]

        x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0)  # [1, T, D]
        with torch.no_grad():
            logits = self.forward(x)
            scores = torch.sigmoid(logits).squeeze(0).numpy()
        return scores.astype(np.float32)

    def detect(self, obs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary detection flags."""
        return (self.score(obs) >= threshold).astype(np.float32)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))
