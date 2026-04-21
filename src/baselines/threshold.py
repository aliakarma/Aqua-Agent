"""
src/baselines/threshold.py
---------------------------
B1: Static pressure-zone threshold detector.

Paper Section 4.4 (Baselines):
  "B1: Static pressure-zone threshold — raises an alert when pressure drops
   below the 5th percentile of a 30-day rolling history."

This is the simplest baseline; it has no learning and no actuation.
"""

import numpy as np


class ThresholdDetector:
    """
    Per-edge percentile-based pressure threshold anomaly detector.

    Maintains a rolling history buffer and flags readings that drop
    below the configured percentile of the historical distribution.
    """

    def __init__(self, num_edges: int, percentile: float = 5.0,
                 history_days: int = 30):
        """
        Args:
            num_edges:    Number of pipe edges / sensors.
            percentile:   Alert threshold as a percentile of history.
            history_days: Rolling history length in days.
        """
        self.num_edges   = num_edges
        self.percentile  = percentile
        self.history_len = history_days * 86400   # seconds

        self._history: list = []       # Rolling deque of flow readings

    def reset(self) -> None:
        self._history.clear()

    def detect(self, obs: np.ndarray) -> np.ndarray:
        """
        Produce binary anomaly flags from a flat observation vector.

        Args:
            obs: Concatenation of [flows, pressures] — takes first num_edges values.

        Returns:
            flags: Binary array [num_edges].
        """
        flows = obs[:self.num_edges]
        self._history.append(flows.copy())
        if len(self._history) > self.history_len:
            self._history.pop(0)

        if len(self._history) < 60:
            # Not enough history — no alarms
            return np.zeros(self.num_edges, dtype=np.float32)

        history_arr = np.stack(self._history, axis=0)   # [H, E]
        thresholds  = np.percentile(history_arr, self.percentile, axis=0)
        flags = (flows < thresholds).astype(np.float32)
        return flags

    def score(self, obs: np.ndarray) -> np.ndarray:
        """Return soft scores (normalised distance below threshold)."""
        flows = obs[:self.num_edges]
        self._history.append(flows.copy())
        if len(self._history) > self.history_len:
            self._history.pop(0)

        if len(self._history) < 60:
            return np.zeros(self.num_edges, dtype=np.float32)

        history_arr = np.stack(self._history, axis=0)
        thresholds  = np.percentile(history_arr, self.percentile, axis=0)
        stds        = history_arr.std(axis=0) + 1e-8
        z_scores    = (thresholds - flows) / stds
        # Map z-score to probability via sigmoid
        probs = 1.0 / (1.0 + np.exp(-z_scores))
        return probs.astype(np.float32)
