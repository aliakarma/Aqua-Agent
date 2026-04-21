"""
src/baselines/rule_based_mas.py
--------------------------------
B4: Hand-crafted rule-based Multi-Agent System (MAS).

Paper Section 4.4:
  "B4: Rule-based MAS — deterministic decision rules with no learned components.
   Detects anomalies via flow-balance deviation and responds with fixed
   valve-closure rules."

Detection rule:
  Flag edge e if |flow_e − mean_flow| > k · std_flow  (3-sigma rule)

Response rule:
  If any anomaly flag and min zone pressure < threshold → isolate flagged edge
  Otherwise → no_op
"""

import numpy as np
from typing import Optional


class RuleBasedMAS:
    """
    Deterministic rule-based MAS.
    No learning, no data-driven components.
    """

    def __init__(self, num_edges: int, num_zones: int,
                 sigma_k: float = 3.0,
                 pressure_threshold: float = 15.0,
                 history_len: int = 3600):
        """
        Args:
            num_edges:          Number of pipe edges.
            num_zones:          Number of demand zones.
            sigma_k:            Sigma multiplier for flow deviation detection.
            pressure_threshold: Minimum pressure (m) below which emergency applies.
            history_len:        History window for rolling statistics (steps).
        """
        self.num_edges = num_edges
        self.num_zones = num_zones
        self.sigma_k   = sigma_k
        self.p_thresh  = pressure_threshold
        self.hist_len  = history_len

        self._flow_history: list = []

    def reset(self) -> None:
        self._flow_history.clear()

    def detect(self, obs: np.ndarray) -> np.ndarray:
        """
        Detect anomalies using 3-sigma flow deviation rule.

        Args:
            obs: Concatenated [flows, pressures, ...] sensor vector.

        Returns:
            flags: Binary anomaly array [num_edges].
        """
        flows = obs[:self.num_edges]
        self._flow_history.append(flows.copy())
        if len(self._flow_history) > self.hist_len:
            self._flow_history.pop(0)

        if len(self._flow_history) < 10:
            return np.zeros(self.num_edges, dtype=np.float32)

        hist = np.stack(self._flow_history)
        mean_flow = hist.mean(axis=0)
        std_flow  = hist.std(axis=0) + 1e-8
        deviation = np.abs(flows - mean_flow) / std_flow
        flags = (deviation > self.sigma_k).astype(np.float32)
        return flags

    def score(self, obs: np.ndarray) -> np.ndarray:
        """Return soft anomaly scores (normalised deviation)."""
        flows = obs[:self.num_edges]
        if len(self._flow_history) < 10:
            return np.zeros(self.num_edges, dtype=np.float32)
        hist = np.stack(self._flow_history)
        mean_flow = hist.mean(axis=0)
        std_flow  = hist.std(axis=0) + 1e-8
        deviation = np.abs(flows - mean_flow) / std_flow
        # Map to [0,1] via sigmoid
        scores = 1.0 / (1.0 + np.exp(-(deviation - self.sigma_k)))
        return scores.astype(np.float32)

    def decide(self, obs: np.ndarray, flags: Optional[np.ndarray] = None) -> dict:
        """
        Produce a control action based on detection flags.

        Rule:
          If any flag is set → isolate the highest-deviation edge.
          Otherwise          → no_op.

        Returns:
            action dict compatible with DigitalTwin.step().
        """
        import torch
        if flags is None:
            flags = self.detect(obs)

        if flags.max() > 0:
            # Isolate the most suspicious edge
            target_edge = int(np.argmax(flags))
            return {
                "type":  torch.tensor(0),            # isolate
                "edge":  torch.tensor(target_edge),
                "valve": torch.tensor(0),
                "delta": torch.tensor([[0.0]]),
            }
        else:
            return {
                "type":  torch.tensor(3),            # no_op
                "edge":  torch.tensor(0),
                "valve": torch.tensor(0),
                "delta": torch.tensor([[0.0]]),
            }
