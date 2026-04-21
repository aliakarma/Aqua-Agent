"""
src/agents/monitoring_agent.py
-------------------------------
Monitoring Agent (MA) — AquaAgent's sensor processing layer.

Paper Section 3.6:
  "The Monitoring Agent interfaces with the sensor network, acquiring flow
   rates, pressure readings, and consumption timestamps at 1-Hz sampling."

Three-stage pipeline:
  1. Outlier rejection: Kalman filtering to remove impulse noise/glitches.
  2. Gap imputation:    Linear interpolation for single-cycle dropout;
                        LSTM-based prediction for multi-cycle gaps.
  3. Feature extraction: Rolling statistics over 5-minute windows →
                         observation tensors o_t^(MA) ∈ R^{|E| × d_feat}
                         where d_feat = 12.

The MA's observation buffers hold a lookback window of 30 steps (matching
the TCN encoder) plus a 5-minute feature window for rolling statistics.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np
import torch

from src.env.digital_twin import SensorReading
from src.env.sensor_noise import KalmanFilterBank
from src.utils.logger import get_logger

logger = get_logger("monitoring_agent")

# Feature indices (paper Assumption A4)
FEAT_MEAN_FLOW     = 0
FEAT_STD_FLOW      = 1
FEAT_ROC_FLOW      = 2   # Rate of change
FEAT_MAX_FLOW      = 3
FEAT_MIN_FLOW      = 4
FEAT_MEAN_PRES     = 5
FEAT_STD_PRES      = 6
FEAT_ROC_PRES      = 7
FEAT_MAX_PRES      = 8
FEAT_MIN_PRES      = 9
FEAT_DEMAND        = 10
FEAT_TIME_SIN      = 11   # Cyclic time-of-day encoding


class MonitoringAgent:
    """
    Processes raw sensor readings into clean, feature-rich observation tensors.

    Maintains a rolling buffer of the last `lookback_window + feature_window`
    readings to support both TCN lookback and rolling statistics.
    """

    def __init__(self, num_edges: int, num_nodes: int, num_zones: int,
                 cfg: dict, seed: int = 0):
        """
        Args:
            num_edges:       |E| — number of pipe edges.
            num_nodes:       |V| — number of network nodes.
            num_zones:       Number of demand zones.
            cfg:             Network monitoring sub-config.
            seed:            Random seed (for LSTM imputer).
        """
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.num_zones = num_zones
        self.sampling_hz: int = cfg.get("sampling_hz", 1)
        self.d_feat: int = cfg.get("feature_dim", 12)
        self.feature_window: int = cfg.get("feature_window_minutes", 5) * 60
        self.lookback: int = 30   # TCN lookback window

        # Total buffer length needed
        self._buf_len = self.lookback + self.feature_window

        # ── Kalman filter banks ──
        kalman_cfg = cfg.get("kalman_process_noise", 1e-4), cfg.get("kalman_measurement_noise", 1e-2)
        self._flow_kalman = KalmanFilterBank(
            num_edges, process_noise=kalman_cfg[0], measurement_noise=kalman_cfg[1]
        )
        self._pres_kalman = KalmanFilterBank(
            num_nodes, process_noise=kalman_cfg[0], measurement_noise=kalman_cfg[1]
        )

        # ── LSTM imputer (Assumption A3) ──
        lstm_hidden = cfg.get("lstm_imputer_hidden", 64)
        self._lstm_imputer = _LSTMImputer(
            input_dim=num_edges + num_nodes,
            hidden_dim=lstm_hidden,
        )

        # ── Rolling buffers ──
        # Each entry: np.ndarray [num_edges] for flow, [num_nodes] for pressure
        self._flow_buf: Deque[np.ndarray] = deque(maxlen=self._buf_len)
        self._pres_buf: Deque[np.ndarray] = deque(maxlen=self._buf_len)
        self._demand_buf: Deque[np.ndarray] = deque(maxlen=self._buf_len)

        # Dropout tracking for gap imputation
        self._gap_counters: np.ndarray = np.zeros(num_edges, dtype=int)
        self._current_step: int = 0

    def reset(self) -> None:
        """Clear all buffers at the start of a new episode."""
        self._flow_buf.clear()
        self._pres_buf.clear()
        self._demand_buf.clear()
        self._gap_counters[:] = 0
        self._current_step = 0
        self._flow_kalman.reset(np.zeros(self.num_edges))
        self._pres_kalman.reset(np.zeros(self.num_nodes))

    def process(self, reading: SensorReading) -> Optional[np.ndarray]:
        """
        Full MA processing pipeline for one sensor reading.

        Args:
            reading: Noisy SensorReading from the DigitalTwin.

        Returns:
            obs_tensor: Feature observation tensor [num_edges, d_feat], or
                        None if the buffer is not yet full (warm-up phase).
        """
        self._current_step += 1

        # ── Stage 1: Outlier rejection via Kalman filter ──
        filtered_flow = self._kalman_filter(reading)

        # ── Stage 2: Gap imputation ──
        imputed_flow = self._impute_gaps(filtered_flow, reading)

        # ── Buffer update ──
        self._flow_buf.append(imputed_flow)
        self._pres_buf.append(self._pres_kalman.update(reading.noisy_pressures))
        self._demand_buf.append(reading.noisy_demands.copy())

        # ── Stage 3: Feature extraction (only when buffer is full) ──
        if len(self._flow_buf) < self._buf_len:
            return None   # Still warming up

        return self._extract_features()

    def get_lookback_tensor(self) -> Optional[torch.Tensor]:
        """
        Return the last `lookback` processed feature frames for TCN input.

        Returns:
            tensor: [lookback, num_edges, d_feat] or None if buffer not ready.
        """
        if len(self._flow_buf) < self._buf_len:
            return None

        frames = []
        flow_list = list(self._flow_buf)
        pres_list = list(self._pres_buf)
        dem_list = list(self._demand_buf)

        for i in range(-self.lookback, 0):
            feat = self._compute_instant_features(
                flow_list[i], pres_list[i], dem_list[i]
            )
            frames.append(feat)

        tensor = np.stack(frames, axis=0)   # [T, E, d_feat]
        return torch.from_numpy(tensor).unsqueeze(0)  # [1, T, E, d_feat]

    # ------------------------------------------------------------------
    # Internal pipeline stages
    # ------------------------------------------------------------------

    def _kalman_filter(self, reading: SensorReading) -> np.ndarray:
        """Apply per-sensor 1-D Kalman filter to flow readings."""
        return self._flow_kalman.update(reading.noisy_flows)

    def _impute_gaps(self, filtered: np.ndarray,
                     reading: SensorReading) -> np.ndarray:
        """
        Impute dropped sensor readings.
         - 1-step gap: linear interpolation from last valid reading.
         - Multi-step gap: LSTM-based prediction.
        """
        # Detect dropped sensors (zero value from noise model)
        dropout_mask = reading.noisy_flows == 0.0

        if not dropout_mask.any():
            self._gap_counters[:] = 0
            return filtered

        imputed = filtered.copy()
        for e in np.where(dropout_mask)[0]:
            self._gap_counters[e] += 1
            gap = self._gap_counters[e]

            if gap == 1 and len(self._flow_buf) >= 2:
                # Linear interpolation between t-2 and current estimate
                prev = list(self._flow_buf)[-1][e]
                imputed[e] = prev   # Use last valid as single-step estimate
            elif gap > 1 and len(self._flow_buf) >= self.lookback:
                # LSTM imputation
                recent = np.stack(list(self._flow_buf)[-self.lookback:])
                imputed[e] = self._lstm_imputer.predict(recent, e)
            # else: use Kalman estimate as-is (handles initial gaps)

        # Reset counter for non-dropped sensors
        self._gap_counters[~dropout_mask] = 0
        return imputed

    def _extract_features(self) -> np.ndarray:
        """
        Compute rolling statistics over the feature window for the latest step.
        Returns feature array [num_edges, d_feat].
        """
        flow_window = np.stack(list(self._flow_buf)[-self.feature_window:])   # [W, E]
        pres_window = np.stack(list(self._pres_buf)[-self.feature_window:])   # [W, V]
        dem = list(self._demand_buf)[-1]                                       # [V]

        return self._compute_windowed_features(flow_window, pres_window, dem)

    def _compute_windowed_features(self, flow_w: np.ndarray,
                                    pres_w: np.ndarray,
                                    dem: np.ndarray) -> np.ndarray:
        """Build d_feat=12 rolling feature matrix [num_edges, 12]."""
        feat = np.zeros((self.num_edges, self.d_feat), dtype=np.float32)

        feat[:, FEAT_MEAN_FLOW] = flow_w.mean(axis=0)
        feat[:, FEAT_STD_FLOW]  = flow_w.std(axis=0)
        feat[:, FEAT_ROC_FLOW]  = (flow_w[-1] - flow_w[0]) / max(len(flow_w) - 1, 1)
        feat[:, FEAT_MAX_FLOW]  = flow_w.max(axis=0)
        feat[:, FEAT_MIN_FLOW]  = flow_w.min(axis=0)

        # Map node-level pressure stats to edge level via source-node proxy
        # Use mean over all nodes as a simple global signal per edge slot
        E = self.num_edges
        V = min(self.num_nodes, E)
        feat[:V, FEAT_MEAN_PRES] = pres_w.mean(axis=0)[:V]
        feat[:V, FEAT_STD_PRES]  = pres_w.std(axis=0)[:V]
        feat[:V, FEAT_ROC_PRES]  = (pres_w[-1] - pres_w[0])[:V] / max(len(pres_w) - 1, 1)
        feat[:V, FEAT_MAX_PRES]  = pres_w.max(axis=0)[:V]
        feat[:V, FEAT_MIN_PRES]  = pres_w.min(axis=0)[:V]

        # Demand (zone-level, broadcast to edge level)
        feat[:min(len(dem), E), FEAT_DEMAND] = dem[:min(len(dem), E)]

        # Cyclic time-of-day encoding
        t_frac = (self._current_step % 86400) / 86400.0
        feat[:, FEAT_TIME_SIN] = np.sin(2 * np.pi * t_frac)

        return feat

    def _compute_instant_features(self, flow: np.ndarray,
                                   pres: np.ndarray,
                                   dem: np.ndarray) -> np.ndarray:
        """Compute instantaneous (single-step) features without rolling stats."""
        feat = np.zeros((self.num_edges, self.d_feat), dtype=np.float32)
        feat[:, FEAT_MEAN_FLOW] = flow
        E = self.num_edges
        V = min(self.num_nodes, E)
        feat[:V, FEAT_MEAN_PRES] = pres[:V]
        feat[:min(len(dem), E), FEAT_DEMAND] = dem[:min(len(dem), E)]
        t_frac = (self._current_step % 86400) / 86400.0
        feat[:, FEAT_TIME_SIN] = np.sin(2 * np.pi * t_frac)
        return feat


import torch.nn as _nn


class _LSTMImputer(_nn.Module):
    """
    Lightweight LSTM for multi-cycle sensor gap imputation.
    Predicts the current value of a dropped sensor from its recent history.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = _nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                             num_layers=1, batch_first=True)
        self.fc = _nn.Linear(hidden_dim, input_dim)

    def predict(self, recent_flows: np.ndarray, edge_idx: int) -> float:
        """
        Predict the missing flow value for `edge_idx` at the current step.

        Args:
            recent_flows: Recent flow history [T, num_edges]
            edge_idx:     Which edge to impute

        Returns:
            Predicted flow value (float).
        """
        import torch
        x = torch.from_numpy(recent_flows).float().unsqueeze(0)  # [1, T, E]
        with torch.no_grad():
            out, _ = self.lstm(x)
            pred = self.fc(out[:, -1, :])   # [1, E]
        return float(pred[0, edge_idx])
