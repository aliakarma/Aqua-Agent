"""
src/env/sensor_noise.py
-----------------------
Sensor noise model and Kalman filter for the Monitoring Agent.

Paper Section 3.3 (Observation Space):
  "o_t^(i) = O^(i)(s_t, ε_t^(i)), where ε_t^(i) ~ N(0, Σ_i) models sensor noise."

Paper Section 3.6 (Monitoring Agent):
  "Kalman filtering to remove impulse noise and sensor glitches."
  "Linear interpolation for single-cycle dropout; LSTM-based prediction
   for multi-cycle gaps."

The noise model provides per-sensor Gaussian corruption and tracks
which sensors are experiencing dropout at each time step.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class SensorNoiseModel:
    """
    Applies Gaussian noise to clean hydraulic readings.
    Also simulates sensor dropout events.
    """

    def __init__(self, num_edges: int, num_nodes: int,
                 pressure_sigma: float = 0.05,
                 flow_sigma: float = 0.01,
                 dropout_rate: float = 0.005,
                 seed: int = 44):
        """
        Args:
            num_edges:      Number of pipe edges (flow sensors).
            num_nodes:      Number of network nodes (pressure sensors).
            pressure_sigma: Standard deviation of pressure noise (m).
            flow_sigma:     Standard deviation of flow noise (L/s).
            dropout_rate:   Fraction of sensors dropping out per step.
            seed:           Random seed.
        """
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.pressure_sigma = pressure_sigma
        self.flow_sigma = flow_sigma
        self.dropout_rate = dropout_rate
        self.rng = np.random.default_rng(seed)

        # Track which sensors are currently in dropout
        self._flow_dropout: np.ndarray = np.zeros(num_edges, dtype=bool)
        self._pressure_dropout: np.ndarray = np.zeros(num_nodes, dtype=bool)
        self._dropout_remaining: np.ndarray = np.zeros(num_edges + num_nodes, dtype=int)

    def reset(self) -> None:
        """Reset dropout state between episodes."""
        self._flow_dropout[:] = False
        self._pressure_dropout[:] = False
        self._dropout_remaining[:] = 0

    def apply(self, clean_flows: np.ndarray,
              clean_pressures: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Corrupt clean readings with Gaussian noise and dropout.

        Args:
            clean_flows:     True flow rates (L/s).
            clean_pressures: True pressure heads (m).

        Returns:
            Tuple of (noisy_flows, noisy_pressures).
        """
        self._update_dropout()

        noisy_flows = clean_flows + self.rng.normal(
            0, self.flow_sigma, size=clean_flows.shape
        ).astype(np.float32)
        noisy_pressures = clean_pressures + self.rng.normal(
            0, self.pressure_sigma, size=clean_pressures.shape
        ).astype(np.float32)

        # Zero out dropped sensors (NaN in real systems; use 0 for simplicity)
        noisy_flows[self._flow_dropout] = 0.0
        noisy_pressures[self._pressure_dropout] = 0.0

        return noisy_flows, noisy_pressures

    def get_dropout_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return boolean dropout masks for flow and pressure sensors."""
        return self._flow_dropout.copy(), self._pressure_dropout.copy()

    def _update_dropout(self) -> None:
        """Randomly toggle sensor dropout states."""
        # Recover some sensors
        self._dropout_remaining[:] = np.maximum(0, self._dropout_remaining - 1)

        # Trigger new dropouts
        n_flow = self.num_edges
        new_dropout = self.rng.random(n_flow + self.num_nodes) < self.dropout_rate
        duration = self.rng.integers(1, 5, size=n_flow + self.num_nodes)

        for i in range(n_flow):
            if new_dropout[i] and self._dropout_remaining[i] == 0:
                self._dropout_remaining[i] = int(duration[i])
        for j in range(self.num_nodes):
            idx = n_flow + j
            if new_dropout[idx] and self._dropout_remaining[idx] == 0:
                self._dropout_remaining[idx] = int(duration[idx])

        self._flow_dropout = self._dropout_remaining[:n_flow] > 0
        self._pressure_dropout = self._dropout_remaining[n_flow:] > 0


class KalmanFilter1D:
    """
    1-D Kalman filter for removing impulse noise from a single sensor stream.

    Paper Section 3.6:
      "Outlier rejection: Kalman filtering to remove impulse noise."

    State model: x_{t+1} = x_t + w,  w ~ N(0, Q)   (random walk process)
    Measurement: z_t = x_t + v,       v ~ N(0, R)
    """

    def __init__(self, process_noise: float = 1e-4,
                 measurement_noise: float = 1e-2):
        """
        Args:
            process_noise:    Q — variance of state evolution noise.
            measurement_noise: R — variance of measurement noise.
        """
        self.Q = process_noise
        self.R = measurement_noise
        self._x: float = 0.0    # State estimate
        self._P: float = 1.0    # Error covariance

    def reset(self, initial_value: float = 0.0) -> None:
        self._x = initial_value
        self._P = 1.0

    def update(self, measurement: float) -> float:
        """
        Run one Kalman predict-update cycle.

        Args:
            measurement: Noisy sensor reading z_t.

        Returns:
            Filtered state estimate x̂_t.
        """
        # Predict
        x_pred = self._x
        P_pred = self._P + self.Q

        # Update (Kalman gain)
        K = P_pred / (P_pred + self.R)
        self._x = x_pred + K * (measurement - x_pred)
        self._P = (1 - K) * P_pred

        return self._x


class KalmanFilterBank:
    """
    Applies independent 1-D Kalman filters to all edges and nodes.
    Used by the Monitoring Agent to pre-process raw sensor streams.
    """

    def __init__(self, num_signals: int,
                 process_noise: float = 1e-4,
                 measurement_noise: float = 1e-2):
        self.filters = [
            KalmanFilter1D(process_noise, measurement_noise)
            for _ in range(num_signals)
        ]

    def reset(self, initial_values: np.ndarray) -> None:
        for i, f in enumerate(self.filters):
            v = float(initial_values[i]) if i < len(initial_values) else 0.0
            f.reset(v)

    def update(self, measurements: np.ndarray) -> np.ndarray:
        """Filter all signals; return array of filtered values."""
        return np.array(
            [f.update(float(m)) for f, m in zip(self.filters, measurements)],
            dtype=np.float32,
        )
