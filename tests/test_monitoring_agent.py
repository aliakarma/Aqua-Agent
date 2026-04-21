"""tests/test_monitoring_agent.py — Unit tests for the Monitoring Agent."""

import numpy as np
import pytest

SMALL_CFG = {
    "sampling_hz": 1,
    "feature_window_minutes": 1,
    "feature_dim": 12,
    "kalman_process_noise": 1e-4,
    "kalman_measurement_noise": 1e-2,
    "lstm_imputer_hidden": 16,
}


def make_reading(num_edges=10, num_nodes=8, step=0):
    from src.env.digital_twin import SensorReading
    return SensorReading(
        noisy_flows=np.random.rand(num_edges).astype(np.float32),
        noisy_pressures=np.random.rand(num_nodes).astype(np.float32),
        noisy_demands=np.random.rand(num_nodes).astype(np.float32),
        timestep=step,
    )


def test_monitoring_agent_warmup():
    from src.agents.monitoring_agent import MonitoringAgent
    ma = MonitoringAgent(10, 8, 3, SMALL_CFG, seed=42)
    ma.reset()
    # buf_len = lookback(30) + feature_window(1min×60s=60) = 90
    # Run 100 steps to guarantee the buffer fills and returns a result
    result = None
    for i in range(100):
        result = ma.process(make_reading(10, 8, i))
    assert result is not None, "MA never exited warmup after 100 steps"
    assert result.shape == (10, 12)


def test_monitoring_agent_reset():
    from src.agents.monitoring_agent import MonitoringAgent
    ma = MonitoringAgent(10, 8, 3, SMALL_CFG, seed=42)
    ma.reset()
    for i in range(200):
        ma.process(make_reading(10, 8, i))
    ma.reset()
    # After reset, buffer should be empty
    result = ma.process(make_reading(10, 8, 0))
    assert result is None


def test_lookback_tensor_shape():
    from src.agents.monitoring_agent import MonitoringAgent
    ma = MonitoringAgent(10, 8, 3, SMALL_CFG, seed=42)
    ma.reset()
    for i in range(200):
        ma.process(make_reading(10, 8, i))
    tensor = ma.get_lookback_tensor()
    assert tensor is not None
    # Shape: [1, lookback, E, d_feat] → after permute in trainer: [1, E, T, d_feat]
    assert len(tensor.shape) == 4
