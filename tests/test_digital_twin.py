"""tests/test_digital_twin.py — Unit tests for the DigitalTwin and sub-components."""

import numpy as np
import pytest

MOCK_CFG = {
    "reproducibility": {
        "strict_epanet": False,
    },
    "network": {
        "num_nodes": 20, "num_edges": 15, "num_demand_zones": 5,
        "num_prv": 2, "inp_file": "data/raw/nonexistent.inp",
        "simulation": {"duration_days": 1},
        "sensor_noise": {"pressure_sigma": 0.05, "flow_sigma": 0.01},
        "monitoring": {"sampling_hz": 1, "feature_window_minutes": 1,
                       "feature_dim": 12, "kalman_process_noise": 1e-4,
                       "kalman_measurement_noise": 1e-2, "lstm_imputer_hidden": 16},
        "leak": {"event_probability": 0.5, "max_simultaneous": 2,
                 "min_simultaneous": 1, "profiles": ["burst"],
                 "profile_weights": [1.0], "lognormal_mu": -1.5,
                 "lognormal_sigma": 0.8, "burst_duration_mean": 0.01,
                 "background_duration_mean": 0.1, "slow_onset_growth_hours": 0.01},
    },
    "governance": {"enabled": False},
    "paths": {"audit_log": "/tmp/test_audit.csv"},
    "mappo": {"reward": {"alpha": 0.55, "beta": 0.45, "gamma_r": 2.0}},
}


def test_digital_twin_reset_mock():
    from src.env.digital_twin import DigitalTwin
    dt = DigitalTwin(MOCK_CFG, seed=42)
    state = dt.reset()
    assert state.flow_rates.shape == (15,)
    assert state.pressures.shape == (20,)
    assert state.demands.shape == (20,)
    assert state.leak_indicator.shape == (15,)
    assert state.exogenous.shape == (5,)


def test_digital_twin_step():
    from src.env.digital_twin import DigitalTwin
    dt = DigitalTwin(MOCK_CFG, seed=42)
    dt.reset()
    action = {"type": 3, "edge": 0, "valve": 0, "delta": 0.0}
    state, done = dt.step(action)
    assert state.flow_rates.shape[0] == 15
    assert isinstance(done, bool)


def test_sensor_readings_noisy():
    from src.env.digital_twin import DigitalTwin
    dt = DigitalTwin(MOCK_CFG, seed=42)
    dt.reset()
    r = dt.get_sensor_readings()
    assert r.noisy_flows.shape == (15,)
    assert r.noisy_pressures.shape == (20,)


def test_leak_injector_basic():
    from src.env.leak_injector import LeakInjector
    cfg = MOCK_CFG["network"]["leak"]
    li = LeakInjector(num_nodes=20, num_edges=15, cfg=cfg, seed=42)
    li.reset()
    assert li.get_leak_indicator().shape == (15,)
    assert li.get_leak_indicator().sum() == 0.0   # No leaks at reset


def test_kalman_filter():
    from src.env.sensor_noise import KalmanFilter1D
    kf = KalmanFilter1D(process_noise=1e-4, measurement_noise=1e-2)
    kf.reset(5.0)
    noisy = [5.0 + np.random.randn() * 0.5 for _ in range(20)]
    estimates = [kf.update(z) for z in noisy]
    # Filter should converge toward true value (5.0)
    assert abs(np.mean(estimates[-5:]) - 5.0) < 1.0


def test_topology_consistency():
    from src.env.digital_twin import DigitalTwin
    dt = DigitalTwin(MOCK_CFG, seed=0)
    dt.reset()
    topo = dt.get_topology()
    assert len(topo["pipe_from"]) == 15
    assert len(topo["pipe_to"]) == 15
    assert len(topo["node_types"]) == 20
