"""conftest.py — Shared pytest fixtures for AquaAgent test suite."""

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def small_cfg():
    """Minimal config dict for fast unit testing."""
    return {
        "network": {
            "num_nodes": 20, "num_edges": 15, "num_demand_zones": 5,
            "num_prv": 2, "inp_file": "data/raw/nonexistent.inp",
            "simulation": {"duration_days": 1},
            "sensor_noise": {"pressure_sigma": 0.05, "flow_sigma": 0.01},
            "monitoring": {
                "sampling_hz": 1, "feature_window_minutes": 1,
                "feature_dim": 12, "kalman_process_noise": 1e-4,
                "kalman_measurement_noise": 1e-2, "lstm_imputer_hidden": 16,
            },
            "leak": {
                "event_probability": 0.5, "max_simultaneous": 2,
                "min_simultaneous": 1, "profiles": ["burst"],
                "profile_weights": [1.0], "lognormal_mu": -1.5,
                "lognormal_sigma": 0.8, "burst_duration_mean": 0.01,
                "background_duration_mean": 0.1, "slow_onset_growth_hours": 0.01,
            },
        },
        "ada": {
            "tcn": {
                "num_channels": 16, "num_layers": 2, "kernel_size": 3,
                "latent_dim": 32, "dilations": [1, 2], "dropout": 0.0,
                "lookback_window": 10,
            },
            "gat": {
                "num_heads": 2, "hidden_dim": 16, "output_dim": 1,
                "dropout": 0.0, "concat_heads": True,
            },
            "threshold": 0.5,
            "training": {"pos_weight": 2.0},
        },
        "mappo": {
            "actor": {"hidden_dims": [64, 32]},
            "critic": {"hidden_dims": [64, 32], "centralised": True},
            "ppo": {
                "total_steps": 100, "num_workers": 1,
                "learning_rate": 3e-4, "eps_clip": 0.2,
                "entropy_coeff": 0.01, "value_loss_coeff": 0.5,
                "gae_lambda": 0.95, "gamma": 0.99,
                "minibatch_size": 32, "num_epochs_per_update": 1,
                "rollout_steps": 64, "gradient_clip_norm": 0.5,
                "normalize_advantages": True,
            },
            "reward": {"alpha": 0.55, "beta": 0.45, "gamma_r": 2.0},
            "checkpoint_interval": 50,
            "eval_episodes": 1,
        },
        "governance": {
            "enabled": True,
            "consumption_cap": {
                "enabled": True, "default_cap_multiplier": 1.5,
                "zone_overrides": {
                    "industrial_zones": [], "industrial_cap_multiplier": 2.0,
                    "critical_zones": [], "critical_cap_multiplier": 1.2,
                },
            },
            "fairness": {"enabled": True, "delta_fair": 0.15, "check_all_pairs": False},
            "emergency": {
                "enabled": True, "default_emergency_fraction": 0.4,
                "always_emergency_zones": [], "critical_emergency_fraction": 0.8,
            },
            "projection": {"method": "l1_search", "valve_delta_resolution": 0.1},
            "audit": {
                "enabled": True, "log_path": "/tmp/conftest_ledger.csv",
                "log_approved": False, "log_overrides": True,
                "justification_codes": {
                    "cap_violation": "C1_CAP", "fairness_violation": "C2_FAIR",
                    "emergency_violation": "C3_EMRG",
                    "approved": "APPROVED", "override": "OVERRIDE",
                },
            },
        },
        "paths": {
            "checkpoints_dir": "/tmp/aquaagent_test_ckpts",
            "logs_dir": "/tmp/aquaagent_test_logs",
            "audit_log": "/tmp/conftest_ledger.csv",
        },
        "seed": 42,
        "num_workers": 1,
        "logging": {"tensorboard": False},
    }


@pytest.fixture
def random_state(small_cfg):
    """A randomly initialised HydraulicState for testing."""
    from src.env.digital_twin import HydraulicState
    rng = np.random.default_rng(0)
    net = small_cfg["network"]
    return HydraulicState(
        flow_rates=rng.random(net["num_edges"]).astype(np.float32),
        pressures=rng.random(net["num_nodes"]).astype(np.float32),
        demands=rng.random(net["num_nodes"]).astype(np.float32),
        leak_indicator=np.zeros(net["num_edges"], dtype=np.float32),
        exogenous=np.zeros(5, dtype=np.float32),
        timestep=0,
    )
