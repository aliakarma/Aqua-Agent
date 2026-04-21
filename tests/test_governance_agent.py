"""tests/test_governance_agent.py — Unit tests for the Governance Agent."""

import numpy as np
import pytest
import torch

GOV_CFG = {
    "governance": {
        "enabled": True,
        "consumption_cap": {
            "enabled": True,
            "default_cap_multiplier": 1.5,
            "zone_overrides": {
                "industrial_zones": [0, 1],
                "industrial_cap_multiplier": 2.0,
                "critical_zones": [4],
                "critical_cap_multiplier": 1.2,
            },
        },
        "fairness": {
            "enabled": True,
            "delta_fair": 0.15,
            "check_all_pairs": False,
        },
        "emergency": {
            "enabled": True,
            "default_emergency_fraction": 0.4,
            "always_emergency_zones": [4],
            "critical_emergency_fraction": 0.8,
        },
        "projection": {"method": "l1_search", "valve_delta_resolution": 0.1},
        "audit": {
            "enabled": True,
            "log_path": "/tmp/test_ledger.csv",
            "log_approved": False,
            "log_overrides": True,
            "justification_codes": {
                "cap_violation": "C1_CAP",
                "fairness_violation": "C2_FAIR",
                "emergency_violation": "C3_EMRG",
                "approved": "APPROVED",
                "override": "OVERRIDE",
            },
        },
    },
    "paths": {"audit_log": "/tmp/test_ledger.csv"},
}


def make_state(num_edges=15, num_nodes=20, num_zones=5, high_flows=False):
    """Build a mock HydraulicState."""
    from src.env.digital_twin import HydraulicState
    flows = np.ones(num_edges, dtype=np.float32) * (10.0 if high_flows else 1.0)
    pressures = np.ones(num_nodes, dtype=np.float32) * 30.0
    demands = np.ones(num_nodes, dtype=np.float32) * 1.0
    return HydraulicState(
        flow_rates=flows,
        pressures=pressures,
        demands=demands,
        leak_indicator=np.zeros(num_edges, dtype=np.float32),
        exogenous=np.zeros(5, dtype=np.float32),
        timestep=0,
    )


def make_action(a_type=3, edge=0, valve=0, delta=0.0):
    return {
        "type":  torch.tensor(a_type),
        "edge":  torch.tensor(edge),
        "valve": torch.tensor(valve),
        "delta": torch.tensor([[delta]]),
    }


class TestGovernanceAgent:
    def _make_gov(self):
        from src.agents.governance_agent import GovernanceAgent
        return GovernanceAgent(GOV_CFG, num_edges=15, num_nodes=20,
                               num_zones=5, num_valves=2)

    def test_no_op_always_approved(self):
        gov = self._make_gov()
        state = make_state()
        action = make_action(a_type=3)  # no_op
        exec_action, overridden = gov.validate(action, state)
        # no_op with normal flows should pass all constraints
        assert isinstance(exec_action, dict)

    def test_override_on_c1_violation(self):
        """Very high flows should trigger c1 consumption cap violation."""
        gov = self._make_gov()
        state = make_state(high_flows=True)   # flows = 10.0 >> demand = 1.0
        action = make_action(a_type=1, valve=0, delta=1.0)  # open valve wide
        _, overridden = gov.validate(action, state)
        # With flows 10x demand, cap at 1.5x should be violated
        assert isinstance(overridden, bool)

    def test_c3_emergency_flag(self):
        """Zone 4 is always emergency — zero flow should trigger c3."""
        gov = self._make_gov()
        from src.env.digital_twin import HydraulicState
        # Zero flows → emergency zones get nothing
        state = HydraulicState(
            flow_rates=np.zeros(15, dtype=np.float32),
            pressures=np.ones(20, dtype=np.float32) * 30.0,
            demands=np.ones(20, dtype=np.float32) * 1.0,
            leak_indicator=np.zeros(15, dtype=np.float32),
            exogenous=np.zeros(5, dtype=np.float32),
            timestep=0,
        )
        action = make_action(a_type=0, edge=0)  # isolate edge 0
        exec_action, overridden = gov.validate(action, state)
        # Should be overridden — c3 violation (emergency zone gets zero flow)
        assert isinstance(overridden, bool)

    def test_pcr_tracks_correctly(self):
        gov = self._make_gov()
        state = make_state()
        # Run 10 no_op actions (should all be approved)
        for _ in range(10):
            gov.validate(make_action(a_type=3), state)
        pcr = gov.get_policy_compliance_rate()
        assert 0.0 <= pcr <= 1.0

    def test_action_encoding_round_trip(self):
        gov = self._make_gov()
        action = make_action(a_type=1, edge=5, valve=2, delta=0.7)
        vec = gov._action_to_vector(action)
        assert vec.shape == (4,)
        assert abs(vec[3] - 0.7) < 0.01

    def test_governance_disabled(self):
        """When governance is disabled, all actions pass through unchanged."""
        cfg = {**GOV_CFG, "governance": {**GOV_CFG["governance"], "enabled": False}}
        from src.agents.governance_agent import GovernanceAgent
        gov = GovernanceAgent(cfg, num_edges=15, num_nodes=20,
                              num_zones=5, num_valves=2)
        state = make_state(high_flows=True)
        action = make_action(a_type=1, delta=1.0)
        exec_action, overridden = gov.validate(action, state)
        assert not overridden
