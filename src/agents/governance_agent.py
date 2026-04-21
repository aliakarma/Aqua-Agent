"""
src/agents/governance_agent.py
--------------------------------
Governance Agent (GA) — Post-hoc policy constraint enforcement.

Paper Section 3.9, Equations (5)-(8):
  c1: q_{t,z}^consumed ≤ q_max^(z)                     [consumption cap]
  c2: |SR_z − SR_z'| ≤ δ_fair  ∀z,z'                   [equity/fairness]
      where SR_z = q_{t,z}^delivered / q_{t,z}^demanded
  c3: q_{t,z}^delivered ≥ q_emrg^(z)  if σ_z = 1       [emergency supply]

  Override rule (Equation 8):
    a* = a_t                             if a_t ∈ Feasible(s_t)
    a* = argmin_{a ∈ Feasible} ‖a − a_t‖₁  otherwise

All governance decisions are logged to an append-only audit ledger.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils.audit_ledger import AuditLedger
from src.utils.logger import get_logger
from src.models.ppo_mlp import action_dict_to_str

logger = get_logger("governance_agent")


class GovernanceAgent:
    """
    Rule-based governance post-processor.

    Receives the DA's candidate action a_t, validates it against the three
    constraint classes (c1, c2, c3), and either passes it through or computes
    the nearest feasible action via L1-projection.
    """

    def __init__(self, cfg: dict, num_edges: int, num_nodes: int,
                 num_zones: int, num_valves: int):
        """
        Args:
            cfg:        Full merged config dict.
            num_edges:  Number of pipe edges.
            num_nodes:  Number of network nodes.
            num_zones:  Number of demand zones.
            num_valves: Number of PRVs.
        """
        self.cfg = cfg
        self.num_edges  = num_edges
        self.num_nodes  = num_nodes
        self.num_zones  = num_zones
        self.num_valves = num_valves

        gov_cfg = cfg.get("governance", {})
        self.enabled = gov_cfg.get("enabled", True)

        # ── c1: Consumption cap ──
        cap_cfg = gov_cfg.get("consumption_cap", {})
        self.cap_enabled = cap_cfg.get("enabled", True)
        self.default_cap_mult = cap_cfg.get("default_cap_multiplier", 1.5)
        self.industrial_zones = set(cap_cfg.get("zone_overrides", {}).get("industrial_zones", []))
        self.industrial_cap = cap_cfg.get("zone_overrides", {}).get("industrial_cap_multiplier", 2.0)
        self.critical_zones = set(cap_cfg.get("zone_overrides", {}).get("critical_zones", []))
        self.critical_cap = cap_cfg.get("zone_overrides", {}).get("critical_cap_multiplier", 1.2)

        # ── c2: Fairness ──
        fair_cfg = gov_cfg.get("fairness", {})
        self.fair_enabled = fair_cfg.get("enabled", True)
        self.delta_fair = fair_cfg.get("delta_fair", 0.15)
        self.check_all_pairs = fair_cfg.get("check_all_pairs", False)

        # ── c3: Emergency ──
        emrg_cfg = gov_cfg.get("emergency", {})
        self.emrg_enabled = emrg_cfg.get("enabled", True)
        self.default_emrg_frac = emrg_cfg.get("default_emergency_fraction", 0.4)
        self.always_emrg_zones = set(emrg_cfg.get("always_emergency_zones", [13, 14, 15]))
        self.critical_emrg_frac = emrg_cfg.get("critical_emergency_fraction", 0.8)

        # Valve delta resolution for L1 search
        proj_cfg = gov_cfg.get("projection", {})
        self.valve_resolution = proj_cfg.get("valve_delta_resolution", 0.1)

        # Audit ledger
        audit_cfg = gov_cfg.get("audit", {})
        self.audit_enabled = audit_cfg.get("enabled", True)
        if self.audit_enabled:
            self.ledger = AuditLedger(
                path=cfg.get("paths", {}).get("audit_log", "logs/audit_ledger.csv"),
                log_approved=audit_cfg.get("log_approved", False),
                log_overrides=audit_cfg.get("log_overrides", True),
            )
        else:
            self.ledger = None

        # Justification codes
        codes = gov_cfg.get("audit", {}).get("justification_codes", {})
        self.CODE_C1 = codes.get("cap_violation", "C1_CAP")
        self.CODE_C2 = codes.get("fairness_violation", "C2_FAIR")
        self.CODE_C3 = codes.get("emergency_violation", "C3_EMRG")
        self.CODE_OK = codes.get("approved", "APPROVED")
        self.CODE_OV = codes.get("override", "OVERRIDE")

        # Running statistics
        self._step = 0

    def validate(self, action: dict, state,
                 base_demands: Optional[np.ndarray] = None) -> Tuple[dict, bool]:
        """
        Validate candidate action against governance constraints.

        Args:
            action:       DA's candidate action dict.
            state:        Current HydraulicState.
            base_demands: Optional baseline demands for computing caps.

        Returns:
            (executed_action, overridden)
            executed_action is either the original action (approved) or the
            nearest feasible action (override).
        """
        self._step += 1

        if not self.enabled:
            return action, False

        # Simulate the hydraulic effect of the action on zone flows
        zone_flows    = self._estimate_zone_flows(action, state)
        zone_demands  = self._get_zone_demands(state, base_demands)
        emrg_flags    = self._get_emergency_flags()

        # ── Check all three constraint classes ──
        c1_ok, c1_violations = self._check_c1(zone_flows, zone_demands)
        c2_ok, c2_violations = self._check_c2(zone_flows, zone_demands)
        c3_ok, c3_violations = self._check_c3(zone_flows, zone_demands, emrg_flags)

        feasible = c1_ok and c2_ok and c3_ok

        if feasible:
            self._log(action, action, overridden=False,
                      code=self.CODE_OK, c1=False, c2=False, c3=False)
            return action, False

        # ── L1-projection: find nearest feasible action ──
        logger.debug(
            f"Step {self._step}: action infeasible "
            f"(c1={not c1_ok}, c2={not c2_ok}, c3={not c3_ok}) — projecting."
        )
        best_action = self._l1_project(action, state, zone_demands, emrg_flags)
        code = self._build_code(c1_ok, c2_ok, c3_ok)
        self._log(action, best_action, overridden=True,
                  code=code,
                  c1=not c1_ok, c2=not c2_ok, c3=not c3_ok)

        return best_action, True

    def get_policy_compliance_rate(self) -> float:
        """
        Return Policy Compliance Rate (PCR) — fraction of steps where
        the DA's action was approved without override.

        Paper Table 1: AquaAgent PCR = 90.9% (override rate ≈ 9.1%)
        """
        if self.ledger is None:
            return 1.0
        summary = self.ledger.summary()
        total = summary["total_evaluated"]
        if total == 0:
            return 1.0
        return 1.0 - summary["override_rate"]

    # ------------------------------------------------------------------
    # Constraint checks
    # ------------------------------------------------------------------

    def _check_c1(self, zone_flows: np.ndarray,
                  zone_demands: np.ndarray) -> Tuple[bool, List[int]]:
        """
        c1: Consumption cap per zone.
        q_{t,z} ≤ q_max^(z)  where q_max^(z) = cap_multiplier × demand_z
        """
        if not self.cap_enabled:
            return True, []

        violations = []
        for z in range(self.num_zones):
            if z >= len(zone_flows) or z >= len(zone_demands):
                break
            d_z = zone_demands[z]
            if z in self.critical_zones:
                cap = self.critical_cap * d_z
            elif z in self.industrial_zones:
                cap = self.industrial_cap * d_z
            else:
                cap = self.default_cap_mult * d_z

            if zone_flows[z] > cap + 1e-6:
                violations.append(z)

        return len(violations) == 0, violations

    def _check_c2(self, zone_flows: np.ndarray,
                  zone_demands: np.ndarray) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        c2: Fairness across zones.
        |SR_z − SR_z'| ≤ δ_fair  ∀z,z'
        where SR_z = delivered / demanded
        """
        if not self.fair_enabled or self.num_zones < 2:
            return True, []

        # Compute service ratios
        sr = np.zeros(self.num_zones)
        for z in range(min(self.num_zones, len(zone_flows), len(zone_demands))):
            d = zone_demands[z]
            sr[z] = zone_flows[z] / d if d > 1e-8 else 0.0

        violations = []
        if self.check_all_pairs:
            for z1 in range(self.num_zones):
                for z2 in range(z1 + 1, self.num_zones):
                    if abs(sr[z1] - sr[z2]) > self.delta_fair:
                        violations.append((z1, z2))
        else:
            # Compare each zone to median (cheaper, paper doesn't specify)
            median_sr = float(np.median(sr))
            for z in range(self.num_zones):
                if abs(sr[z] - median_sr) > self.delta_fair:
                    violations.append((z, -1))

        return len(violations) == 0, violations

    def _check_c3(self, zone_flows: np.ndarray,
                  zone_demands: np.ndarray,
                  emrg_flags: np.ndarray) -> Tuple[bool, List[int]]:
        """
        c3: Emergency supply guarantee.
        q_{t,z}^delivered ≥ q_emrg^(z)  if σ_z = 1
        """
        if not self.emrg_enabled:
            return True, []

        violations = []
        for z in range(min(self.num_zones, len(zone_flows), len(zone_demands))):
            if not emrg_flags[z]:
                continue
            d_z = zone_demands[z]
            frac = self.critical_emrg_frac if z in self.always_emrg_zones \
                   else self.default_emrg_frac
            threshold = frac * d_z
            if zone_flows[z] < threshold - 1e-6:
                violations.append(z)

        return len(violations) == 0, violations

    # ------------------------------------------------------------------
    # L1-projection (paper Equation 8)
    # ------------------------------------------------------------------

    def _l1_project(self, action: dict, state,
                    zone_demands: np.ndarray,
                    emrg_flags: np.ndarray) -> dict:
        """
        Find nearest feasible action in L1 distance.

        Strategy:
          1. Try no_op first (always safe as reference).
          2. Search over alternative actions from a small candidate set.
          3. Return the candidate with smallest L1 distance to original.

        (Assumption A7: brute-force over discrete action space.)
        """
        # Build candidate set
        candidates = self._generate_candidates(action)

        best_action = {"type": torch.tensor(3), "edge": torch.tensor(0),
                       "valve": torch.tensor(0),
                       "delta": torch.tensor([[0.0]])}  # no_op
        best_dist = float("inf")

        orig_encoding = self._action_to_vector(action)

        for candidate in candidates:
            zone_flows = self._estimate_zone_flows(candidate, state)
            c1_ok, _ = self._check_c1(zone_flows, zone_demands)
            c2_ok, _ = self._check_c2(zone_flows, zone_demands)
            c3_ok, _ = self._check_c3(zone_flows, zone_demands, emrg_flags)

            if c1_ok and c2_ok and c3_ok:
                dist = float(np.sum(np.abs(
                    self._action_to_vector(candidate) - orig_encoding
                )))
                if dist < best_dist:
                    best_dist = dist
                    best_action = candidate

        return best_action

    def _generate_candidates(self, original_action: dict) -> List[dict]:
        """
        Generate a small set of action candidates for L1-projection search.
        Includes no_op plus nearby adjust_valve actions at different δ values.
        """
        candidates = []

        # no_op is always a valid candidate
        candidates.append({
            "type": torch.tensor(3), "edge": original_action["edge"],
            "valve": original_action["valve"], "delta": torch.tensor([[0.0]])
        })

        # Try scaled-down versions of adjust_valve
        orig_type = int(original_action["type"].item()
                        if torch.is_tensor(original_action["type"])
                        else original_action["type"])
        if orig_type == 1:  # adjust_valve
            for scale in [0.5, 0.25, -0.25, -0.5]:
                delta_val = float(
                    original_action["delta"].item()
                    if torch.is_tensor(original_action["delta"])
                    else original_action["delta"]
                ) * scale
                delta_val = float(np.clip(delta_val, -1.0, 1.0))
                candidates.append({
                    "type": torch.tensor(1),
                    "edge": original_action["edge"],
                    "valve": original_action["valve"],
                    "delta": torch.tensor([[delta_val]])
                })

        return candidates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_zone_flows(self, action: dict, state) -> np.ndarray:
        """
        Approximate the per-zone flows after applying `action`.
        A full hydraulic re-solve is expensive; we use a linear approximation.
        """
        # Aggregate edge flows to zones (round-robin zone assignment)
        zone_flows = np.zeros(self.num_zones, dtype=np.float32)
        flows = state.flow_rates.copy()

        # Apply action effect
        a_type = int(action["type"].item() if torch.is_tensor(action["type"])
                     else action["type"])
        if a_type == 0:   # isolate edge
            edge = int(action["edge"].item() if torch.is_tensor(action["edge"])
                       else action["edge"])
            if 0 <= edge < len(flows):
                flows[edge] = 0.0
        elif a_type == 1:  # adjust_valve
            delta = float(action["delta"].item() if torch.is_tensor(action["delta"])
                         else action["delta"])
            flows = np.clip(flows * (1 + 0.1 * float(delta)), 0, None).astype(np.float32)

        # Aggregate flows to zones
        edges_per_zone = max(1, len(flows) // self.num_zones)
        for z in range(self.num_zones):
            start = z * edges_per_zone
            end = start + edges_per_zone
            zone_flows[z] = float(flows[start:end].sum())

        return zone_flows

    def _get_zone_demands(self, state,
                          base_demands: Optional[np.ndarray]) -> np.ndarray:
        """Aggregate node-level demands to zone level."""
        demands = state.demands if base_demands is None else base_demands
        zone_demands = np.zeros(self.num_zones, dtype=np.float32)
        nodes_per_zone = max(1, len(demands) // self.num_zones)
        for z in range(self.num_zones):
            start = z * nodes_per_zone
            end = start + nodes_per_zone
            zone_demands[z] = float(demands[start:end].sum())
        return zone_demands

    def _get_emergency_flags(self) -> np.ndarray:
        """Return binary emergency flags σ_z for all zones."""
        flags = np.zeros(self.num_zones, dtype=bool)
        for z in self.always_emrg_zones:
            if z < self.num_zones:
                flags[z] = True
        return flags

    def _action_to_vector(self, action: dict) -> np.ndarray:
        """Encode action as a float vector for L1 distance computation."""
        a_type  = float(action["type"].item() if torch.is_tensor(action["type"]) else action["type"])
        a_edge  = float(action["edge"].item() if torch.is_tensor(action["edge"]) else action["edge"])
        a_valve = float(action["valve"].item() if torch.is_tensor(action["valve"]) else action["valve"])
        a_delta = float(action["delta"].item() if torch.is_tensor(action["delta"]) else action["delta"])
        return np.array([a_type, a_edge, a_valve, a_delta], dtype=np.float32)

    def _build_code(self, c1_ok: bool, c2_ok: bool, c3_ok: bool) -> str:
        parts = []
        if not c1_ok:
            parts.append(self.CODE_C1)
        if not c2_ok:
            parts.append(self.CODE_C2)
        if not c3_ok:
            parts.append(self.CODE_C3)
        return "+".join(parts) if parts else self.CODE_OV

    def _log(self, proposed: dict, executed: dict,
             overridden: bool, code: str,
             c1: bool, c2: bool, c3: bool) -> None:
        if self.ledger is not None:
            self.ledger.append(
                step=self._step,
                proposed_action=action_dict_to_str(proposed),
                executed_action=action_dict_to_str(executed),
                overridden=overridden,
                justification_code=code,
                c1_violated=c1,
                c2_violated=c2,
                c3_violated=c3,
            )
