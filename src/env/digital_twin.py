"""
src/env/digital_twin.py
-----------------------
High-fidelity hydraulic digital twin for AquaAgent.

Wraps EPANET 2.2 via the `epyt` Python library.
Paper Section 3.5:
  "The digital twin is implemented using EPANET 2.2 as the hydraulic solver,
   interfaced with Python via the epyt library."

Network: 48 demand zones, 213 pipe segments, 18 PRVs, 4 tanks, 2 reservoirs.

The digital twin is used for:
  1. Safe offline training of both the ADA and Decision Agent.
  2. Evaluation against held-out test episodes.
  3. Scalability analysis (parameterised by network size).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger
from src.env.leak_injector import LeakInjector, LeakEvent
from src.env.sensor_noise import SensorNoiseModel

logger = get_logger("digital_twin")


@dataclass
class HydraulicState:
    """
    Full system state at time t.
    Paper Equation (1): s_t = (f_t, p_t, q_t, l_t, e_t)
    """
    flow_rates: np.ndarray       # f_t  ∈ R^|E|  (L/s)
    pressures: np.ndarray        # p_t  ∈ R^|V|  (m head)
    demands: np.ndarray          # q_t  ∈ R^|V|  (L/s)
    leak_indicator: np.ndarray   # l_t  ∈ {0,1}^|E|  (latent, not observable)
    exogenous: np.ndarray        # e_t  ∈ R^d  (time-of-day, season, temp)
    timestep: int = 0


@dataclass
class SensorReading:
    """Noisy sensor observations emitted to the Monitoring Agent."""
    noisy_flows: np.ndarray      # f_t + ε_flow
    noisy_pressures: np.ndarray  # p_t + ε_pressure
    noisy_demands: np.ndarray    # q_t  (zone-level meters)
    timestep: int = 0


class DigitalTwin:
    """
    EPANET-backed hydraulic digital twin.

    Usage:
        dt = DigitalTwin(cfg)
        dt.reset()
        for t in range(horizon):
            obs = dt.get_sensor_readings()
            state = dt.get_state()
            dt.step(action)
    """

    def __init__(self, cfg: dict, seed: int = 42):
        """
        Args:
            cfg:  Full configuration dict (merged from YAML configs).
            seed: Random seed for leak injection and demand noise.
        """
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        net_cfg = cfg.get("network", {})
        self.num_nodes: int = net_cfg.get("num_nodes", 261)
        self.num_edges: int = net_cfg.get("num_edges", 213)
        self.num_zones: int = net_cfg.get("num_demand_zones", 48)
        self.num_prv: int = net_cfg.get("num_prv", 18)
        self.inp_file: str = net_cfg.get("inp_file", "data/raw/aquaagent_dma.inp")

        # Simulation state
        self._epanet = None           # epyt EpanetSimulator instance
        self._current_step: int = 0
        self._current_state: Optional[HydraulicState] = None

        # Topology arrays (populated by reset())
        self.pipe_from: Optional[np.ndarray] = None
        self.pipe_to: Optional[np.ndarray] = None
        self.node_types: Optional[np.ndarray] = None

        # Sub-components
        self.leak_injector = LeakInjector(
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            cfg=net_cfg.get("leak", {}),
            seed=seed + 1,
        )
        self.noise_model = SensorNoiseModel(
            num_edges=self.num_edges,
            num_nodes=self.num_nodes,
            pressure_sigma=net_cfg.get("sensor_noise", {}).get("pressure_sigma", 0.05),
            flow_sigma=net_cfg.get("sensor_noise", {}).get("flow_sigma", 0.01),
            seed=seed + 2,
        )

        # Try to load epyt; gracefully fall back to mock simulation unless
        # strict mode is enabled (cfg.reproducibility.strict_epanet: true).
        # FIX-04 / Reviewer 1 Issue 4 + Reviewer 2 Issue 4.
        self._epyt_available = self._try_import_epyt()
        self._strict_epanet: bool = bool(
            cfg.get("reproducibility", {}).get("strict_epanet", False)
        )
        if not self._epyt_available:
            if self._strict_epanet:
                raise RuntimeError(
                    "strict_epanet=True but epyt is not installed. "
                    "Install with: pip install epyt\n"
                    "To run in mock mode set reproducibility.strict_epanet: false "
                    "in configs/default.yaml."
                )
            logger.warning(
                "epyt not available — using stochastic mock simulation. "
                "Install epyt for full EPANET integration: pip install epyt\n"
                "NOTE: All metrics produced in mock mode are NOT equivalent to "
                "EPANET hydraulic results (see README §Reproducibility)."
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> HydraulicState:
        """Reset the digital twin to initial conditions for a new episode."""
        self._current_step = 0
        self.leak_injector.reset()
        self.noise_model.reset()

        epanet_ready = self._epyt_available and os.path.exists(self.inp_file)
        if not epanet_ready and self._strict_epanet:
            raise RuntimeError(
                f"strict_epanet=True but EPANET prerequisites missing: "
                f"epyt={self._epyt_available}, "
                f"inp_exists={os.path.exists(self.inp_file)} (expected at {self.inp_file}).\n"
                f"Either provide the network file or set "
                f"reproducibility.strict_epanet: false in configs/default.yaml.\n"
                f"See data/raw/README.md for instructions on obtaining the network file."
            )

        if epanet_ready:
            self._current_state = self._epanet_reset()
        else:
            self._current_state = self._mock_reset()

        return self._current_state

    def step(self, action: dict) -> Tuple[HydraulicState, bool]:
        """
        Advance the simulation by one time step, applying the governance-
        approved action to the hydraulic network.

        Args:
            action: Dict with keys 'type', 'edge'/'valve'/'zone', 'delta'.
                    Produced by GovernanceAgent.validate().

        Returns:
            state:   Updated HydraulicState after hydraulic solve.
            done:    True if episode length is reached.
        """
        self._current_step += 1

        # Determine which edges are isolated by the current action (used by
        # leak_injector to stop loss accumulation on those edges — FIX-11).
        isolated_edges: set = set()
        if action is not None and action.get("type") == "isolate":
            edge = action.get("edge")
            if edge is not None and 0 <= int(edge) < self.num_edges:
                isolated_edges.add(int(edge))

        # Apply action to EPANET model (or mock)
        epanet_ready = self._epyt_available and os.path.exists(self.inp_file)
        if epanet_ready:
            self._apply_action_epanet(action)
            self._current_state = self._epanet_step()
        else:
            self._current_state = self._mock_step(action)

        # Inject / update leak events
        self.leak_injector.step(self._current_state, isolated_edges=isolated_edges)
        # Update leak indicator in state
        self._current_state.leak_indicator = self.leak_injector.get_leak_indicator()
        self._current_state.timestep = self._current_step

        done = self._current_step >= self.cfg.get("network", {}).get(
            "simulation", {}
        ).get("duration_days", 365) * 86400

        return self._current_state, done

    def get_state(self) -> HydraulicState:
        """Return the current (true) hydraulic state."""
        if self._current_state is None:
            raise RuntimeError("Call reset() before get_state().")
        return self._current_state

    def get_sensor_readings(self) -> SensorReading:
        """
        Return noisy sensor observations (the Monitoring Agent's inputs).
        True state is perturbed by sensor noise model ε ~ N(0, Σ_i).
        """
        state = self.get_state()
        noisy_f, noisy_p = self.noise_model.apply(
            state.flow_rates, state.pressures
        )
        return SensorReading(
            noisy_flows=noisy_f,
            noisy_pressures=noisy_p,
            noisy_demands=state.demands.copy(),
            timestep=self._current_step,
        )

    def get_topology(self) -> dict:
        """Return network topology arrays for graph construction."""
        if self.pipe_from is None:
            self._init_topology()
        return {
            "pipe_from": self.pipe_from,
            "pipe_to": self.pipe_to,
            "node_types": self.node_types,
        }

    # ------------------------------------------------------------------
    # EPANET integration (requires epyt + .inp file)
    # ------------------------------------------------------------------

    def _try_import_epyt(self) -> bool:
        try:
            import epyt  # noqa: F401
            return True
        except ImportError:
            return False

    def _epanet_reset(self) -> HydraulicState:
        """Initialise epyt simulator and run initial hydraulic solve."""
        import epyt
        if self._epanet is not None:
            try:
                self._epanet.unload()
            except Exception:
                pass
        self._epanet = epyt.epanet(self.inp_file)
        self._epanet.setTimeSimulationDuration(1)  # 1-second step
        results = self._epanet.getComputedHydraulicTimeSeries()
        self._init_topology_from_epanet()
        return self._results_to_state(results, step=0)

    def _epanet_step(self) -> HydraulicState:
        """Advance epyt by one hydraulic time step."""
        results = self._epanet.getComputedHydraulicTimeSeries()
        return self._results_to_state(results, step=self._current_step)

    def _apply_action_epanet(self, action: dict) -> None:
        """Translate a governance-approved action to EPANET control commands."""
        if action is None or action.get("type") == "no_op":
            return
        action_type = action.get("type")
        if action_type == "adjust_valve":
            valve_id = action.get("valve")
            delta = action.get("delta", 0.0)
            # delta ∈ [-1, 1] mapped to PRV setting change
            if valve_id is not None and self._epanet is not None:
                try:
                    current = self._epanet.getLinkSettings(valve_id)
                    new_setting = float(np.clip(current + delta * 10.0, 0, 100))
                    self._epanet.setLinkSettings(valve_id, new_setting)
                except Exception as e:
                    logger.debug(f"Valve adjustment failed: {e}")
        elif action_type == "isolate":
            edge_id = action.get("edge")
            if edge_id is not None and self._epanet is not None:
                try:
                    self._epanet.setLinkStatus(edge_id, 0)  # 0 = closed
                except Exception as e:
                    logger.debug(f"Pipe isolation failed: {e}")

    def _results_to_state(self, results, step: int) -> HydraulicState:
        """Convert epyt results to HydraulicState."""
        t = min(step, results.Flow.shape[0] - 1)
        flow_rates = np.abs(results.Flow[t]).astype(np.float32)
        pressures = results.Pressure[t].astype(np.float32)
        demands = results.Demand[t].astype(np.float32)
        exog = self._build_exogenous(step)
        return HydraulicState(
            flow_rates=flow_rates[:self.num_edges],
            pressures=pressures[:self.num_nodes],
            demands=demands[:self.num_nodes],
            leak_indicator=np.zeros(self.num_edges, dtype=np.float32),
            exogenous=exog,
            timestep=step,
        )

    def _init_topology_from_epanet(self) -> None:
        """Extract topology arrays from epyt."""
        conn = self._epanet.getLinkConnectivityMatrix()
        self.pipe_from = np.array([c[0] - 1 for c in conn])[:self.num_edges]
        self.pipe_to = np.array([c[1] - 1 for c in conn])[:self.num_edges]
        self.node_types = np.zeros(self.num_nodes, dtype=int)

    # ------------------------------------------------------------------
    # Mock simulation (used when epyt / .inp file is unavailable)
    # ------------------------------------------------------------------

    def _mock_reset(self) -> HydraulicState:
        """Initialise random baseline hydraulic state."""
        self._init_topology()
        flows = self.rng.exponential(scale=1.0, size=self.num_edges).astype(np.float32)
        pressures = self.rng.uniform(20, 80, size=self.num_nodes).astype(np.float32)
        demands = self.rng.exponential(scale=0.5, size=self.num_nodes).astype(np.float32)
        return HydraulicState(
            flow_rates=flows,
            pressures=pressures,
            demands=demands,
            leak_indicator=np.zeros(self.num_edges, dtype=np.float32),
            exogenous=self._build_exogenous(0),
            timestep=0,
        )

    def _mock_step(self, action: dict) -> HydraulicState:
        """
        Stochastic mock: carry forward previous state with small perturbations
        plus demand cycles and action effects.
        """
        prev = self._current_state
        hour_of_day = (self._current_step % 86400) / 3600.0
        # Sinusoidal demand cycle (daily pattern)
        demand_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (hour_of_day - 7) / 24)

        flows = np.clip(
            prev.flow_rates * demand_factor
            + self.rng.normal(0, 0.05, size=self.num_edges),
            0, None,
        ).astype(np.float32)
        pressures = np.clip(
            prev.pressures + self.rng.normal(0, 0.1, size=self.num_nodes),
            0, None,
        ).astype(np.float32)
        demands = (prev.demands * demand_factor).astype(np.float32)

        # Apply action effects (simplified)
        if action is not None:
            if action.get("type") == "adjust_valve":
                delta = action.get("delta", 0.0)
                flows = np.clip(flows * (1 + 0.1 * delta), 0, None).astype(np.float32)
            elif action.get("type") == "isolate":
                edge = action.get("edge")
                if edge is not None and 0 <= edge < self.num_edges:
                    flows[edge] = 0.0

        return HydraulicState(
            flow_rates=flows,
            pressures=pressures,
            demands=demands,
            leak_indicator=prev.leak_indicator.copy(),
            exogenous=self._build_exogenous(self._current_step),
            timestep=self._current_step,
        )

    def _init_topology(self) -> None:
        """Set up synthetic topology (used in mock mode)."""
        from src.utils.graph_utils import synthetic_network_topology
        topo = synthetic_network_topology(
            self.num_nodes, self.num_edges, seed=self.seed
        )
        self.pipe_from = topo["pipe_from"]
        self.pipe_to = topo["pipe_to"]
        self.node_types = topo["node_types"]

    def _build_exogenous(self, step: int) -> np.ndarray:
        """
        Build exogenous context vector e_t ∈ R^d (paper Eq. 1).
        Encodes: time_sin, time_cos (daily cycle), weekday, season_sin, season_cos.
        """
        sec_of_day = step % 86400
        sec_of_year = step % (365 * 86400)
        return np.array([
            np.sin(2 * np.pi * sec_of_day / 86400),
            np.cos(2 * np.pi * sec_of_day / 86400),
            (step // 86400) % 7 / 6.0,               # Weekday (normalised)
            np.sin(2 * np.pi * sec_of_year / (365 * 86400)),
            np.cos(2 * np.pi * sec_of_year / (365 * 86400)),
        ], dtype=np.float32)
