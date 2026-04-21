"""
src/env/leak_injector.py
------------------------
Stochastic leak injection engine for the AquaAgent digital twin.

Paper Section 3.5:
  "Leak events are injected by reducing the emitter coefficient at randomly
   selected nodes; leak magnitude follows a log-normal distribution calibrated
   to the IWA infrastructure leakage index."

Paper Section 4.1:
  "Leak events were injected on 180 of 365 days (49%), with between 1 and 5
   simultaneous leaks per episode, covering burst, background, and slow-onset
   leak profiles."

The leak state l_t ∈ {0,1}^|E| is a latent variable — it is NOT directly
observable by any agent; only its hydraulic effects are observed via sensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("leak_injector")


@dataclass
class LeakEvent:
    """Represents a single active leak event."""
    edge_idx: int           # Which pipe edge has the leak
    node_idx: int           # Emitter node (endpoint of edge)
    profile: str            # "burst" | "background" | "slow_onset"
    magnitude: float        # Full leak rate (L/s) from log-normal distribution
    start_step: int         # Time step when leak began
    duration_steps: int     # How many steps this leak lasts
    current_step: int = 0   # Steps since leak started

    @property
    def current_magnitude(self) -> float:
        """
        Return the current effective leak magnitude (accounts for slow-onset
        ramp-up and natural decay for burst events).
        """
        if self.profile == "burst":
            # Burst: full magnitude immediately, then decays exponentially
            return self.magnitude * np.exp(-0.001 * self.current_step)
        elif self.profile == "background":
            # Background: constant low-level loss
            return self.magnitude
        elif self.profile == "slow_onset":
            # Slow-onset: linear ramp up over growth_steps to full magnitude
            growth_steps = int(6 * 3600)   # 6 hours at 1Hz (config A10)
            ramp = min(1.0, self.current_step / max(growth_steps, 1))
            return self.magnitude * ramp
        return self.magnitude

    @property
    def is_expired(self) -> bool:
        return self.current_step >= self.duration_steps


class LeakInjector:
    """
    Manages stochastic leak injection across a simulation episode.

    At each new day (every 86,400 steps), decides probabilistically whether
    to inject between 1 and max_simultaneous leaks, sampling their profile,
    location, and magnitude from calibrated distributions.
    """

    def __init__(self, num_nodes: int, num_edges: int,
                 cfg: dict, seed: int = 43):
        """
        Args:
            num_nodes: Total nodes in the network.
            num_edges: Total pipe edges in the network.
            cfg:       Leak sub-config from network.yaml.
            seed:      Random seed.
        """
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        # Configuration parameters
        self.event_probability: float = cfg.get("event_probability", 0.493)
        self.max_simultaneous: int = cfg.get("max_simultaneous", 5)
        self.min_simultaneous: int = cfg.get("min_simultaneous", 1)
        self.profiles: List[str] = cfg.get("profiles",
                                           ["burst", "background", "slow_onset"])
        self.profile_weights: List[float] = cfg.get(
            "profile_weights", [0.33, 0.34, 0.33]
        )
        self.lognormal_mu: float = cfg.get("lognormal_mu", -1.5)
        self.lognormal_sigma: float = cfg.get("lognormal_sigma", 0.8)

        # Duration parameters (in steps at 1Hz)
        self.burst_duration_mean_steps = int(
            cfg.get("burst_duration_mean", 2.0) * 3600
        )
        self.background_duration_mean_steps = int(
            cfg.get("background_duration_mean", 48.0) * 3600
        )
        self.slow_onset_growth_steps = int(
            cfg.get("slow_onset_growth_hours", 6.0) * 3600
        )

        # Active leaks
        self._active_leaks: List[LeakEvent] = []
        self._current_step: int = 0
        self._leak_indicator: np.ndarray = np.zeros(num_edges, dtype=np.float32)

        # Statistics
        self.total_injected: int = 0
        self.total_loss_L: float = 0.0   # Cumulative leak volume

    def reset(self) -> None:
        """Clear all active leaks for a new episode."""
        self._active_leaks.clear()
        self._current_step = 0
        self._leak_indicator = np.zeros(self.num_edges, dtype=np.float32)
        self.total_injected = 0
        self.total_loss_L = 0.0

    def step(self, state) -> None:
        """
        Advance leak dynamics by one time step.
        Called after each hydraulic solve step.

        Args:
            state: Current HydraulicState (used to check if new day started).
        """
        self._current_step += 1

        # Advance existing leaks
        expired = []
        for leak in self._active_leaks:
            leak.current_step += 1
            self.total_loss_L += leak.current_magnitude  # 1-second accumulation
            if leak.is_expired:
                expired.append(leak)

        for leak in expired:
            self._active_leaks.remove(leak)
            logger.debug(f"Leak expired on edge {leak.edge_idx}")

        # Daily injection decision (once per 86400 steps)
        if self._current_step % 86400 == 0:
            self._try_inject_daily()

        # Update binary leak indicator
        self._update_indicator()

    def get_leak_indicator(self) -> np.ndarray:
        """Return binary leak indicator l_t ∈ {0,1}^|E|."""
        return self._leak_indicator.copy()

    def get_active_leaks(self) -> List[LeakEvent]:
        """Return list of currently active LeakEvent objects."""
        return list(self._active_leaks)

    def get_total_loss(self) -> float:
        """Return cumulative water loss volume in Litres."""
        return self.total_loss_L

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_inject_daily(self) -> None:
        """Probabilistically inject leaks for the upcoming day."""
        if self.rng.random() > self.event_probability:
            return
        n_leaks = int(self.rng.integers(
            self.min_simultaneous, self.max_simultaneous + 1
        ))
        for _ in range(n_leaks):
            self._inject_single()

    def _inject_single(self) -> None:
        """Sample and inject one leak event."""
        # Sample edge location
        edge_idx = int(self.rng.integers(0, self.num_edges))
        # Map edge to node (use source node of edge)
        node_idx = edge_idx % self.num_nodes

        # Sample profile
        profile = self.rng.choice(self.profiles, p=self._normalised_weights())

        # Sample magnitude from log-normal (IWA calibrated)
        magnitude = float(np.exp(
            self.lognormal_mu + self.lognormal_sigma * self.rng.standard_normal()
        ))
        magnitude = float(np.clip(magnitude, 0.01, 10.0))  # L/s

        # Sample duration
        if profile == "burst":
            duration = max(1, int(self.rng.exponential(
                scale=self.burst_duration_mean_steps
            )))
        elif profile == "slow_onset":
            duration = max(1, int(self.rng.exponential(
                scale=self.background_duration_mean_steps
            )))
        else:
            duration = max(1, int(self.rng.exponential(
                scale=self.background_duration_mean_steps
            )))

        leak = LeakEvent(
            edge_idx=edge_idx,
            node_idx=node_idx,
            profile=profile,
            magnitude=magnitude,
            start_step=self._current_step,
            duration_steps=duration,
        )
        self._active_leaks.append(leak)
        self.total_injected += 1
        logger.debug(
            f"Injected {profile} leak on edge {edge_idx}: "
            f"{magnitude:.3f} L/s for {duration} steps"
        )

    def _normalised_weights(self) -> np.ndarray:
        w = np.array(self.profile_weights, dtype=float)
        return w / w.sum()

    def _update_indicator(self) -> None:
        """Refresh the binary leak indicator array."""
        self._leak_indicator[:] = 0
        for leak in self._active_leaks:
            if 0 <= leak.edge_idx < self.num_edges:
                self._leak_indicator[leak.edge_idx] = 1
