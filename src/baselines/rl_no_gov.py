"""
src/baselines/rl_no_gov.py
---------------------------
DEPRECATED — this file is retained for historical reference only.

FIX-10 (Reviewer 1 Issue M3): RLNoGovAgent is dead code.  B3 evaluation is
handled directly in src/evaluation/evaluate.py (_eval_no_gov) by instantiating
DecisionAgent with governance disabled.  This wrapper was never called by any
evaluation or training entry point and has been removed from the package
__init__.py export surface.

Do NOT import or use this class in new code.

Original docstring preserved below:

B3: AquaAgent without the Governance Agent.

Paper Section 4.4:
  "B3: AquaAgent without governance — the full RL policy is deployed without
   any post-hoc constraint checking, allowing potentially unsafe actions."

This is a thin wrapper that uses DecisionAgent but skips the GovernanceAgent
validation step. It expects a separately trained checkpoint (da_no_gov_best.pt)
trained without governance penalties in the reward function.
"""

import numpy as np
import torch

from src.agents.decision_agent import DecisionAgent, build_obs_vector


class RLNoGovAgent:
    """
    Wraps the DecisionAgent policy without governance constraints.
    All proposed actions are executed directly without validation.
    """

    def __init__(self, obs_dim: int, cfg: dict, device: str = "cpu"):
        # Copy config but remove governance violation penalty from reward
        import copy
        no_gov_cfg = copy.deepcopy(cfg)
        no_gov_cfg.setdefault("mappo", {}).setdefault("reward", {})["gamma_r"] = 0.0
        no_gov_cfg.setdefault("governance", {})["enabled"] = False

        self.da = DecisionAgent(obs_dim=obs_dim, cfg=no_gov_cfg, device=device)
        self.device = torch.device(device)

    def act(self, obs: np.ndarray) -> dict:
        """Return unvalidated action from policy."""
        action, _, _ = self.da.act(obs)
        return action

    def detect(self, obs: np.ndarray) -> np.ndarray:
        """No detection capability — returns zero flags."""
        num_edges = len(obs) // 4  # Approximate from obs vector
        return np.zeros(num_edges, dtype=np.float32)

    def score(self, obs: np.ndarray) -> np.ndarray:
        """No anomaly scoring — returns zero probabilities."""
        return self.detect(obs)

    def load(self, path: str) -> None:
        self.da.load(path)
