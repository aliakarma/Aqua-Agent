"""
src/agents/decision_agent.py
-----------------------------
Decision Agent (DA) — MAPPO-trained PPO policy.

Paper Section 3.8:
  "The Decision Agent is trained with Multi-Agent PPO (MAPPO). Its policy
   π_φ: S × L̂ → Δ(A_dec) takes the full system state and anomaly flags as
   input and outputs a candidate action."

Paper Equation (12) — PPO clipped objective:
  L_PPO(φ) = E_t[ min(r_t(φ)·Â_t, clip(r_t(φ), 1−ε, 1+ε)·Â_t) ]
  where r_t = π_φ(a_t|o_t) / π_φ_old(a_t|o_t)

Paper: GAE with λ = 0.95, γ = 0.99, entropy coefficient = 0.01
Paper Reward (Eq. 3): R = α·r_eff − β·r_leak − γ_r·r_viol
                         α=0.55, β=0.45, γ_r=2.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.ppo_mlp import ActorCritic, action_dict_to_str
from src.utils.logger import get_logger

logger = get_logger("decision_agent")


class RolloutBuffer:
    """
    Fixed-size rollout buffer for PPO experience collection.
    Stores (obs, action, log_prob, reward, done, value) tuples.
    """

    def __init__(self, rollout_steps: int, obs_dim: int,
                 num_agents: int = 1, device: str = "cpu"):
        self.rollout_steps = rollout_steps
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        n = self.rollout_steps
        self.obs       = torch.zeros(n, self.obs_dim).to(self.device)
        self.actions   = {}          # Dict of action component tensors
        self.log_probs = torch.zeros(n).to(self.device)
        self.rewards   = torch.zeros(n).to(self.device)
        self.dones     = torch.zeros(n).to(self.device)
        self.values    = torch.zeros(n).to(self.device)
        self.ptr       = 0
        self._action_types  = torch.zeros(n, dtype=torch.long).to(self.device)
        self._action_edges  = torch.zeros(n, dtype=torch.long).to(self.device)
        self._action_valves = torch.zeros(n, dtype=torch.long).to(self.device)
        self._action_deltas = torch.zeros(n, 1).to(self.device)

    def add(self, obs: torch.Tensor, action: dict,
            log_prob: torch.Tensor, reward: float,
            done: bool, value: torch.Tensor) -> None:
        i = self.ptr
        self.obs[i]       = obs.squeeze(0)
        self.log_probs[i] = log_prob.squeeze()
        self.rewards[i]   = float(reward)
        self.dones[i]     = float(done)
        self.values[i]    = value.squeeze()
        self._action_types[i]  = action["type"].squeeze().long()
        self._action_edges[i]  = action["edge"].squeeze().long()
        self._action_valves[i] = action["valve"].squeeze().long()
        self._action_deltas[i] = action["delta"].squeeze().unsqueeze(-1)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.rollout_steps

    def get_action_dict(self) -> dict:
        return {
            "type":  self._action_types,
            "edge":  self._action_edges,
            "valve": self._action_valves,
            "delta": self._action_deltas,
        }


class DecisionAgent:
    """
    MAPPO Decision Agent.

    Wraps ActorCritic with:
      - GAE advantage estimation (λ=0.95)
      - PPO clipped update (ε=0.2)
      - Entropy bonus (coeff=0.01)
      - Gradient clipping
      - Checkpoint save/load
    """

    def __init__(self, obs_dim: int, cfg: dict, device: str = "cpu"):
        """
        Args:
            obs_dim: Dimension of the observation vector fed to the policy.
            cfg:     Full merged config.
            device:  Torch device string.
        """
        self.cfg = cfg
        self.device = torch.device(device)
        mappo_cfg = cfg.get("mappo", {})
        ppo_cfg = mappo_cfg.get("ppo", {})
        reward_cfg = mappo_cfg.get("reward", {})
        net_cfg = cfg.get("network", {})

        # PPO hyperparameters (paper Section 4.3)
        self.gamma: float        = ppo_cfg.get("gamma", 0.99)
        self.gae_lambda: float   = ppo_cfg.get("gae_lambda", 0.95)
        self.eps_clip: float     = ppo_cfg.get("eps_clip", 0.2)
        self.entropy_coeff: float = ppo_cfg.get("entropy_coeff", 0.01)
        self.value_coeff: float  = ppo_cfg.get("value_loss_coeff", 0.5)
        self.grad_clip: float    = ppo_cfg.get("gradient_clip_norm", 0.5)
        self.minibatch: int      = ppo_cfg.get("minibatch_size", 512)
        self.n_epochs: int       = ppo_cfg.get("num_epochs_per_update", 4)
        self.rollout_steps: int  = ppo_cfg.get("rollout_steps", 2048)
        self.norm_adv: bool      = ppo_cfg.get("normalize_advantages", True)

        # Reward coefficients (paper Equation 3)
        self.alpha:   float = reward_cfg.get("alpha", 0.55)
        self.beta:    float = reward_cfg.get("beta", 0.45)
        self.gamma_r: float = reward_cfg.get("gamma_r", 2.0)

        num_edges  = net_cfg.get("num_edges", 213)
        num_valves = net_cfg.get("num_prv", 18)

        # Actor-Critic model
        self.model = ActorCritic(
            obs_dim=obs_dim,
            num_action_types=4,
            num_edges=num_edges,
            num_valves=num_valves,
            hidden_dims=mappo_cfg.get("actor", {}).get("hidden_dims", [512, 256]),
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=ppo_cfg.get("learning_rate", 3e-4),
        )

        self.buffer = RolloutBuffer(
            rollout_steps=self.rollout_steps,
            obs_dim=obs_dim,
            device=str(self.device),
        )

        self._total_steps = 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the current policy.

        Args:
            obs: Observation vector [obs_dim]

        Returns:
            action_dict, log_prob, value
        """
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action, log_prob, _, value = self.model.get_action_and_value(obs_t)
        return action, log_prob, value

    def compute_reward(self, state, prev_state, overridden: bool) -> float:
        """
        Compute scalar reward from hydraulic state.

        Paper Equation (3):
          R = α·r_eff − β·r_leak − γ_r·r_viol

        Args:
            state:      Current HydraulicState after action.
            prev_state: Previous HydraulicState.
            overridden: Whether the governance agent overrode the action.

        Returns:
            Scalar reward value.
        """
        # r_eff: delivery efficiency — fraction of demand met
        total_demand = np.sum(state.demands) + 1e-8
        # Approximate delivered water as proportional to flow rates
        total_flow = np.sum(state.flow_rates)
        r_eff = float(np.clip(total_flow / total_demand, 0, 1))

        # r_leak: normalised total active leak rate
        prev_flow = np.sum(prev_state.flow_rates) + 1e-8
        curr_flow = np.sum(state.flow_rates)
        r_leak = float(np.clip((prev_flow - curr_flow) / prev_flow, 0, 1))

        # r_viol: governance penalty (binary)
        r_viol = 1.0 if overridden else 0.0

        return self.alpha * r_eff - self.beta * r_leak - self.gamma_r * r_viol

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def store_transition(self, obs: np.ndarray, action: dict,
                         log_prob: torch.Tensor, reward: float,
                         done: bool, value: torch.Tensor) -> None:
        """Store one transition in the rollout buffer."""
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        self.buffer.add(obs_t, action, log_prob, reward, done, value)
        self._total_steps += 1

    def update(self, last_obs: np.ndarray) -> Dict[str, float]:
        """
        Run PPO update on the collected rollout buffer.

        Args:
            last_obs: Final observation for bootstrapping returns.

        Returns:
            Dict of training metrics (losses, clip fraction, etc.)
        """
        # ── Compute GAE advantages ──
        advantages, returns = self._compute_gae(last_obs)

        # ── Flatten buffer ──
        obs_b      = self.buffer.obs
        act_b      = self.buffer.get_action_dict()
        logp_old_b = self.buffer.log_probs
        ret_b      = returns
        adv_b      = advantages

        # Normalise advantages
        if self.norm_adv:
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # ── PPO mini-batch updates ──
        metrics = {"policy_loss": 0., "value_loss": 0.,
                   "entropy": 0., "clip_frac": 0.}
        n = obs_b.shape[0]
        n_updates = 0

        for _ in range(self.n_epochs):
            # Shuffle indices for mini-batches
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch):
                mb_idx = idx[start: start + self.minibatch]
                if len(mb_idx) < 4:
                    continue

                mb_obs  = obs_b[mb_idx]
                mb_adv  = adv_b[mb_idx]
                mb_ret  = ret_b[mb_idx]
                mb_logp_old = logp_old_b[mb_idx]
                mb_act  = {k: v[mb_idx] for k, v in act_b.items()}

                # Re-evaluate actions under current policy
                _, new_logp, entropy, new_val = self.model.get_action_and_value(
                    mb_obs, action=mb_act
                )

                # Probability ratio r_t(φ) = π_φ(a|s) / π_φ_old(a|s)
                log_ratio = new_logp - mb_logp_old
                ratio = torch.exp(log_ratio.clamp(-10, 10))

                # PPO clipped surrogate loss (paper Equation 12)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (Huber loss for robustness)
                value_loss = nn.functional.huber_loss(new_val, mb_ret)

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + self.value_coeff * value_loss
                        + self.entropy_coeff * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                # Clip fraction tracking
                with torch.no_grad():
                    clip_frac = ((ratio - 1).abs() > self.eps_clip).float().mean()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"]  += value_loss.item()
                metrics["entropy"]     += (-entropy_loss).item()
                metrics["clip_frac"]   += clip_frac.item()
                n_updates += 1

        if n_updates > 0:
            for k in metrics:
                metrics[k] /= n_updates

        self.buffer.reset()
        return metrics

    def _compute_gae(self, last_obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalised Advantage Estimation (Schulman et al., 2016).

        Paper: GAE λ = 0.95, γ = 0.99

        Returns:
            advantages: [rollout_steps]
            returns:    [rollout_steps]
        """
        with torch.no_grad():
            last_obs_t = torch.from_numpy(last_obs.astype(np.float32)).unsqueeze(0).to(self.device)
            last_value = self.model.get_value(last_obs_t).squeeze()

        rewards = self.buffer.rewards
        dones   = self.buffer.dones
        values  = self.buffer.values

        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_val = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_steps": self._total_steps,
        }, path)
        logger.info(f"DA checkpoint saved: {path}  (steps={self._total_steps})")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._total_steps = ckpt.get("total_steps", 0)
        logger.info(f"DA checkpoint loaded: {path}  (steps={self._total_steps})")


def build_obs_vector(state, anomaly_flags: np.ndarray) -> np.ndarray:
    """
    Construct the flat observation vector for the policy.

    Concatenates: [flow_rates | pressures | demands | anomaly_flags | exogenous]

    Paper: "π_φ: S × L̂ → Δ(A_dec) takes the full system state and anomaly
            flags as input."

    Returns:
        obs: np.ndarray [obs_dim]
    """
    return np.concatenate([
        state.flow_rates,
        state.pressures,
        state.demands,
        anomaly_flags,
        state.exogenous,
    ]).astype(np.float32)
