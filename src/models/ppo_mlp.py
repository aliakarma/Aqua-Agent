"""
src/models/ppo_mlp.py
---------------------
Shared Actor-Critic MLP for the MAPPO-trained Decision Agent.

Paper Section 3.8:
  "The policy network is a two-layer MLP (512, 256 units, ReLU activations)
   shared across agents; the value network mirrors this architecture."

Paper Equation (12) — PPO objective:
  L_PPO(φ) = E_t[ min(r_t(φ)·Â_t, clip(r_t(φ), 1−ε, 1+ε)·Â_t) ]
  where r_t(φ) = π_φ(a_t|s_t) / π_φ_old(a_t|s_t)

The action space (paper Equation 4) includes:
  - isolate(e):            Binary per-edge isolation
  - adjust_valve(v, δ):    Continuous δ ∈ [−1,1] per PRV
  - reroute(e1 → e2):     Pair selection (approximated as discrete)
  - no_op:                 Do nothing

We implement a hybrid discrete/continuous action head:
  - Discrete head (Categorical): action type ∈ {isolate, adjust, reroute, no_op}
  - Continuous head (Gaussian):  δ for valve adjustment ∈ [−1, 1]
  - Discrete head (Categorical): target edge/valve index
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, List, Optional, Tuple


class MLP(nn.Module):
    """Configurable multi-layer perceptron with ReLU activations."""

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int, activation: str = "relu",
                 output_activation: bool = False):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), act_fn()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation:
            layers.append(act_fn())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Shared actor-critic network for MAPPO.

    Architecture (paper Section 3.8):
      Shared backbone: MLP(input_dim → 512 → 256, ReLU)
      Actor head:      Linear(256 → num_actions)    [Categorical]
      Valve δ head:    Linear(256 → num_valves×2)   [Gaussian: mean + log_std]
      Critic head:     Linear(256 → 1)              [V(s) scalar]

    The critic uses the full centralised state (CTDE — Centralised Training,
    Decentralised Execution), consistent with MAPPO (Assumption A9).
    """

    def __init__(self,
                 obs_dim: int,
                 num_action_types: int = 4,    # {isolate, adjust, reroute, no_op}
                 num_edges: int = 213,
                 num_valves: int = 18,
                 hidden_dims: List[int] = None,
                 central_critic_dim: Optional[int] = None):
        """
        Args:
            obs_dim:            Dimension of local observation vector.
            num_action_types:   Number of discrete action type categories.
            num_edges:          Number of pipe edges (for isolate/reroute targets).
            num_valves:         Number of PRVs (for adjust_valve targets).
            hidden_dims:        MLP hidden layer sizes.
            central_critic_dim: If given, critic uses a larger global state input.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]   # Paper architecture

        # Shared feature backbone (actor)
        self.actor_backbone = MLP(obs_dim, hidden_dims[:-1], hidden_dims[-1])

        # Centralised critic backbone
        critic_in = central_critic_dim if central_critic_dim else obs_dim
        self.critic_backbone = MLP(critic_in, hidden_dims[:-1], hidden_dims[-1])

        backbone_out = hidden_dims[-1]

        # --- Action heads ---
        # 1. Action type (discrete): which action to take
        self.action_type_head = nn.Linear(backbone_out, num_action_types)

        # 2. Edge selection (discrete): which edge to isolate/reroute
        self.edge_select_head = nn.Linear(backbone_out, num_edges)

        # 3. Valve δ (continuous): normalised PRV setpoint adjustment
        self.valve_select_head = nn.Linear(backbone_out, num_valves)
        self.valve_delta_mean = nn.Linear(backbone_out, 1)
        self.valve_delta_log_std = nn.Parameter(torch.zeros(1))

        # Critic
        self.critic_head = nn.Linear(backbone_out, 1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        central_obs: Optional[torch.Tensor] = None,
        action: Optional[Dict] = None,
    ) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action and compute log-prob, entropy, and value estimate.

        Args:
            obs:         Local observation [batch, obs_dim]
            central_obs: Global state for centralised critic [batch, central_dim]
            action:      If provided, evaluate log-prob of this action (for PPO update).

        Returns:
            action_dict: {'type': int, 'edge': int, 'valve': int, 'delta': float}
            log_prob:    Log-probability of the sampled/given action
            entropy:     Policy entropy (for entropy bonus)
            value:       Critic estimate V(s)
        """
        feat = self.actor_backbone(obs)
        critic_input = central_obs if central_obs is not None else obs
        value = self.critic_head(self.critic_backbone(critic_input))

        # --- Action type distribution ---
        type_logits = self.action_type_head(feat)
        type_dist = Categorical(logits=type_logits)

        if action is None:
            action_type = type_dist.sample()
        else:
            action_type = action["type"]

        # --- Edge selection distribution ---
        edge_logits = self.edge_select_head(feat)
        edge_dist = Categorical(logits=edge_logits)
        if action is None:
            edge_idx = edge_dist.sample()
        else:
            edge_idx = action["edge"]

        # --- Valve selection and δ distribution ---
        valve_logits = self.valve_select_head(feat)
        valve_dist = Categorical(logits=valve_logits)
        if action is None:
            valve_idx = valve_dist.sample()
        else:
            valve_idx = action["valve"]

        delta_mean = torch.tanh(self.valve_delta_mean(feat))  # ∈ (−1, 1)
        delta_std = torch.exp(self.valve_delta_log_std).clamp(0.01, 1.0)
        delta_dist = Normal(delta_mean, delta_std)
        if action is None:
            delta = delta_dist.sample().clamp(-1.0, 1.0)
        else:
            delta = action["delta"]

        # --- Aggregate log-probs ---
        log_prob = (
            type_dist.log_prob(action_type)
            + edge_dist.log_prob(edge_idx)
            + valve_dist.log_prob(valve_idx)
            + delta_dist.log_prob(delta).sum(-1)
        )

        entropy = (
            type_dist.entropy()
            + edge_dist.entropy()
            + valve_dist.entropy()
            + delta_dist.entropy().sum(-1)
        )

        action_dict = {
            "type": action_type,
            "edge": edge_idx,
            "valve": valve_idx,
            "delta": delta,
        }

        return action_dict, log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor,
                  central_obs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Critic-only forward pass."""
        critic_input = central_obs if central_obs is not None else obs
        feat = self.critic_backbone(critic_input)
        return self.critic_head(feat).squeeze(-1)


def action_dict_to_str(action: dict) -> str:
    """Convert an action dict to a human-readable string for audit logging."""
    a_type = int(action["type"]) if torch.is_tensor(action["type"]) else action["type"]
    names = {0: "isolate", 1: "adjust_valve", 2: "reroute", 3: "no_op"}
    name = names.get(a_type, "unknown")
    if name == "adjust_valve":
        valve = int(action.get("valve", -1))
        delta = float(action.get("delta", 0.0))
        return f"adjust_valve(v={valve}, δ={delta:.3f})"
    elif name == "isolate":
        edge = int(action.get("edge", -1))
        return f"isolate(e={edge})"
    elif name == "reroute":
        e1 = int(action.get("edge", -1))
        return f"reroute(e={e1})"
    return "no_op"
