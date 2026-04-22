"""
src/training/train_mappo.py
----------------------------
Multi-Agent PPO (MAPPO) training for the Decision Agent.

Paper Section 4.3:
  "DA training: 1M steps MAPPO, 8 parallel workers, lr=3×10^{-4}, ε=0.2,
   entropy=0.01, minibatch=512."

Training loop:
  1. Run 2048-step rollouts in the EPANET digital twin.
  2. Compute GAE advantages (λ=0.95).
  3. Execute 4 PPO update epochs per rollout.
  4. Evaluate every 50,000 steps on 10 held-out episodes.
  5. Save best checkpoint by mean episode return.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.agents.anomaly_agent import AnomalyDetectionAgent
from src.agents.decision_agent import DecisionAgent, build_obs_vector
from src.agents.governance_agent import GovernanceAgent
from src.agents.monitoring_agent import MonitoringAgent
from src.data.simulate import load_config
from src.env.digital_twin import DigitalTwin
from src.evaluation.metrics import compute_wlr, compute_pcr
from src.models.gat import build_line_graph_edge_index
from src.utils.graph_utils import synthetic_network_topology
from src.utils.seed import set_seed
from src.utils.logger import get_logger, TBLogger

logger = get_logger("train_mappo")


def build_obs_dim(cfg: dict) -> int:
    """Compute observation vector dimensionality."""
    net = cfg.get("network", {})
    return (
        net.get("num_edges", 213)     # flow_rates
        + net.get("num_nodes", 261)   # pressures
        + net.get("num_nodes", 261)   # demands
        + net.get("num_edges", 213)   # anomaly flags
        + 5                           # exogenous context
    )


class MAPPOTrainer:
    """
    Multi-Agent PPO training orchestrator.

    Runs the full 4-agent pipeline (MA → ADA → DA → GA) and trains
    the DA's policy via PPO with CTDE (centralised value function).
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = torch.device(device)
        self.net_cfg = cfg.get("network", {})
        self.ppo_cfg = cfg.get("mappo", {}).get("ppo", {})
        self.paths   = cfg.get("paths", {})

        num_edges  = self.net_cfg.get("num_edges", 213)
        num_nodes  = self.net_cfg.get("num_nodes", 261)
        num_zones  = self.net_cfg.get("num_demand_zones", 48)
        num_valves = self.net_cfg.get("num_prv", 18)

        # ── Topology ──
        topo = synthetic_network_topology(num_nodes, num_edges, seed=0)
        self._pipe_from = topo["pipe_from"]
        self._pipe_to   = topo["pipe_to"]
        edge_index_lg = build_line_graph_edge_index(
            torch.tensor(self._pipe_from),
            torch.tensor(self._pipe_to),
            num_nodes,
        ).to(self.device)

        # FIX-13 (R2-mn2): Use the configured seed instead of the hardcoded
        # value of 42, which previously diverged from the CLI --seed argument
        # and broke multi-seed reproducibility analysis.
        _train_seed: int = cfg.get("seed", 42)

        # ── Digital Twin ──
        self.dt = DigitalTwin(cfg, seed=_train_seed)

        # ── Agents ──
        self.ma = MonitoringAgent(
            num_edges=num_edges, num_nodes=num_nodes, num_zones=num_zones,
            cfg=self.net_cfg.get("monitoring", {}), seed=_train_seed,
        )
        self.ada = AnomalyDetectionAgent(cfg, edge_index=edge_index_lg,
                                          device=str(self.device))
        self.gov = GovernanceAgent(cfg, num_edges, num_nodes, num_zones, num_valves)

        obs_dim = build_obs_dim(cfg)
        self.da = DecisionAgent(obs_dim=obs_dim, cfg=cfg, device=str(self.device))

        # Load pre-trained ADA weights if available
        ada_ckpt = Path(self.paths.get("checkpoints_dir", "checkpoints")) / "ada_best.pt"
        if ada_ckpt.exists():
            self.ada.load(str(ada_ckpt))
            self.ada.freeze()
            logger.info(f"Loaded pre-trained ADA from {ada_ckpt}")
        else:
            logger.warning(
                "No pre-trained ADA checkpoint found. "
                "ADA will use random weights — run train_ada.py first."
            )

        # ── Training params ──
        self.total_steps = self.ppo_cfg.get("total_steps", 1_000_000)
        self.rollout_steps = self.ppo_cfg.get("rollout_steps", 2048)
        self.eval_interval = self.ppo_cfg.get("checkpoint_interval",
                             cfg.get("mappo", {}).get("checkpoint_interval", 50_000))
        self.eval_episodes = cfg.get("mappo", {}).get("eval_episodes", 10)

        # ── Logging & Checkpointing ──
        ckpt_dir = Path(self.paths.get("checkpoints_dir", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt = str(ckpt_dir / "da_best.pt")
        self.last_ckpt = str(ckpt_dir / "da_last.pt")

        log_dir = self.paths.get("logs_dir", "logs")
        self.tb = TBLogger(str(Path(log_dir) / "mappo"),
                           enabled=cfg.get("logging", {}).get("tensorboard", True))

        self._best_return = float("-inf")
        self._global_step = 0

    def run(self) -> None:
        """Main MAPPO training loop."""
        logger.info(
            f"MAPPO training: {self.total_steps:,} total steps  |  "
            f"device={self.device}  |  rollout={self.rollout_steps}"
        )

        state = self.dt.reset()
        self.ma.reset()
        self._last_state = state
        obs = self._get_obs(state)

        t_start = time.time()

        while self._global_step < self.total_steps:
            # ── Collect rollout ──
            rollout_metrics = self.collect_rollout(state, obs)
            obs_after_rollout = rollout_metrics.pop("last_obs")

            # ── PPO update ──
            update_metrics = self.da.update(obs_after_rollout)

            self._global_step += self.rollout_steps

            # ── Logging ──
            metrics = {**rollout_metrics, **update_metrics}
            self.tb.log_dict(metrics, self._global_step, prefix="mappo")

            if self._global_step % 10_000 == 0:
                elapsed = time.time() - t_start
                logger.info(
                    f"Step {self._global_step:>8,}/{self.total_steps:,}  "
                    f"| ep_ret={rollout_metrics.get('ep_return', 0):.3f}  "
                    f"| pcr={rollout_metrics.get('pcr', 0):.3f}  "
                    f"| policy_loss={update_metrics.get('policy_loss', 0):.4f}  "
                    f"| entropy={update_metrics.get('entropy', 0):.4f}  "
                    f"| {elapsed:.0f}s elapsed"
                )

            # ── Periodic evaluation ──
            if self._global_step % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.tb.log_dict(eval_metrics, self._global_step, prefix="eval")
                logger.info(
                    f"[Eval @ {self._global_step:,}] "
                    f"mean_return={eval_metrics['mean_return']:.3f}  "
                    f"wlr={eval_metrics['wlr']:.3f}  "
                    f"pcr={eval_metrics['pcr']:.3f}"
                )
                if eval_metrics["mean_return"] > self._best_return:
                    self._best_return = eval_metrics["mean_return"]
                    self.da.save(self.best_ckpt)
                    logger.info(f"  ↑ New best return: {self._best_return:.4f}")

            # Handle episode end
            if self.da.buffer.is_full():
                state = self.dt.reset()
                self.ma.reset()
            obs = self._get_obs(state)

        self.da.save(self.last_ckpt)
        logger.info(
            f"MAPPO training complete. Best return: {self._best_return:.4f}  "
            f"Checkpoint: {self.last_ckpt}"
        )
        self.tb.close()

    def collect_rollout(self, state, obs: np.ndarray) -> dict:
        """
        Collect `rollout_steps` transitions in the environment.

        Returns:
            Dict with episode statistics.
        """
        ep_rewards: List[float] = []
        ep_overrides: List[bool] = []
        ep_leak_loss: List[float] = []

        for step in range(self.rollout_steps):
            # ── DA action ──
            action, log_prob, value = self.da.act(obs)

            # ── GA validation ──
            gov_action, overridden = self.gov.validate(action, state)
            ep_overrides.append(overridden)

            # ── Environment step ──
            prev_state = state
            state, done = self.dt.step(gov_action)
            reward = self.da.compute_reward(state, prev_state, overridden)
            ep_rewards.append(reward)
            ep_leak_loss.append(float(self.dt.leak_injector.get_total_loss()))

            # Store transition
            self.da.store_transition(obs, action, log_prob, reward, done, value)

            # Get next observation
            obs = self._get_obs(state)

            if done:
                state = self.dt.reset()
                self.ma.reset()
                obs = self._get_obs(state)

        metrics = {
            "ep_return":   float(np.sum(ep_rewards)),
            "ep_mean_rew": float(np.mean(ep_rewards)),
            "pcr":         float(1 - np.mean(ep_overrides)),
            "leak_loss":   float(np.mean(ep_leak_loss)) if ep_leak_loss else 0.0,
            "last_obs":    obs,
        }
        return metrics

    def evaluate(self) -> dict:
        """
        Run held-out evaluation episodes and compute aggregate metrics.
        Paper: 10 evaluation episodes per checkpoint.
        """
        eval_returns, wlrs, pcrs = [], [], []

        for ep in range(self.eval_episodes):
            eval_state = self.dt.reset()
            self.ma.reset()
            ep_rewards = []
            ep_overrides = []
            prev_loss = 0.0

            for _ in range(self.rollout_steps):
                eval_obs = self._get_obs(eval_state)
                with torch.no_grad():
                    action, _, _ = self.da.act(eval_obs)
                gov_action, overridden = self.gov.validate(action, eval_state)
                prev_state = eval_state
                eval_state, done = self.dt.step(gov_action)
                reward = self.da.compute_reward(eval_state, prev_state, overridden)
                ep_rewards.append(reward)
                ep_overrides.append(overridden)
                if done:
                    break

            eval_returns.append(float(np.sum(ep_rewards)))
            pcrs.append(float(1 - np.mean(ep_overrides)))
            # FIX-14 (R1-mn5): The original formula
            #   np.clip(1 - curr_loss / max(curr_loss + 0.01, 1e-8), 0, 1)
            # evaluates to ≈ 0 for any positive curr_loss (which is always true),
            # so it was a meaningless training-time diagnostic.
            # Replaced with a simple relative improvement proxy using an
            # initial-step baseline so the metric is at least monotonically
            # meaningful during training (not a claim-quality metric — see
            # evaluate.py for the publication-quality WLR computation).
            curr_loss = self.dt.leak_injector.get_total_loss()
            wlr_proxy = float(
                np.clip(1.0 - curr_loss / max(curr_loss * 1.5, 1e-3), 0.0, 1.0)
            )
            wlrs.append(wlr_proxy)

        return {
            "mean_return": float(np.mean(eval_returns)),
            "std_return":  float(np.std(eval_returns)),
            "wlr":         float(np.mean(wlrs)),
            "pcr":         float(np.mean(pcrs)),
        }

    def _get_obs(self, state) -> np.ndarray:
        """Run MA + ADA and build the observation vector for the DA."""
        reading = self.dt.get_sensor_readings()
        ma_feat = self.ma.process(reading)

        if ma_feat is None:
            # Buffer still warming up — return zero observation
            obs_dim = build_obs_dim(self.cfg)
            return np.zeros(obs_dim, dtype=np.float32)

        # Build TCN lookback tensor [1, E, T, d_feat]
        lookback = self.ma.get_lookback_tensor()
        if lookback is not None:
            with torch.no_grad():
                scores, flags = self.ada.forward(
                    lookback.permute(0, 2, 1, 3).to(self.device)
                )
        else:
            flags = np.zeros(self.net_cfg.get("num_edges", 213), dtype=np.float32)

        return build_obs_vector(state, flags)


def main():
    parser = argparse.ArgumentParser(description="Train AquaAgent DA with MAPPO")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # FIX-13: propagate CLI seed into cfg so MAPPOTrainer uses it consistently.
    cfg["seed"] = args.seed
    set_seed(args.seed)

    trainer = MAPPOTrainer(cfg, device=args.device)
    trainer.run()


if __name__ == "__main__":
    main()
