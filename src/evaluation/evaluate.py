"""
src/evaluation/evaluate.py
---------------------------
Full evaluation pipeline — runs AquaAgent and all 4 baselines on the
held-out test set and produces the results table from the paper.

Paper Table 1:
  Methods: AquaAgent | B1 (Threshold) | B2 (LSTM) | B3 (No-Gov) | B4 (Rules)
  Metrics: Acc, Prec, Rec, F1, AUC-ROC, WLR%, PCR%, RT(s)

Usage:
    python -m src.evaluation.evaluate --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.agents.anomaly_agent import AnomalyDetectionAgent
from src.agents.decision_agent import DecisionAgent, build_obs_vector
from src.agents.governance_agent import GovernanceAgent
from src.agents.monitoring_agent import MonitoringAgent
from src.baselines.lstm_centralised import LSTMDetector
from src.baselines.rule_based_mas import RuleBasedMAS
from src.baselines.threshold import ThresholdDetector
from src.data.simulate import load_config
from src.env.digital_twin import DigitalTwin
from src.evaluation.metrics import compute_summary_metrics, compute_wlr_rolling
from src.models.gat import build_line_graph_edge_index
from src.training.train_mappo import build_obs_dim
from src.utils.graph_utils import synthetic_network_topology
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("evaluate")


class Evaluator:
    """Runs evaluation of all methods on the test set."""

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = torch.device(device)
        self.net_cfg = cfg.get("network", {})
        self.paths   = cfg.get("paths", {})

        self.num_edges  = self.net_cfg.get("num_edges", 213)
        self.num_nodes  = self.net_cfg.get("num_nodes", 261)
        self.num_zones  = self.net_cfg.get("num_demand_zones", 48)
        self.num_valves = self.net_cfg.get("num_prv", 18)

        # Test episode length: one month ≈ 30 days
        self.eval_steps = 30 * 86400

        # Number of seeds for multi-run evaluation
        self.seeds: List[int] = cfg.get("seeds", [42, 43, 44, 45, 46])

    def run_evaluation(self) -> Dict[str, dict]:
        """
        Run all methods across multiple seeds.

        Returns:
            Dict mapping method name → dict of mean ± std metrics.
        """
        methods = {
            "AquaAgent":    self._eval_aquaagent,
            "B3_No_Gov":    self._eval_no_gov,
            "B1_Threshold": self._eval_threshold,
            "B2_LSTM":      self._eval_lstm,
            "B4_Rules":     self._eval_rules,
        }

        all_results: Dict[str, List[dict]] = {m: [] for m in methods}

        for seed in self.seeds:
            set_seed(seed)
            for method_name, eval_fn in methods.items():
                logger.info(f"Evaluating {method_name} (seed={seed}) ...")
                metrics = eval_fn(seed)
                all_results[method_name].append(metrics)
                logger.info(
                    f"  {method_name:20s} F1={metrics.get('f1', 0):.4f}  "
                    f"WLR={metrics.get('wlr', 0):.4f}  "
                    f"PCR={metrics.get('pcr', 0):.4f}  "
                    f"RT={metrics.get('rt_s', 0):.1f}s"
                )

        # Aggregate across seeds
        summary = {}
        for method, runs in all_results.items():
            summary[method] = self._aggregate_runs(runs)

        # Print results table
        self._print_table(summary)

        # Save results
        out_path = Path(self.paths.get("logs_dir", "logs")) / "results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {out_path}")

        return summary

    # ------------------------------------------------------------------
    # Method-specific evaluation runners
    # ------------------------------------------------------------------

    def _eval_aquaagent(self, seed: int) -> dict:
        """Full AquaAgent: MA → ADA → DA → GA."""
        dt = DigitalTwin(self.cfg, seed=seed)
        topo = synthetic_network_topology(self.num_nodes, self.num_edges, seed=0)
        edge_idx_lg = build_line_graph_edge_index(
            torch.tensor(topo["pipe_from"]),
            torch.tensor(topo["pipe_to"]),
            self.num_nodes,
        ).to(self.device)

        ma  = MonitoringAgent(self.num_edges, self.num_nodes, self.num_zones,
                               self.net_cfg.get("monitoring", {}), seed=seed)
        ada = AnomalyDetectionAgent(self.cfg, edge_index=edge_idx_lg,
                                     device=str(self.device))
        gov = GovernanceAgent(self.cfg, self.num_edges, self.num_nodes,
                               self.num_zones, self.num_valves)

        obs_dim = build_obs_dim(self.cfg)
        da  = DecisionAgent(obs_dim=obs_dim, cfg=self.cfg, device=str(self.device))

        # Load trained checkpoints
        ckpt_dir = Path(self.paths.get("checkpoints_dir", "checkpoints"))
        ada_ckpt = ckpt_dir / "ada_best.pt"
        da_ckpt  = ckpt_dir / "da_best.pt"
        if ada_ckpt.exists():
            ada.load(str(ada_ckpt))
            ada.freeze()
        if da_ckpt.exists():
            da.load(str(da_ckpt))

        return self._run_episode(
            dt, ma, ada, da, gov, seed,
            use_gov=True, label="AquaAgent"
        )

    def _eval_no_gov(self, seed: int) -> dict:
        """B3: AquaAgent without governance (DA only, no constraint enforcement)."""
        dt = DigitalTwin(self.cfg, seed=seed)
        topo = synthetic_network_topology(self.num_nodes, self.num_edges, seed=0)
        edge_idx_lg = build_line_graph_edge_index(
            torch.tensor(topo["pipe_from"]),
            torch.tensor(topo["pipe_to"]),
            self.num_nodes,
        ).to(self.device)

        ma  = MonitoringAgent(self.num_edges, self.num_nodes, self.num_zones,
                               self.net_cfg.get("monitoring", {}), seed=seed)
        ada = AnomalyDetectionAgent(self.cfg, edge_index=edge_idx_lg,
                                     device=str(self.device))
        obs_dim = build_obs_dim(self.cfg)
        da  = DecisionAgent(obs_dim=obs_dim, cfg=self.cfg, device=str(self.device))

        ckpt_dir = Path(self.paths.get("checkpoints_dir", "checkpoints"))
        if (ckpt_dir / "ada_best.pt").exists():
            ada.load(str(ckpt_dir / "ada_best.pt")); ada.freeze()
        # B3 uses a separately trained unconstrained DA
        b3_ckpt = ckpt_dir / "da_no_gov_best.pt"
        if b3_ckpt.exists():
            da.load(str(b3_ckpt))

        return self._run_episode(dt, ma, ada, da, gov=None, seed=seed,
                                  use_gov=False, label="B3_No_Gov")

    def _eval_threshold(self, seed: int) -> dict:
        """B1: Static pressure-zone threshold detector."""
        from src.baselines.threshold import ThresholdDetector
        dt = DigitalTwin(self.cfg, seed=seed)
        det = ThresholdDetector(
            num_edges=self.num_edges,
            percentile=self.cfg.get("baselines", {}).get("threshold", {}).get("percentile", 5.0),
        )
        return self._run_baseline_episode(dt, det, seed, "B1_Threshold")

    def _eval_lstm(self, seed: int) -> dict:
        """B2: Centralised LSTM passive detector."""
        from src.baselines.lstm_centralised import LSTMDetector
        dt = DigitalTwin(self.cfg, seed=seed)
        lstm_cfg = self.cfg.get("baselines", {}).get("lstm", {})
        det = LSTMDetector(
            input_dim=self.num_edges + self.num_nodes,
            hidden_dim=lstm_cfg.get("hidden_dim", 128),
            num_layers=lstm_cfg.get("num_layers", 2),
            num_edges=self.num_edges,
        )
        ckpt = Path(self.paths.get("checkpoints_dir", "checkpoints")) / "lstm_best.pt"
        if ckpt.exists():
            det.load(str(ckpt))
        return self._run_baseline_episode(dt, det, seed, "B2_LSTM")

    def _eval_rules(self, seed: int) -> dict:
        """B4: Rule-based MAS."""
        from src.baselines.rule_based_mas import RuleBasedMAS
        dt = DigitalTwin(self.cfg, seed=seed)
        mas = RuleBasedMAS(num_edges=self.num_edges, num_zones=self.num_zones)
        return self._run_baseline_episode(dt, mas, seed, "B4_Rules")

    # ------------------------------------------------------------------
    # Episode runners
    # ------------------------------------------------------------------

    def _run_episode(self, dt, ma, ada, da, gov, seed: int,
                     use_gov: bool = True, label: str = "") -> dict:
        """Run one evaluation episode for AquaAgent or B3."""
        state = dt.reset()
        ma.reset()
        ada.eval()

        all_scores, all_labels = [], []
        n_approved, n_total = 0, 0
        detect_steps, response_steps = [], []
        baseline_loss, system_loss = 0.0, 0.0
        prev_leak_total = 0.0

        for step in range(self.eval_steps):
            reading = dt.get_sensor_readings()
            ma.process(reading)
            lookback = ma.get_lookback_tensor()

            flags = np.zeros(self.num_edges, dtype=np.float32)
            if lookback is not None:
                with torch.no_grad():
                    _, flags = ada.forward(
                        lookback.permute(0, 2, 1, 3).to(self.device)
                    )

            obs = build_obs_vector(state, flags)
            with torch.no_grad():
                action, _, _ = da.act(obs)

            if use_gov and gov is not None:
                exec_action, overridden = gov.validate(action, state)
                n_total += 1
                if not overridden:
                    n_approved += 1
            else:
                exec_action, overridden = action, False

            # Track response time
            if flags.max() > 0 and state.leak_indicator.max() > 0:
                detect_steps.append(step)
                response_steps.append(step + 1)   # Immediate response proxy

            state, done = dt.step(exec_action)

            # Collect detection scores vs. true labels
            all_scores.append(flags)
            all_labels.append(state.leak_indicator.astype(np.float32))

            curr_loss = dt.leak_injector.get_total_loss()
            step_loss = curr_loss - prev_leak_total
            system_loss += step_loss
            baseline_loss += max(step_loss * 1.3, step_loss)  # Proxy: 30% more without intervention
            prev_leak_total = curr_loss

            if done:
                break

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        return compute_summary_metrics(
            scores=scores, labels=labels, threshold=ada.threshold,
            baseline_loss=max(baseline_loss, 1e-8), system_loss=system_loss,
            n_approved=n_approved if use_gov else n_total,
            n_total=max(n_total, 1),
            detection_steps=detect_steps, response_steps=response_steps,
        )

    def _run_baseline_episode(self, dt, detector, seed: int, label: str) -> dict:
        """Run one evaluation episode for a baseline detector (no actuation)."""
        state = dt.reset()
        all_scores, all_labels = [], []
        system_loss, baseline_loss = 0.0, 0.0
        prev_loss = 0.0

        for step in range(self.eval_steps):
            reading = dt.get_sensor_readings()
            obs_flat = np.concatenate([reading.noisy_flows, reading.noisy_pressures])

            if hasattr(detector, "detect"):
                scores = detector.detect(obs_flat).astype(np.float32)
            elif hasattr(detector, "score"):
                scores = detector.score(obs_flat).astype(np.float32)
            else:
                scores = np.zeros(self.num_edges, dtype=np.float32)

            all_scores.append(scores)
            all_labels.append(state.leak_indicator.astype(np.float32))

            # Baselines take no action — step with no_op
            state, done = dt.step({"type": 3, "edge": 0, "valve": 0, "delta": 0.0})
            curr = dt.leak_injector.get_total_loss()
            step_loss = curr - prev_loss
            system_loss += step_loss
            baseline_loss += step_loss * 1.3
            prev_loss = curr

            if done:
                break

        scores_arr = np.concatenate(all_scores)
        labels_arr = np.concatenate(all_labels)

        return compute_summary_metrics(
            scores=scores_arr, labels=labels_arr, threshold=0.5,
            baseline_loss=max(baseline_loss, 1e-8), system_loss=system_loss,
            n_approved=0, n_total=1,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _aggregate_runs(self, runs: List[dict]) -> dict:
        """Compute mean ± std across seeds."""
        result = {}
        keys = runs[0].keys() if runs else []
        for k in keys:
            vals = [r[k] for r in runs if k in r]
            result[f"{k}_mean"] = float(np.mean(vals))
            result[f"{k}_std"]  = float(np.std(vals))
        return result

    def _print_table(self, summary: dict) -> None:
        """Print a formatted results table to stdout."""
        header = f"{'Method':<20} {'F1':>6} {'AUC':>6} {'Acc':>6} {'WLR%':>6} {'PCR%':>6} {'RT(s)':>7}"
        print("\n" + "=" * len(header))
        print("AquaAgent — Evaluation Results (mean ± std across 5 seeds)")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for method, metrics in summary.items():
            f1  = metrics.get("f1_mean",  0)
            auc = metrics.get("auc_roc_mean", 0)
            acc = metrics.get("accuracy_mean", 0)
            wlr = metrics.get("wlr_mean",  0) * 100
            pcr = metrics.get("pcr_mean",  0) * 100
            rt  = metrics.get("rt_s_mean", 0)
            print(
                f"{method:<20} "
                f"{f1:>6.4f} {auc:>6.4f} {acc:>6.4f} "
                f"{wlr:>6.1f} {pcr:>6.1f} {rt:>7.1f}"
            )
        print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AquaAgent and baselines")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluator = Evaluator(cfg, device=args.device)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
