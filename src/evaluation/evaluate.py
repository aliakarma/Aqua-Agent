"""
src/evaluation/evaluate.py
---------------------------
Full evaluation pipeline — runs AquaAgent and all 4 baselines on the
held-out test set and produces the results table from the paper.

Paper Table 1:
  Methods: AquaAgent | B1 (Threshold) | B2 (LSTM) | B3 (No-Gov) | B4 (Rules)
  Metrics: Acc, Prec, Rec, F1, AUC-ROC, WLR%, PCR%, RT(s)

Fixes applied
-------------
FIX-01 (R1-C1 / R2-C3): RT is now computed from genuine per-leak
  first-detection and first-corrective-action steps instead of the
  hardcoded ``step + 1`` proxy.  Score/label alignment corrected:
  appended *before* ``dt.step()`` so both refer to the same timestep.

FIX-02 (R1-C2 / R2-C3): WLR baseline is a true no-action counterfactual
  episode, replacing the circular 1.3x proxy.

FIX-12 (R1-M5 / R2-M5): Wilcoxon signed-rank p-values and Cohen's d
  added to _aggregate_runs_all.

FIX-22 (R2-mn3): --strict CLI flag raises FileNotFoundError when
  required checkpoints are absent.

Usage:
    python -m src.evaluation.evaluate --config configs/default.yaml
    python -m src.evaluation.evaluate --config configs/default.yaml --strict
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

    def __init__(self, cfg: dict, device: str = "cpu", strict: bool = False):
        self.cfg    = cfg
        self.device = torch.device(device)
        self.strict = strict   # FIX-22
        self.net_cfg = cfg.get("network", {})
        self.paths   = cfg.get("paths", {})

        self.num_edges  = self.net_cfg.get("num_edges", 213)
        self.num_nodes  = self.net_cfg.get("num_nodes", 261)
        self.num_zones  = self.net_cfg.get("num_demand_zones", 48)
        self.num_valves = self.net_cfg.get("num_prv", 18)

        self.eval_steps = 30 * 86400
        self.seeds: List[int] = cfg.get("seeds", [42, 43, 44, 45, 46])

        # FIX-02: cache no-action loss per seed.
        self._no_action_loss_cache: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # FIX-02: No-action counterfactual reference runner
    # ------------------------------------------------------------------

    def _get_no_action_loss(self, seed: int) -> float:
        """
        Run a no-op episode under the same seed and return total water loss.

        Replaces the circular baseline_loss += step_loss * 1.3 proxy.
        WLR = 1 - system_loss / no_action_loss.
        """
        if seed in self._no_action_loss_cache:
            return self._no_action_loss_cache[seed]

        logger.info(f"  Computing no-action reference loss for seed={seed} ...")
        dt_ref = DigitalTwin(self.cfg, seed=seed)
        dt_ref.reset()
        noop = {"type": 3, "edge": 0, "valve": 0, "delta": 0.0}
        for _ in range(self.eval_steps):
            _, done = dt_ref.step(noop)
            if done:
                break
        loss = dt_ref.leak_injector.get_total_loss()
        self._no_action_loss_cache[seed] = loss
        logger.info(f"  No-action loss (seed={seed}): {loss:.2f} L")
        return loss

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_evaluation(self) -> Dict[str, dict]:
        """Run all methods across multiple seeds."""
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
            self._get_no_action_loss(seed)  # pre-compute once per seed
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

        summary = self._aggregate_runs_all(all_results)
        self._print_table(summary)

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
        """Full AquaAgent: MA -> ADA -> DA -> GA."""
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

        ckpt_dir = Path(self.paths.get("checkpoints_dir", "checkpoints"))
        ada_ckpt = ckpt_dir / "ada_best.pt"
        da_ckpt  = ckpt_dir / "da_best.pt"

        if self.strict:
            if not ada_ckpt.exists():
                raise FileNotFoundError(
                    f"[strict] ADA checkpoint missing: {ada_ckpt}\n"
                    "Run: bash scripts/run_train_ada.sh"
                )
            if not da_ckpt.exists():
                raise FileNotFoundError(
                    f"[strict] DA checkpoint missing: {da_ckpt}\n"
                    "Run: bash scripts/run_train_mappo.sh"
                )

        if ada_ckpt.exists():
            ada.load(str(ada_ckpt)); ada.freeze()
        else:
            logger.warning(f"ADA checkpoint not found at {ada_ckpt}; using random weights.")
        if da_ckpt.exists():
            da.load(str(da_ckpt))
        else:
            logger.warning(f"DA checkpoint not found at {da_ckpt}; using random weights.")

        return self._run_episode(dt, ma, ada, da, gov, seed, use_gov=True, label="AquaAgent")

    def _eval_no_gov(self, seed: int) -> dict:
        """B3: AquaAgent without governance."""
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

        b3_ckpt = ckpt_dir / "da_no_gov_best.pt"
        if self.strict and not b3_ckpt.exists():
            raise FileNotFoundError(
                f"[strict] B3 checkpoint missing: {b3_ckpt}\n"
                "Run: bash scripts/run_train_no_gov.sh"
            )
        if b3_ckpt.exists():
            da.load(str(b3_ckpt))
        else:
            logger.warning(f"B3 checkpoint not found at {b3_ckpt}; using random weights.")

        return self._run_episode(dt, ma, ada, da, gov=None, seed=seed,
                                  use_gov=False, label="B3_No_Gov")

    def _eval_threshold(self, seed: int) -> dict:
        """B1: Static percentile threshold detector."""
        dt  = DigitalTwin(self.cfg, seed=seed)
        det = ThresholdDetector(
            num_edges=self.num_edges,
            percentile=self.cfg.get("baselines", {}).get("threshold", {}).get("percentile", 5.0),
        )
        return self._run_baseline_episode(dt, det, seed, "B1_Threshold")

    def _eval_lstm(self, seed: int) -> dict:
        """B2: Centralised LSTM passive detector."""
        dt = DigitalTwin(self.cfg, seed=seed)
        lstm_cfg = self.cfg.get("baselines", {}).get("lstm", {})
        det = LSTMDetector(
            input_dim=self.num_edges + self.num_nodes,
            hidden_dim=lstm_cfg.get("hidden_dim", 128),
            num_layers=lstm_cfg.get("num_layers", 2),
            num_edges=self.num_edges,
        )
        ckpt = Path(self.paths.get("checkpoints_dir", "checkpoints")) / "lstm_best.pt"
        if self.strict and not ckpt.exists():
            raise FileNotFoundError(
                f"[strict] B2 checkpoint missing: {ckpt}\n"
                "Run: bash scripts/run_train_lstm.sh"
            )
        if ckpt.exists():
            det.load(str(ckpt))
        else:
            logger.warning(f"B2 checkpoint not found at {ckpt}; using random weights.")
        return self._run_baseline_episode(dt, det, seed, "B2_LSTM")

    def _eval_rules(self, seed: int) -> dict:
        """B4: Rule-based MAS."""
        dt  = DigitalTwin(self.cfg, seed=seed)
        mas = RuleBasedMAS(num_edges=self.num_edges, num_zones=self.num_zones)
        return self._run_baseline_episode(dt, mas, seed, "B4_Rules")

    # ------------------------------------------------------------------
    # Episode runners
    # ------------------------------------------------------------------

    def _run_episode(self, dt, ma, ada, da, gov, seed: int,
                     use_gov: bool = True, label: str = "") -> dict:
        """
        One evaluation episode for AquaAgent or B3.

        FIX-01: Per-leak RT tracking; score/label alignment before dt.step().
        FIX-02: WLR uses no-action reference loss.
        """
        state = dt.reset()
        ma.reset()
        ada.eval()

        all_scores, all_labels = [], []
        n_approved, n_total = 0, 0
        detect_steps, response_steps = [], []
        system_loss = 0.0
        prev_leak_total = 0.0

        # FIX-01: per-leak first-detection dict.
        # Maps leaking edge_idx -> step at which ADA first flagged it.
        _leak_detected_at: dict = {}

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

            # FIX-01a: Record first detection step for each newly-flagged
            # leaking edge (ADA flag > 0 AND true leak present).
            for e_idx in np.where(state.leak_indicator > 0)[0]:
                e_idx = int(e_idx)
                if e_idx not in _leak_detected_at and flags[e_idx] > 0:
                    _leak_detected_at[e_idx] = step

            # FIX-01b: Record response step on first corrective action
            # (type 0 = isolate, 1 = adjust_valve) targeting a detected edge.
            action_type = int(exec_action.get("type", 3))
            if action_type in (0, 1):
                target_edge = int(exec_action.get("edge", -1))
                if target_edge in _leak_detected_at:
                    detect_steps.append(_leak_detected_at.pop(target_edge))
                    response_steps.append(step)

            # FIX-01c: Append before dt.step() — both vectors correspond to
            # the same hydraulic state at the decision timestep.
            all_scores.append(flags)
            all_labels.append(state.leak_indicator.astype(np.float32))

            state, done = dt.step(exec_action)

            curr_loss = dt.leak_injector.get_total_loss()
            system_loss += curr_loss - prev_leak_total
            prev_leak_total = curr_loss

            if done:
                break

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        # FIX-02: genuine no-action counterfactual baseline.
        baseline_loss = self._get_no_action_loss(seed)

        return compute_summary_metrics(
            scores=scores, labels=labels, threshold=ada.threshold,
            baseline_loss=max(baseline_loss, 1e-8), system_loss=system_loss,
            n_approved=n_approved if use_gov else n_total,
            n_total=max(n_total, 1),
            detection_steps=detect_steps, response_steps=response_steps,
        )

    def _run_baseline_episode(self, dt, detector, seed: int, label: str) -> dict:
        """
        One evaluation episode for a passive baseline (no actuation).

        FIX-01c: score/label alignment before dt.step().
        FIX-02:  WLR uses the no-action reference loss.
        """
        state = dt.reset()
        all_scores, all_labels = [], []
        system_loss = 0.0
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

            # FIX-01c: append before step.
            all_scores.append(scores)
            all_labels.append(state.leak_indicator.astype(np.float32))

            state, done = dt.step({"type": 3, "edge": 0, "valve": 0, "delta": 0.0})
            curr = dt.leak_injector.get_total_loss()
            system_loss += curr - prev_loss
            prev_loss = curr

            if done:
                break

        scores_arr = np.concatenate(all_scores)
        labels_arr = np.concatenate(all_labels)

        # FIX-02
        baseline_loss = self._get_no_action_loss(seed)

        return compute_summary_metrics(
            scores=scores_arr, labels=labels_arr, threshold=0.5,
            baseline_loss=max(baseline_loss, 1e-8), system_loss=system_loss,
            n_approved=0, n_total=1,
        )

    # ------------------------------------------------------------------
    # Reporting & statistics
    # ------------------------------------------------------------------

    def _aggregate_runs_all(
        self, all_results: Dict[str, List[dict]]
    ) -> Dict[str, dict]:
        """Aggregate across seeds; add statistical comparisons (FIX-12)."""
        aquaagent_runs = all_results.get("AquaAgent", [])
        summary: Dict[str, dict] = {}
        for method, runs in all_results.items():
            entry = self._aggregate_runs(runs)
            if method != "AquaAgent" and aquaagent_runs:
                entry.update(self._compute_significance(aquaagent_runs, runs))
            summary[method] = entry
        return summary

    def _aggregate_runs(self, runs: List[dict]) -> dict:
        """Compute mean +/- std across seeds."""
        result = {}
        keys = runs[0].keys() if runs else []
        for k in keys:
            vals = [r[k] for r in runs if k in r]
            result[f"{k}_mean"] = float(np.mean(vals))
            result[f"{k}_std"]  = float(np.std(vals))
        return result

    def _compute_significance(
        self,
        aq_runs: List[dict],
        baseline_runs: List[dict],
        metrics: tuple = ("f1", "auc_roc", "wlr", "rt_s"),
    ) -> dict:
        """
        FIX-12: Wilcoxon signed-rank p-value and Cohen's d for each metric.
        """
        try:
            from scipy import stats as sp_stats
        except ImportError:
            logger.warning("scipy unavailable — skipping significance tests.")
            return {}

        result: dict = {}
        for m in metrics:
            aq_v  = np.array([r.get(m, np.nan) for r in aq_runs])
            bl_v  = np.array([r.get(m, np.nan) for r in baseline_runs])
            mask = ~(np.isnan(aq_v) | np.isnan(bl_v))
            aq_v, bl_v = aq_v[mask], bl_v[mask]

            if len(aq_v) < 2:
                result[f"{m}_pvalue"]   = float("nan")
                result[f"{m}_cohens_d"] = float("nan")
                continue

            try:
                _, pvalue = sp_stats.wilcoxon(aq_v, bl_v, alternative="two-sided")
            except ValueError:
                pvalue = float("nan")

            diff = aq_v - bl_v
            sd = np.std(diff, ddof=1)
            cohens_d = float(np.mean(diff) / sd) if sd > 1e-12 else float("nan")
            result[f"{m}_pvalue"]   = float(pvalue)
            result[f"{m}_cohens_d"] = cohens_d

        return result

    def _print_table(self, summary: dict) -> None:
        """Print formatted results table with optional significance block."""
        header = (
            f"{'Method':<20} {'F1':>6} {'AUC':>6} {'Acc':>6} "
            f"{'WLR%':>6} {'PCR%':>6} {'RT(s)':>7}"
        )
        print("\n" + "=" * len(header))
        print("AquaAgent — Evaluation Results (mean +/- std across seeds)")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for method, metrics in summary.items():
            f1  = metrics.get("f1_mean",       0)
            auc = metrics.get("auc_roc_mean",  0)
            acc = metrics.get("accuracy_mean", 0)
            wlr = metrics.get("wlr_mean",      0) * 100
            pcr = metrics.get("pcr_mean",      0) * 100
            rt  = metrics.get("rt_s_mean",     0)
            print(
                f"{method:<20} "
                f"{f1:>6.4f} {auc:>6.4f} {acc:>6.4f} "
                f"{wlr:>6.1f} {pcr:>6.1f} {rt:>7.1f}"
            )
        print("=" * len(header))

        # Significance block (FIX-12)
        primary_metrics = ("f1", "auc_roc", "wlr", "rt_s")
        has_stats = any(
            f"{m}_pvalue" in v
            for v in summary.values()
            for m in primary_metrics
        )
        if has_stats:
            sig_hdr = "%-20s %-10s %10s %10s" % ("Method", "Metric", "p-value", "Cohen's d")
            print("\n--- Wilcoxon signed-rank vs AquaAgent ---")
            print(sig_hdr)
            print("-" * len(sig_hdr))
            for method, metrics in summary.items():
                if method == "AquaAgent":
                    continue
                for m in primary_metrics:
                    pval = metrics.get(f"{m}_pvalue")
                    cd   = metrics.get(f"{m}_cohens_d")
                    if pval is not None:
                        flag = "*" if not np.isnan(pval) and pval < 0.05 else " "
                        cd_str = f"{cd:.4f}" if not np.isnan(cd) else "  nan"
                        print(f"{method:<20} {m:<10} {pval:>10.4f}{flag} {cd_str:>10}")
        print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AquaAgent and baselines")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help=(
            "FIX-22: Fail fast if any required checkpoint is missing "
            "rather than silently using random weights."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluator = Evaluator(cfg, device=args.device, strict=args.strict)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
