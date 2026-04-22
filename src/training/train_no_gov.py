"""
src/training/train_no_gov.py
------------------------------
MAPPO training for the B3 (No-Governance) baseline Decision Agent.

FIX-03 (Reviewer 1 Issue 3 / Reviewer 2 Moderate Issue 3):
  The original repository expected checkpoints/da_no_gov_best.pt but provided
  no script to generate it.  This script trains a DA with governance disabled
  (gamma_r = 0.0, governance.enabled = False), producing a reproducible B3
  checkpoint that is separate from the AquaAgent DA checkpoint.

Usage:
    bash scripts/run_train_no_gov.sh
    # or directly:
    python -m src.training.train_no_gov --config configs/default.yaml --seed 42
"""

from __future__ import annotations

import argparse
import copy

import torch

from src.data.simulate import load_config
from src.training.train_mappo import MAPPOTrainer
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger("train_no_gov")


def build_no_gov_cfg(base_cfg: dict) -> dict:
    """
    Return a deep copy of base_cfg modified for the B3 no-governance regime:
      - governance.enabled: False  (skips constraint checking)
      - mappo.ppo.gamma_r: 0.0    (no governance reward penalty)
      - checkpoints saved to da_no_gov_best.pt / da_no_gov_last.pt

    This keeps the MAPPO training loop identical to the AquaAgent run;
    only the governance flag and checkpoint filenames differ.
    """
    cfg = copy.deepcopy(base_cfg)

    # Disable governance agent enforcement.
    if "governance" not in cfg:
        cfg["governance"] = {}
    cfg["governance"]["enabled"] = False

    # Zero out governance penalty weight so B3 is not penalised for violations.
    mappo = cfg.setdefault("mappo", {})
    ppo   = mappo.setdefault("ppo", {})
    ppo["gamma_r"] = 0.0

    # Redirect checkpoint names so B3 does not overwrite AquaAgent checkpoints.
    import os
    ckpt_dir = cfg.get("paths", {}).get("checkpoints_dir", "checkpoints")
    cfg["_no_gov_best_ckpt"] = os.path.join(ckpt_dir, "da_no_gov_best.pt")
    cfg["_no_gov_last_ckpt"] = os.path.join(ckpt_dir, "da_no_gov_last.pt")

    return cfg


class NoGovMAPPOTrainer(MAPPOTrainer):
    """
    Thin subclass of MAPPOTrainer that:
      1. Disables governance enforcement in the rollout.
      2. Saves checkpoints to da_no_gov_best.pt instead of da_best.pt.
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        super().__init__(cfg, device=device)
        from pathlib import Path
        ckpt_dir = Path(cfg.get("paths", {}).get("checkpoints_dir", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt = str(ckpt_dir / "da_no_gov_best.pt")
        self.last_ckpt = str(ckpt_dir / "da_no_gov_last.pt")
        logger.info(
            "NoGovMAPPOTrainer: governance disabled, "
            f"checkpoint -> {self.best_ckpt}"
        )

    def _governance_step(self, action, state):
        """Override: always pass the raw action through without GA validation."""
        # Return action as-is; overridden=False; PCR not tracked for B3.
        return action, False


def main():
    parser = argparse.ArgumentParser(
        description="Train B3 No-Governance MAPPO baseline"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    cfg      = build_no_gov_cfg(base_cfg)
    set_seed(args.seed)

    logger.info("Starting B3 No-Governance MAPPO training ...")
    trainer = NoGovMAPPOTrainer(cfg, device=args.device)
    trainer.train()
    logger.info(
        f"B3 training complete. Checkpoint: {trainer.best_ckpt}"
    )


if __name__ == "__main__":
    main()
