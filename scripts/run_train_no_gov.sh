#!/usr/bin/env bash
# scripts/run_train_no_gov.sh
# MAPPO training for the B3 No-Governance baseline Decision Agent.
#
# FIX-03 (Reviewer 1 Issue 3 / Reviewer 2 Moderate Issue 3):
#   This script was missing from the original repository.  It produces
#   checkpoints/da_no_gov_best.pt which is required by evaluate.py for the B3
#   column in Table 1.
#
# Differences from run_train_mappo.sh:
#   - governance.enabled is set to False (no constraint enforcement).
#   - mappo.ppo.gamma_r is set to 0.0 (no governance penalty in reward).
#   - Output checkpoint: checkpoints/da_no_gov_best.pt (not da_best.pt).
#
# Usage: bash scripts/run_train_no_gov.sh [SEED] [DEVICE]
#   SEED   — random seed (default: 42)
#   DEVICE — 'cuda' or 'cpu' (default: cuda)

set -euo pipefail

SEED=${1:-42}
DEVICE=${2:-cuda}
CONFIG="configs/default.yaml"

echo "============================================="
echo " AquaAgent — B3 No-Governance MAPPO Training"
echo " Seed:   $SEED"
echo " Device: $DEVICE"
echo " Config: $CONFIG"
echo "============================================="

# Verify ADA checkpoint exists (B3 reuses the same ADA for detection)
if [ ! -f "checkpoints/ada_best.pt" ]; then
    echo "[!] No ADA checkpoint found. Running ADA training first..."
    bash scripts/run_train_ada.sh "$SEED" "$DEVICE"
fi

python -m src.training.train_no_gov \
    --config "$CONFIG" \
    --seed   "$SEED"   \
    --device "$DEVICE"

echo "[OK] B3 No-Governance MAPPO training complete."
echo "     Best checkpoint: checkpoints/da_no_gov_best.pt"
echo "     Last checkpoint: checkpoints/da_no_gov_last.pt"
echo "     TensorBoard logs: logs/no_gov/"
