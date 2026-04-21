#!/usr/bin/env bash
# scripts/run_train_mappo.sh
# MAPPO multi-agent RL training of the Decision Agent.
# Usage: bash scripts/run_train_mappo.sh [--seed 42] [--device cuda]

set -euo pipefail

SEED=${1:-42}
DEVICE=${2:-cuda}
CONFIG="configs/default.yaml"

echo "============================================="
echo " AquaAgent — MAPPO Training"
echo " Seed:   $SEED"
echo " Device: $DEVICE"
echo " Config: $CONFIG"
echo "============================================="

# Verify ADA checkpoint exists
if [ ! -f "checkpoints/ada_best.pt" ]; then
    echo "[!] No ADA checkpoint found. Running ADA training first..."
    bash scripts/run_train_ada.sh "$SEED" "$DEVICE"
fi

python -m src.training.train_mappo \
    --config "$CONFIG" \
    --seed   "$SEED"   \
    --device "$DEVICE"

echo "[✓] MAPPO training complete."
echo "    Best checkpoint: checkpoints/da_best.pt"
echo "    Last checkpoint: checkpoints/da_last.pt"
echo "    TensorBoard logs: logs/mappo/"
