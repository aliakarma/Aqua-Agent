#!/usr/bin/env bash
# scripts/run_train_ada.sh
# Supervised pre-training of the Anomaly Detection Agent.
# Usage: bash scripts/run_train_ada.sh [--seed 42] [--device cuda]

set -euo pipefail

SEED=${1:-42}
DEVICE=${2:-cuda}
CONFIG="configs/default.yaml"

echo "============================================="
echo " AquaAgent — ADA Pre-Training"
echo " Seed:   $SEED"
echo " Device: $DEVICE"
echo " Config: $CONFIG"
echo "============================================="

# Verify dataset exists
if [ ! -f "data/generated/simulation.h5" ] && [ ! -f "data/generated/simulation.npz" ]; then
    echo "[!] No dataset found. Running data generation first..."
    bash scripts/generate_data.sh 365 "$SEED"
fi

python -m src.training.train_ada \
    --config "$CONFIG" \
    --seed   "$SEED"   \
    --device "$DEVICE"

echo "[✓] ADA pre-training complete."
echo "    Best checkpoint: checkpoints/ada_best.pt"
echo "    Last checkpoint: checkpoints/ada_last.pt"
