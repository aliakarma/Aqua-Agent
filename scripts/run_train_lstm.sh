#!/usr/bin/env bash
# scripts/run_train_lstm.sh
# Supervised training of the B2 Centralised LSTM baseline detector.
#
# FIX-03 (Reviewer 1 Issue 3 / Reviewer 2 Moderate Issue 3):
#   This script was missing from the original repository.  It produces
#   checkpoints/lstm_best.pt which is required by evaluate.py for the B2
#   column in Table 1.
#
# Usage: bash scripts/run_train_lstm.sh [SEED] [DEVICE]
#   SEED   — random seed (default: 42)
#   DEVICE — 'cuda' or 'cpu' (default: cuda)

set -euo pipefail

SEED=${1:-42}
DEVICE=${2:-cuda}
CONFIG="configs/default.yaml"

echo "============================================="
echo " AquaAgent — B2 LSTM Baseline Training"
echo " Seed:   $SEED"
echo " Device: $DEVICE"
echo " Config: $CONFIG"
echo "============================================="

# Verify dataset exists
if [ ! -f "data/generated/simulation.h5" ] && [ ! -f "data/generated/simulation.npz" ]; then
    echo "[!] No dataset found. Running data generation first..."
    bash scripts/generate_data.sh 365 "$SEED"
fi

python -m src.training.train_lstm \
    --config "$CONFIG" \
    --seed   "$SEED"   \
    --device "$DEVICE"

echo "[OK] B2 LSTM training complete."
echo "     Best checkpoint: checkpoints/lstm_best.pt"
echo "     Last checkpoint: checkpoints/lstm_last.pt"
echo "     TensorBoard logs: logs/lstm/"
