#!/usr/bin/env bash
# scripts/generate_data.sh
# Generate the full 365-day simulation dataset.
# Usage: bash scripts/generate_data.sh [--days 365] [--seed 42]

set -euo pipefail

DAYS=${1:-365}
SEED=${2:-42}
CONFIG="configs/default.yaml"

echo "============================================="
echo " AquaAgent — Data Generation"
echo " Days:   $DAYS"
echo " Seed:   $SEED"
echo " Config: $CONFIG"
echo "============================================="

python -m src.data.simulate \
    --config "$CONFIG" \
    --days   "$DAYS"   \
    --seed   "$SEED"   \
    --output-dir data/generated \
    --splits-dir data/splits

echo "[✓] Dataset generation complete."
echo "    Output: data/generated/simulation.{h5,npz}"
echo "    Splits: data/splits/{train,val,test}_idx.npy"
