#!/usr/bin/env bash
# scripts/run_evaluate.sh
# Run full evaluation of AquaAgent and all baselines.
# Usage: bash scripts/run_evaluate.sh [--device cuda]

set -euo pipefail

DEVICE=${1:-cuda}
CONFIG="configs/default.yaml"

echo "============================================="
echo " AquaAgent — Full Evaluation"
echo " Device: $DEVICE"
echo " Config: $CONFIG"
echo "============================================="

python -m src.evaluation.evaluate \
    --config "$CONFIG" \
    --device "$DEVICE"

echo "[✓] Evaluation complete."
echo "    Results table: logs/results.json"
echo "    Audit ledger:  logs/audit_ledger.csv"
