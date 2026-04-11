#!/usr/bin/env bash
set -euo pipefail
TS=$(date -u +%Y%m%d_%H%M%S)
OUT_DIR="runs/final_A2_${TS}"
mkdir -p "${OUT_DIR}"

python3 -u src/Code4_colab_fixed_v12_actionscale_minpatch.py \
  --scenarios STABLE,HIGH_VOLATILITY,SUPPLY_DISRUPTION,SEASONAL_SURGE \
  --horizons 90,180 \
  --forecasters lstm_rf \
  --reward_modes inventory_plus_routing \
  --ga_modes on,off \
  --out_dir "${OUT_DIR}" \
  --log_every 500 \
  --checkpoint_every 2000

echo "DONE: ${OUT_DIR}"
