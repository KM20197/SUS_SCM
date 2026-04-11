#!/usr/bin/env bash
set -euo pipefail
TS=$(date -u +%Y%m%d_%H%M%S)
OUT_DIR="runs/diag_sd_${TS}"
mkdir -p "${OUT_DIR}"

export CODE4_ABLATION=none

python3 -u src/Code4_colab_fixed_v12_actionscale_minpatch.py \
  --scenarios SUPPLY_DISRUPTION \
  --horizons 180 \
  --forecasters lstm_rf \
  --reward_modes inventory_only \
  --ga_modes off \
  --quick \
  --out_dir "${OUT_DIR}" \
  --log_every 25 \
  --checkpoint_every 50

echo "DONE: ${OUT_DIR}"
