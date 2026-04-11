#!/usr/bin/env bash
set -euo pipefail
TS=$(date -u +%Y%m%d_%H%M%S)
OUT_DIR="runs/smoke_${TS}"
mkdir -p "${OUT_DIR}"

python3 -u src/Code4_colab_fixed_v12_actionscale_minpatch.py \
  --scenarios STABLE \
  --horizons 90 \
  --forecasters lstm_rf \
  --reward_modes inventory_plus_routing \
  --ga_modes off \
  --quick \
  --out_dir "${OUT_DIR}" \
  --log_every 100 \
  --checkpoint_every 200

echo "DONE: ${OUT_DIR}"
