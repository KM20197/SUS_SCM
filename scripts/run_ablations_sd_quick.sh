#!/usr/bin/env bash
set -euo pipefail
TS=$(date -u +%Y%m%d_%H%M%S)
ROOT="runs/abl_sd_${TS}"
mkdir -p "${ROOT}"

# no_bonus
mkdir -p "${ROOT}/abl_no_bonus"
export CODE4_ABLATION=no_bonus
python3 -u src/Code4_colab_fixed_v12_actionscale_minpatch.py \
  --scenarios SUPPLY_DISRUPTION \
  --horizons 180 \
  --forecasters lstm_rf \
  --reward_modes inventory_plus_routing \
  --ga_modes on \
  --quick \
  --out_dir "${ROOT}/abl_no_bonus" \
  --log_every 25 \
  --checkpoint_every 50

# wide_action (mult=3)
mkdir -p "${ROOT}/abl_wide_action"
export CODE4_ABLATION=wide_action
export CODE4_WIDE_ACTION_MULT=3
python3 -u src/Code4_colab_fixed_v12_actionscale_minpatch.py \
  --scenarios SUPPLY_DISRUPTION \
  --horizons 180 \
  --forecasters lstm_rf \
  --reward_modes inventory_plus_routing \
  --ga_modes on \
  --quick \
  --out_dir "${ROOT}/abl_wide_action" \
  --log_every 25 \
  --checkpoint_every 50

# fallback
mkdir -p "${ROOT}/abl_fallback"
export CODE4_ABLATION=fallback
export CODE4_FALLBACK_THETA=0.90
export CODE4_FALLBACK_K=3
export CODE4_FALLBACK_M=5
export CODE4_FALLBACK_Z=2.0
python3 -u src/Code4_colab_fixed_v12_actionscale_minpatch.py \
  --scenarios SUPPLY_DISRUPTION \
  --horizons 180 \
  --forecasters lstm_rf \
  --reward_modes inventory_plus_routing \
  --ga_modes on \
  --quick \
  --out_dir "${ROOT}/abl_fallback" \
  --log_every 25 \
  --checkpoint_every 50

echo "DONE: ${ROOT}"
