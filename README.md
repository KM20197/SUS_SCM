# SUS Supply Chain Hybrid AI Benchmark

This repository contains code and documentation for a simulation-based benchmark of inventory policies in a decentralized public healthcare supply-chain setting.

The benchmark compares a hybrid AI architecture against classical baselines under multiple demand regimes and decision horizons.

## Current repository status

This public repository is a reviewed and partial replication package.

It already contains runnable Python scripts, shell runners, summary CSV outputs, and reproducibility utilities. It does **not** yet contain every source file, audit bundle, and public-facing artifact referenced across the manuscript history.

## Canonical script status

The repository currently contains more than one large Python script. They do not represent the same code generation stage.

### Primary runner

`Code4_colab_fixed_v12_actionscale_minpatch.py`

This is the current canonical scenario-based runner for the benchmark. It implements:

- scenario-level simulation blocks
- configurable horizons
- warm-up and evaluation windows
- paired replication logic
- Wilcoxon + Holm post-processing
- checkpointing and run fingerprints
- routing ablation outputs
- ranking-stability outputs

### Progress/checkpoint variant

`Code4_colab_fixed_v12_progress_checkpoint_sigint.py`

Operational variant of the main runner for long executions with progress and interrupt-safe checkpointing.

### Reproducibility utility

`Code_compare_fingerprints.py`

Utility for comparing run fingerprints and exported outputs across runs.

### Within-run equivalence utility

`compute_tost_mde_withinrun.py`

Computes within-run TOST equivalence tests and minimum detectable effects from `raw_replications.csv` outputs.

## Legacy script

`FinalCode.py`

This file is retained for historical traceability. It should **not** be treated as the canonical runner for the current manuscript state.

It reflects an earlier unit-level Monte Carlo workflow with different analysis and reporting conventions from the current scenario-based Code4 v12 runner.

Recommended interpretation:

- keep for provenance
- do not cite as the main experimental entry point
- do not use as the default runner for new experiments

## Repository structure

- `Code4_colab_fixed_v12_actionscale_minpatch.py`  
  Main benchmark runner

- `Code4_colab_fixed_v12_progress_checkpoint_sigint.py`  
  Operational runner with progress/checkpoint behavior

- `Code_compare_fingerprints.py`  
  Output-comparison utility

- `compute_tost_mde_withinrun.py`  
  Within-run equivalence and MDE utility

- `code4_tf_predict_safe_patch.py`  
  Safe prediction helper for the LSTM+RF forecaster; modularized fallback logic

- `scripts/`  
  Shell runners for experiment grids and targeted diagnostics

- `configs/`  
  Example configuration stub(s)

- `summary_means_ci.csv`, `wilcoxon_holm.csv`, `raw_replications.csv`  
  Example exported outputs currently present in the public package

## Benchmark design summary

The benchmark evaluates policies under scenario-defined demand regimes, including:

- `STABLE`
- `HIGH_VOLATILITY`
- `SUPPLY_DISRUPTION`
- `SEASONAL_SURGE`

The current Code4 v12 runner supports configurable horizons and separates warm-up days from the evaluation window.

The main policy set includes classical baselines and a hybrid AI policy. Exact policy labels depend on the exported run outputs.

## Statistical reporting used by the current runner

The current canonical runner is aligned with:

- paired replications
- Wilcoxon signed-rank comparisons
- Holm correction within comparison families
- bootstrap confidence intervals in exported summaries
- optional equivalence testing through `compute_tost_mde_withinrun.py`

## Example execution

Minimal quick test:

```bash
python -u Code4_colab_fixed_v12_actionscale_minpatch.py \
  --quick \
  --scenarios STABLE \
  --horizons 90 \
  --forecasters lstm_rf \
  --reward_modes inventory_only \
  --ga_modes off \
  --out_dir runs/test_quick
