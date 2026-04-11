# SUS Supply Chain Hybrid AI Benchmark

This repository contains code and supporting files for a simulation-based benchmark of inventory policies in a decentralized public healthcare supply-chain setting calibrated to the Brazilian Unified Health System (SUS).

The benchmark compares a hybrid AI architecture against tuned classical baselines under multiple demand regimes and planning horizons. The current public package focuses on the scenario-based Code4 v12 workflow and on the statistical validation pipeline used to summarize paired Monte Carlo results.

## Repository status

This repository is the public code package associated with the manuscript:

**Economic Evaluation of Hybrid AI and Classical Inventory Policies in a Healthcare Supply Chain**

The public package contains runnable code, shell runners, environment specifications, example outputs, and reproducibility utilities. It does not contain restricted SUS raw operational data or every large audit bundle generated during execution.

## Scope of the benchmark

The study evaluates a hybrid policy that combines:

- machine-learning demand forecasting
- reinforcement-learning inventory control
- genetic-algorithm routing-cost optimization

The benchmark compares this policy against classical baselines under scenario-defined demand regimes, with paired Monte Carlo replications generated from common random numbers.

## Canonical entry point

The current canonical runner is:

`Code4_colab_fixed_v12_actionscale_minpatch.py`

This script is the main scenario-based implementation used for the current benchmark structure. It supports:

- scenario-level experiment grids
- configurable horizons
- warm-up and evaluation windows
- paired replication logic
- Wilcoxon signed-rank tests with Holm correction
- checkpointing
- reproducibility fingerprints
- routing ablations
- ranking-stability summaries

## Other key files

- `Code4_colab_fixed_v12_progress_checkpoint_sigint.py`  
  Operational variant of the main runner with progress reporting and interrupt-safe checkpointing.

- `Code_compare_fingerprints.py`  
  Utility for comparing run fingerprints and exported outputs across runs.

- `compute_tost_mde_withinrun.py`  
  Utility for within-run TOST equivalence tests and minimum detectable effects.

- `legacy/FinalCode_unit_level_montecarlo_legacy.py`  
  Legacy script retained for provenance. It reflects an earlier unit-level Monte Carlo workflow and should not be treated as the canonical runner for the current manuscript state.

## Repository layout

- `Code4_colab_fixed_v12_actionscale_minpatch.py` — main benchmark runner
- `Code4_colab_fixed_v12_progress_checkpoint_sigint.py` — progress/checkpoint runner
- `Code_compare_fingerprints.py` — fingerprint comparison utility
- `compute_tost_mde_withinrun.py` — within-run equivalence/MDE utility
- `scripts/` — shell runners for the main grid and targeted diagnostics
- `configs/` — example configuration file(s)
- `requirements.txt` — Python dependencies
- `database.csv` — public example dataset file currently included in the repository
- `raw_replications.csv` — example exported raw replication output
- `summary_means_ci.csv` — example exported summary output
- `wilcoxon_holm.csv` — example exported inferential summary
- `CHANGELOG.md` — package change log
- `CITATION.cff` — citation metadata

## Scripts currently included in `scripts/`

- `run_phaseA2_grid.sh`
- `run_diag_supply_disruption_quick.sh`
- `run_ablations_sd_quick.sh`
- `smoke_stable_90.sh`

## Configuration currently included in `configs/`

- `example.yml`

## Experimental design reflected in the current codebase

The current public runner is organized around four scenario labels:

- `STABLE`
- `HIGH_VOLATILITY`
- `SUPPLY_DISRUPTION`
- `SEASONAL_SURGE`

The benchmark supports configurable horizons and separates warm-up days from the evaluation window.

The main policy set includes classical baselines and a hybrid AI controller. Exact policy labels are written to the exported run outputs.

## Statistical reporting reflected in the current codebase

The current public runner and utilities are organized around:

- paired replications with common random numbers
- Wilcoxon signed-rank comparisons
- Holm correction within comparison families
- bootstrap confidence intervals in exported summaries
- optional equivalence testing through `compute_tost_mde_withinrun.py`

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
Minimal smoke test:
python -u Code4_colab_fixed_v12_actionscale_minpatch.py \
  --quick \
  --scenarios STABLE \
  --horizons 90 \
  --forecasters lstm_rf \
  --reward_modes inventory_only \
  --ga_modes off \
  --out_dir runs/test_quickShell-based runs are also provided in scripts/.

## Typical outputs from the main runner

The main runner can generate files such as:
- `raw_replications.csv`
- `summary_means_ci.csv`
- `wilcoxon_holm.csv`
- `routing_ablation.csv`
- `forecast_ranking_stability.csv`
- `run_manifest.json`
- `run_fingerprint.json`

Some shell pipelines also generate pooled tail-risk tables and Word-ready CSV exports during post-processing.

-## Relationship to the manuscript
The public package is intended to support the manuscript’s paired simulation workflow and its statistical validation pipeline.
In manuscript terms, the repository should be read as the public code package for the current scenario-based benchmark and related validation utilities. The legacy script is preserved for traceability, not as the preferred entry point for new runs.

-## Data availability
Raw SUS operational transaction data are not distributed in this repository.
The manuscript uses operational demand records from SUS institutions under data-use restrictions. Publicly shared materials in this repository are intended to support code inspection, workflow replication, and validation of exported numerical summaries, but they do not include the restricted institutional raw logs.

## Naming policy
# To avoid ambiguity:
•	treat Code4_colab_fixed_v12_actionscale_minpatch.py as the canonical runner 
•	treat legacy/FinalCode_unit_level_montecarlo_legacy.py as a legacy artifact 
•	avoid labels such as Final, Last, or similar for future canonical files 
•	use descriptive filenames tied to function and workflow stage 

##Citation
Please use the repository citation metadata in CITATION.cff and cite the associated manuscript when referring to this codebase in academic work.

## License
This repository is distributed under the license included in LICENSE.
