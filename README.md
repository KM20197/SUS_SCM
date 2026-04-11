# SUS Supply Chain Hybrid AI Benchmark (ML + RL + GA)

This repository provides code and documentation to reproduce the simulation-based evaluation reported in the manuscript.

## Status of this package

This reviewed package replaces the manuscript included in the original scaffold with the current version dated 2026-04-01 and keeps the previous manuscript in `paper/archive/` for traceability.

The package remains **not fully runnable yet** because the core source files and some manuscript-declared public materials are not present in the current workspace. See:
- `src/REQUIRED_FROM_SERVER.txt`
- `docs/PACKAGE_STATUS_2026-04-07.md`
- `docs/RELEASE_CHECKLIST.md`

## Scope

The study benchmarks a hybrid policy that combines:
- demand forecasting (ML),
- inventory control via reinforcement learning (RL),
- vehicle-routing optimization via a genetic algorithm (GA),

against multi-tier baselines (static, dynamic, and forecast-driven OR policies). The simulator targets a decentralized public-sector setting and reports costs and service outcomes under multiple demand regimes.

## Included in this reviewed package

- `src/`: placeholder plus a precise list of source files still to be copied from the execution environment/server.
- `scripts/`: runnable shells for the main experiment grid and targeted diagnostics/ablations.
- `configs/`: example configuration stub.
- `docs/`: reproducibility notes, data statement, audit-artifact guide, release checklist, and package-status memo.

## Not included in the current workspace

- Raw SUS operational data.
- Large audit bundles (`*.tar.gz`) and run folders.
- Core source files copied from the server.
- Validation CSVs, synthetic-parameter files, and table-generation/post-processing scripts referenced in the manuscript but not present among the files available here.

## Quick start (after source files are added)

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run a minimal smoke test:

```bash
bash scripts/smoke_stable_90.sh
```

## Reproducing the main results

Primary grid:

```bash
bash scripts/run_phaseA2_grid.sh
```

Targeted diagnostics and ablations (Supply Disruption):

```bash
bash scripts/run_diag_supply_disruption_quick.sh
bash scripts/run_ablations_sd_quick.sh
```

See `docs/REPRODUCIBILITY.md` for operational details and expected outputs.

## License

See `LICENSE`. If you plan to reuse code in a different licensing model, separate code and manuscript licensing explicitly.
