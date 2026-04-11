# SUS Supply Chain Hybrid AI Benchmark (ML + RL + GA)

This repository provides code and documentation to reproduce the simulation-based evaluation reported in the published manuscript.

## Scope

The study benchmarks a hybrid policy that combines:
- demand forecasting (ML),
- inventory control via reinforcement learning (RL),
- vehicle-routing optimization via a genetic algorithm (GA),

against multi-tier baselines (static, dynamic, and forecast-driven OR policies). The simulator targets a decentralized public-sector setting and reports costs and service outcomes under multiple demand regimes.

## Included in this reviewed package

- `src/`: source files to the execution environment/server.
- `scripts/`: runnable shells for the main experiment grid and targeted diagnostics/ablations.
- `configs/`: example configuration stub.

## Not included in the current workspace

- Raw SUS operational data.
- Large audit bundles (`*.tar.gz`) and run folders.
- Core source files copied from the server.
- Validation CSVs, synthetic-parameter files, and table-generation/post-processing scripts.
