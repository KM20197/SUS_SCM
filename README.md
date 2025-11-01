# A Hybrid AI Architecture for Healthcare Supply Chain Optimization

This repository contains the official source code for: **"A Hybrid Artificial Intelligence Architecture for Healthcare Supply Chain Optimization: Benchmarking Machine Learning, Reinforcement Learning, and Genetic Algorithms Against Operations Research Baselines."**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![DOI: 10.5281/zenodo.17420367](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17420367-blue.svg)](https://doi.org/10.5281/zenodo.17420367)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Quick Start

```bash
git clone https://github.com/your-org/hybrid-ai-supply-chain.git
cd hybrid-ai-supply-chain
pip install -r requirements.txt
python FinalCode.py
```
**Runtime:** 40-45 minutes | **Output:** `outputs/` | **Tests:** `python tests/validation_tests.py`

## Overview

A hybrid AI framework combining **Machine Learning (LSTM-RF ensemble)**, **Reinforcement Learning (Double Q-learning)**, and **Genetic Algorithms (VRP optimization)** for healthcare supply chain management. Evaluated against classical Operations Research baselines using 100 Monte Carlo replications on 9,170 real SUS (Brazil's Unified Health System) transactions.

**Key Results:**
- AI achieves **65.3% cost reduction** vs. static baseline (p < 0.0001)
- Maintains **99.25% fill rate** (service level)
- Significantly outperforms analytical base-stock policy (p = 0.006, Bonferroni-corrected)

## Contents

| Component | Description | Status |
|-----------|-------------|--------|
| **FinalCode.py** | Main simulation framework (5 policies, 100 MC replications) | ✅ v1.1.0 (corrected) |
| **data/database.csv** | Real SUS data (9,170 transactions, 365 days, 3 facilities) | ✅ Included |
| **tests/validation_tests.py** | Policy specification validation suite (4 tests) | ✅ Complete |
| **README.md** | Complete documentation | ✅ v1.1.0 |
| **LICENSE** | CC BY-NC-SA 4.0 | ✅ Included |

## Framework Architecture

### Core Components

**1. Machine Learning: LSTM-RF Ensemble**
- 14-day sequence modeling with LSTM (64 units)
- Non-linear feature interactions via Random Forest (120 estimators)
- Probabilistic forecasts with uncertainty quantification (σ_forecast)

**2. Reinforcement Learning: Double Q-Learning**
- State space: 100 (inventory × forecast uncertainty)
- Actions: 5 ordering levels
- Training: 100 episodes with adaptive exploration (ε: 0.2 → 0.02)

**3. Genetic Algorithm: Vehicle Routing Problem**
- Population: 30, Generations: 12 (1.5s budget)
- Minimizes distance + time subject to vehicle capacity
- Integrates with inventory for joint optimization

### Baseline Policies (Fair Comparison)

All policies receive **identical ML forecasts** and operate under **identical constraints**:

| Policy | Mechanism | Specification |
|--------|-----------|---------------|
| **STATIC** | Fixed reorder point | ROP = μ×LT + z×σ×√LT (z=1.96) |
| **DYNAMIC** | Exponential smoothing | Adaptive forecast with α=0.2, β=0.1 |
| **BASE_STOCK_PERFECT** | Omniscient policy | Knows actual demand (theoretical optimum) |
| **BASE_STOCK_UNCERTAIN** | ML-guided analytics | S = μ_forecast + 1.645×σ_forecast×√(LT+1), LT~N(7.2,1.8) |
| **AI (RL-based)** | Adaptive learning | Double Q-learning with integrated routing |

**Key difference (v1.1.0):** BASE_STOCK_UNCERTAIN corrected to use z_α=1.645 (95% service level) and stochastic lead time (empirical SUS: mean 7.2 days, SD 1.8 days).

## Installation

```bash
# 1. Clone
git clone https://github.com/your-org/hybrid-ai-supply-chain.git
cd hybrid-ai-supply-chain

# 2. Virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
python -c "import tensorflow; import deap; print('✓ Ready')"
```

## Usage

### Run Full Simulation

```bash
python FinalCode.py
```

Automatically:
- Loads included SUS dataset (9,170 records)
- Trains LSTM-RF ensemble + RL agent (train set through 2021-06-30)
- Runs 100 MC replications on test set (2021-07-01 to 2021-09-29)
- Compares 5 policies with Mann-Whitney U tests + Bonferroni correction
- Saves results to `outputs/`

**Expected output:**
```
DESCRIPTIVE STATISTICS (90-day normalized horizon):
static                : Cost=3,604,269,457.74±2,656,652,678.32 FR=0.9604
dynamic               : Cost=50,199,985.26±36,709,204.24 FR=0.9613
ai                    : Cost=1,251,921,848.98±1,367,618,166.74 FR=0.9925
base_stock_perfect    : Cost=1,156,791,065.38±909,960,418.79 FR=0.9972
base_stock_uncertain  : Cost=2,064,134,674.15±1,769,277,269.20 FR=0.9920

STATISTICAL TESTS (Mann-Whitney U + Glass's Delta):
ai vs base_stock_uncertain: U=50834 p=0.006002 ✓ SIG Glass's Δ=0.5128
```

### Run Validation Tests Only

Verify policy specifications without full 40-minute simulation:

```bash
python tests/validation_tests.py
```

Tests validate:
1. **TEST 1:** BASE_STOCK_UNCERTAIN (z_α=1.645, LT~N(7.2,1.8), fair comparison)
2. **TEST 2:** Stress scenarios (CV 0.42→0.63, disruption 20%, surge 180-220%)
3. **TEST 3:** Fair comparison (identical forecasts & constraints for AI vs. BASE_STOCK)
4. **TEST 4:** Empirical grounding (HIGH_VOL vs H1N1 0.61/COVID 0.68, DISRUPTION vs COVID-19 Q2 18-23%, SURGE vs flu 195-240%)

### Use Your Own Data

```bash
export DATA_PATH=/path/to/your/data.csv
python FinalCode.py
```

**Required columns:** DATA, ALMOXARIFADO, CODIGO, QUANTIDADE, UNITCOST, LATITUDE, LONGITUDE

## Dataset

**Included:** `data/database.csv`
- **Records:** 9,170 transactions
- **Period:** 365 days (2021-01-01 to 2021-12-31)
- **Facilities:** 3 Brazilian SUS units (City01, City02, City03)
- **SKUs:** 50 distinct medications
- **Source:** State of Bahia Health Secretariat (SESAB) + Federal University Hospital (HUPES)
- **Privacy:** Facility-SKU-date aggregation; no patient identifiers; LGPD-compliant

## Output Files

| File | Description |
|------|-------------|
| **checkpoint.csv** | 300+ rows (MC replications × policies × facilities) |
| **results_summary.csv** | Aggregate statistics per policy |
| **statistical_tests.csv** | Mann-Whitney U results + effect sizes |
| **[unit]_plots.png** | Cost distributions by facility |

## v1.1.0 Updates (November 2025)

**Implementation Corrections:**
- ✅ z_score corrected: 1.96 → 1.645 (95% service level per SUS equity objectives)
- ✅ Lead time stochastic: Fixed 5d → Normal(μ=7.2, σ=1.8) empirical SUS distribution
- ✅ Fair comparison validated: AI and BASE_STOCK_UNCERTAIN receive identical forecasts
- ✅ Empirical grounding: Stress scenarios validated against H1N1 (2009), COVID-19 (2020), Zika (2016)

**Impact:** Results stable (AI 65.3% reduction remains); baseline policies now rigorously specified and empirically grounded.

## Statistical Framework

- **Primary test:** Mann-Whitney U (non-parametric, robust to non-normal distributions)
- **Multiple comparisons:** Bonferroni correction (α_adjusted = 0.017 for 6 pairwise comparisons)
- **Effect size:** Glass's Δ with Cohen interpretation (|Δ| 0.5-0.8 = medium)
- **Confidence intervals:** Bootstrap percentile method (1,000 resamples, 95% CI)
- **Replication:** 100 Monte Carlo iterations per facility

## Configuration

**ML Ensemble (LSTMRFEnsemble):**
- LSTM: 64 units, dropout 0.3, sequence length 14 days
- RF: 120 estimators, max depth 12
- Early stopping: patience=3

**RL Agent (Double Q-Learning):**
- States: 100 | Actions: 5 | Learning rate: 0.1 | Discount: 0.95
- Training: 100 episodes, exploration decay (ε: 0.2→0.02)

**GA/VRP (DEAP):**
- Population: 30 | Generations: 12 | Timeout: 1.5s per optimization

**Simulation:**
- MC replications: 100 per facility
- Horizon: 90 days (2021-07-01 to 2021-09-29)
- Random seed: 42 (deterministic reproducibility)
- Bonferroni α: 0.017

## Reproducibility

```bash
# Obtain exact manuscript results:
# 1. Use included data/database.csv (unchanged)
# 2. Keep all parameters at defaults
# 3. Run with Python 3.8+
# 4. Fixed seed=42 ensures determinism
# 5. Results identical across runs (verified)
```

To verify reproducibility:
```bash
python FinalCode.py  # First run
python FinalCode.py  # Second run → results identical
```

## Citation

```bibtex
@article{Mello2025HybridAI,
  title   = {A Hybrid Artificial Intelligence Architecture for Healthcare Supply Chain Optimization: Benchmarking Machine Learning, Reinforcement Learning, and Genetic Algorithms Against Operations Research Baselines},
  author  = {Mello, Ricardo Coutinho and Chen, Bo and Fernandes, Felipe Schuler and Claro, Daniela Barreiro and Fernandes, Antônio Sérgio Araújo and Ladeira, Rodrigo},
  year    = {2025},
  publisher = {Zenodo},
  version = {v1.1.0},
  doi     = {10.5281/zenodo.17420367},
  url     = {https://doi.org/10.5281/zenodo.17420367}
}
```

## Authors

- **Ricardo Coutinho Mello** — Lead author; methodology, AI architecture, manuscript
- **Bo Chen** — Advisor; manuscript coordination
- **Felipe Schuler Fernandes** — Co-author; validation tests, code implementation
- **Daniela Barreiro Claro** — Co-author; RL design, AI/ML integration
- **Antônio Sérgio Araújo Fernandes** — Advisor; public health policy, SUS context
- **Rodrigo Ladeira** — Advisor; logistics domain, VRP optimization

## Support

**Issues or questions?**
- Check [GitHub Issues](https://github.com/your-org/hybrid-ai-supply-chain/issues)
- Run validation tests: `python tests/validation_tests.py`
- GPU issues: `export CUDA_VISIBLE_DEVICES=-1` (force CPU)

**Documentation:**
- Validation details: See `tests/` directory
- Data structure: See example CSV columns in Usage section
- Methodology: See full manuscript at Zenodo DOI

## License

**CC BY-NC-SA 4.0** — Creative Commons Attribution-NonCommercial-ShareAlike

- ✓ Share, adapt, use for research/education/non-profit
- ✗ Commercial use prohibited
- → Attribute authors, share adaptations under same license

See [LICENSE](LICENSE) or [Creative Commons](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Zenodo Archive

Permanently preserved via Zenodo:
- **DOI:** [10.5281/zenodo.17420367](https://doi.org/10.5281/zenodo.17420367)
- **Version:** v1.1.0
- **Archive:** Guaranteed indefinitely

---

**Version:** 1.1.0 (Validated & Corrected) | **Status:** Peer-reviewed publication | **License:** CC BY-NC-SA 4.0 | **Updated:** November 1, 2025
