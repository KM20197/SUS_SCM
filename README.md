# A Hybrid AI Architecture for Healthcare Supply Chain Optimization

This repository contains the official source code for the manuscript titled: "A Hybrid Artificial Intelligence Architecture for Healthcare Supply Chain Optimization: Benchmarking Machine Learning, Reinforcement Learning, and Genetic Algorithms Against Operations Research Baselines."

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![DOI: 10.5281/zenodo.17420367](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17420367-blue.svg)](https://doi.org/10.5281/zenodo.17420367)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Abstract

The study develops and evaluates a hybrid artificial intelligence (AI) architecture that integrates machine learning (ML), reinforcement learning (RL), and genetic algorithms (GA) for healthcare supply chain optimization. Using operational data from Brazil's Unified Health System (SUS), the research conducts 100 Monte Carlo replications across 90 daily decision points to benchmark the hybrid AI architecture against classical operations research (OR) baselines. Complete empirical dataset (9,170 transactions from 3 facilities, 365-day observation period) is included in the repository for full reproducibility. Empirical findings indicate that the AI architecture achieves superior performance under volatile or disrupted conditions, while classical OR policies remain more cost-efficient under stable demand. The study contributes a replicable methodological architecture, a validated benchmarking protocol, and empirically derived deployment criteria for high-uncertainty healthcare environments.

## Citation

If you use this code or the concepts presented in our work, please cite our paper:

```bibtex
@article{Mello2025HybridAI,
  title   = {A Hybrid Artificial Intelligence Architecture for Healthcare Supply Chain Optimization: Benchmarking Machine Learning, Reinforcement Learning, and Genetic Algorithms Against Operations Research Baselines},
  author  = {Mello, Ricardo Coutinho and Chen, Bo and Fernandes, Felipe Schuler and Claro, Daniela Barreiro and Ladeira, Rodrigo and Fernandes, Antônio Sérgio Araújo},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.17420367},
  url          = {https://doi.org/10.5281/zenodo.17420367}
}
```

## Quick Start

**Clone and run immediately with included dataset (no configuration needed):**

```bash
git clone https://github.com/your-org/hybrid-ai-supply-chain.git
cd hybrid-ai-supply-chain
pip install -r requirements.txt
python FinalCode.py
```

Expected runtime: 15-45 minutes | Results saved to `outputs/`

## Methodological Overview

The framework integrates three core AI components into a closed-loop simulation environment:

1. **Machine Learning (ML) Forecasting:** A hybrid ensemble of a Long Short-Term Memory (LSTM) network and a Random Forest (RF) regressor generates demand predictions based on 14-day historical sequences.

2. **Reinforcement Learning (RL) Inventory Policy:** A Double Q-learning agent learns an adaptive inventory policy with 100 states and 5 action levels, trained over 100 episodes with dynamic discretization.

3. **Genetic Algorithm (GA) Routing:** A computationally-budgeted DEAP-based GA (population=30, generations=12) solves the Vehicle Routing Problem (VRP) to provide realistic transportation costs.

The performance of the AI framework is benchmarked against four classical operations research policies:
- **STATIC:** Reorder-point policy with fixed safety stock
- **DYNAMIC:** Adaptive policy with exponential smoothing  
- **BASE_STOCK_PERFECT:** Optimal OR policy with perfect demand information
- **BASE_STOCK_UNCERTAIN:** Adaptive OR policy using ML-generated forecasts

## System Requirements

- **Python:** 3.8 or higher
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** 1 GB for dependencies + 50 MB for outputs
- **Optional:** GPU (CUDA/cuDNN) for 2-3x speedup on LSTM training

**Required Python packages:** See `requirements.txt`

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/hybrid-ai-supply-chain.git
cd hybrid-ai-supply-chain
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import tensorflow; import deap; print('✓ All dependencies installed')"
```

## Usage

### Run with Included SUS Dataset (Recommended)

No configuration needed—the code automatically uses the included example data:

```bash
python FinalCode.py
```

**What happens:**
- Loads `data/database.csv` (9,170 SUS transactions, 365 days, 2021)
- Executes 100 Monte Carlo replications
- Compares 5 inventory policies
- Generates statistical analysis with Mann-Whitney U tests
- Saves results to `outputs/` directory

### Run with Your Own Data

Option A: Export environment variable
```bash
export DATA_PATH=/path/to/your/data.csv
python FinalCode.py
```

Option B: Modify code (line 1-2 of FinalCode.py)
```python
data_path = '/path/to/your/data.csv'
```

### Expected Output Example

```
================================================================================
HYBRID AI-RL-GA FRAMEWORK FOR SUPPLY CHAIN OPTIMIZATION
================================================================================
[DataProcessor] Loading data from data/database.csv...
[DataProcessor] Raw records: 9170
[DataProcessor] Processed records: 9078
[SimulationSystem] Units to process: ['City01', 'City02', 'City03']

[STATISTICAL ANALYSIS - Mann-Whitney U with Bonferroni Correction]

DESCRIPTIVE STATISTICS (Normalized cost in R$):
static                   : Cost=105,876 (±78,074) FR=0.9599
dynamic                  : Cost=1,479 (±1,081) FR=0.9607
ai                       : Cost=35,944 (±39,750) FR=0.9970
base_stock_perfect       : Cost=34,195 (±26,568) FR=0.9972
base_stock_uncertain     : Cost=77,653 (±60,060) FR=0.9893

STATISTICAL TESTS (Mann-Whitney U + Glass's Delta + Bonferroni):
dynamic vs static        : U=20000 p<0.001 ✓ SIG Glass's Δ=-1.89
ai vs static            : U=20585 p<0.001 ✓ SIG Glass's Δ=-1.13
ai vs dynamic           : U=61582 p<0.001 ✓ SIG Glass's Δ=1.22

SIMULATION COMPLETED in 23.5 minutes
================================================================================
```

Results saved to:
- `outputs/checkpoint.csv` - Full simulation results (100 MC × 5 policies)
- `outputs/results_summary.csv` - Aggregated performance metrics
- `outputs/statistical_tests.csv` - Statistical test results with effect sizes

## Data Options

### Option 1: Use Included Example Data ✓ Recommended for First Run

The repository includes a complete SUS dataset for immediate testing:

**File:** `data/database.csv` (0.89 MB)

**Dataset Characteristics:**
- **Source:** Brazil's Unified Health System (SUS)
- **Records:** 9,170 transactions
- **Period:** January 1 - December 31, 2021 (365 consecutive days)
- **Healthcare Facilities:** 3 units (City01, City02, City03)
- **Pharmaceutical SKUs:** 50 distinct medications (MED_000004 to MED_000113)
- **Quantity Range:** 62-2,345,577 units per transaction
- **Cost Range:** R$ 0.08 to R$ 19,516.67 per unit
- **Data Quality:** No missing values, fully validated

**Advantages:**
- ✓ Reproduces exact manuscript results
- ✓ No data preparation required
- ✓ Immediate execution
- ✓ Validates installation

**Usage:**
```bash
python FinalCode.py  # Automatically loads data/database.csv
```

### Option 2: Use Your Own Data

Prepare a CSV file with these **required columns**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| DATA | date (YYYY-MM-DD) | Transaction date | 2021-01-15 |
| ALMOXARIFADO | string | Healthcare facility identifier | City01 |
| CODIGO | string | Pharmaceutical SKU code | MED_000026 |
| QUANTIDADE | numeric | Units received/requested | 919699.39 |
| UNITCOST | numeric | Price per unit (R$) | 443.97 |
| LATITUDE | numeric | Facility latitude | -12.0956 |
| LONGITUDE | numeric | Facility longitude | -45.7866 |

**Optional column:**
- VALOR: Total transaction value (computed automatically if missing)

**Data Requirements:**
- Minimum: 1,000 records (recommended 5,000+)
- Duration: At least 90 consecutive days
- Multiple facilities: 3+ distinct ALMOXARIFADO values
- Multiple SKUs: 10+ distinct CODIGO values
- Date range: Ideally 2020-2022 (model calibrated for this period)

**Example CSV Structure:**

```csv
DATA,CODIGO,ALMOXARIFADO,QUANTIDADE,UNITCOST,LATITUDE,LONGITUDE
2021-01-01,MED_000026,City01,919699.39,443.97,-12.0956,-45.7866
2021-01-01,MED_000038,City02,259330.14,3.30,-10.7171,-43.6302
2021-01-02,MED_000026,City01,850000.00,450.00,-12.0956,-45.7866
2021-01-02,MED_000094,City03,210000.00,5.30,-11.5000,-44.5000
```

**Usage:**

```bash
export DATA_PATH=path/to/your/data.csv
python FinalCode.py
```

## Example Dataset Dictionary

The included `data/database.csv` contains:

| Column | Type | Count | Min | Max | Mean |
|--------|------|-------|-----|-----|------|
| DATA | date | 365 unique | 2021-01-01 | 2021-12-31 | — |
| CODIGO | string | 50 SKUs | MED_000004 | MED_000113 | — |
| ALMOXARIFADO | string | 3 facilities | City01 | City03 | — |
| QUANTIDADE | numeric | 9,170 records | 62.21 | 2,345,577 | 42,256 |
| UNITCOST | numeric | 9,170 values | R$ 0.08 | R$ 19,516.67 | R$ 102.61 |
| LATITUDE | numeric | 3 coordinates | -12.42 | -10.72 | -11.89 |
| LONGITUDE | numeric | 3 coordinates | -45.79 | -41.77 | -45.28 |

**Data Aggregation Level:** SKU-facility-date (not patient-level) - ensures privacy

## Computational Parameters

### ML Ensemble (LSTMRFEnsemble)
- **Sequence length:** 14 days
- **LSTM units:** 64
- **Dropout rate:** 0.3
- **Random Forest estimators:** 120
- **RF max depth:** 12
- **Early stopping:** patience=3 (validation loss)

### RL Agent (QLearningAgent)
- **State space:** 100 (10×10 discretization grid)
- **Action space:** 5 (ordering levels)
- **Learning rate (α):** 0.1
- **Discount factor (γ):** 0.95
- **Exploration (ε):** Initial 0.2, decay to 0.02
- **Training epochs:** 100 total (30+40+30 episodes)
- **Algorithm:** Double Q-learning (reduces overestimation bias)

### GA/VRP (DEAP)
- **Population size:** 30 individuals
- **Generations:** 12
- **Timeout per optimization:** 1.5 seconds
- **Max routing points:** 7 locations
- **Crossover probability:** 0.8 (ordered crossover)
- **Mutation probability:** 0.2 (shuffle indexes)
- **Selection:** Tournament (size=3)

### Cost Function Parameters
```python
stockout_penalty = 25.0        # R$ per unit shortage
holding_rate = 0.003           # Daily inventory holding cost
ordering_rate = 1.5            # Fixed ordering cost
service_bonus = 8.0            # Service achievement reward
safety_stock_factor = 2.0      # Initial inventory buffer multiplier
```

### Simulation Configuration
- **Monte Carlo replications:** 100 per facility
- **Evaluation horizon:** 90 days
- **Train/test split date:** June 30, 2021
- **Bootstrap resamples:** 1,000 (confidence intervals)
- **Bonferroni α:** 0.0083 (adjusted for 6 comparisons)

## Output Files

Upon execution, `outputs/` directory contains:

1. **checkpoint.csv** (5,000+ rows)
   - Full simulation results for every MC replication
   - Columns: unit, sim, [policy]_cost, [policy]_fr
   - Use for detailed analysis or custom post-processing

2. **results_summary.csv** (5 rows)
   - Aggregated statistics per policy
   - Columns: policy, mean_cost, std_cost, mean_fr
   - Quick reference for performance comparison

3. **statistical_tests.csv** (6 rows)
   - Pairwise comparisons (Mann-Whitney U)
   - Columns: comparison, u_stat, p_value, bonf_sig, glass_delta
   - Evidence for statistical significance of differences

## Statistical Analysis

The framework includes rigorous hypothesis testing:

- **Primary Test:** Mann-Whitney U (non-parametric, robust to non-normality)
- **Effect Size:** Glass's Delta with interpretation guide
  - |Δ| < 0.2: negligible
  - 0.2 ≤ |Δ| < 0.5: small
  - 0.5 ≤ |Δ| < 0.8: medium
  - |Δ| ≥ 0.8: large
- **Multiple Comparisons:** Bonferroni correction (α_adjusted = 0.017 / 6 = 0.0083)
- **Confidence Intervals:** Bootstrap percentile method (1,000 resamples, 95% CI)
- **Replication:** 100 Monte Carlo iterations with perturbation analysis

## Reproduction Notes

**To obtain results identical to manuscript Table 1:**

1. Use included `data/database.csv` without modification
2. Keep all parameters at defaults (no code changes)
3. Set random seed before running (already done: `seed=42`)
4. Expected cost ratio: DYNAMIC ≤ BASE_STOCK_PERFECT ≤ AI ≤ BASE_STOCK_UNCERTAIN ≤ STATIC

**If results differ:**
- Check if different CSV file was used (compare file hash)
- Verify Python version (3.8+) and dependency versions
- Run twice—results should be identical (deterministic)
- Check for modified parameters in code

## Data Acknowledgment

The example dataset included in this repository originates from:

**Data Providers:**
- State of Bahia Health Secretariat (SESAB)
- Federal University Hospital (HUPES)
- Brazil's Unified Health System (SUS)

**Data Processing:**
- Aggregated to facility-SKU-date level
- Anonymized facility names (City01/02/03)
- No patient-level identifiable information

**Citation:** If using this dataset for research, acknowledge both this repository and the original data custodians. For access to additional datasets or research partnerships, contact data providers directly.

## Repository Structure

```
hybrid-ai-supply-chain/
├── FinalCode.py                        # Main simulation script
├── requirements.txt                    # Python dependencies (8 packages)
├── README.md                           # This file
├── LICENSE                             # CC BY-NC-SA 4.0 license
├── .gitignore                          # Git exclusion patterns
├── data/
│   └── database.csv                    # Example SUS dataset (9,170 records, 0.89 MB)
├── outputs/
│   ├── .gitkeep
│   ├── checkpoint.csv                  # Full results (after execution)
│   ├── results_summary.csv             # Aggregate metrics (after execution)
│   └── statistical_tests.csv           # Statistical tests (after execution)
└── docs/
    └── INSTALL.md                      # Detailed installation guide
```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License** (CC BY-NC-SA 4.0).

**You are free to:**
- Share and redistribute the material in any format
- Adapt, remix, and build upon the material

**Under these terms:**
- **Attribution:** Give appropriate credit to the original authors
- **NonCommercial:** Do not use for commercial purposes
- **ShareAlike:** Distribute adaptations under the same license

**Full license:** See [LICENSE](LICENSE) file or visit [Creative Commons](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Research Context

This framework operationalizes peer-reviewed research on AI-driven healthcare logistics optimization:

- **Study Design:** Simulation-based benchmarking with 100 Monte Carlo replications
- **Data Integration:** Real SUS operational records (9,170+ transactions)
- **Evaluation Period:** 90-day decision horizons with train-test temporal splits
- **Baseline Policies:** Five competing inventory policies representing theory and practice
- **Performance Metrics:** Cost efficiency (primary) and service-level fill rates (secondary)
- **Publication:** Zenodo DOI: [10.5281/zenodo.17420367](https://doi.org/10.5281/zenodo.17420367)

## Support & Troubleshooting

### Common Issues

**Issue: "No module named 'tensorflow'"**
```bash
pip install --force-reinstall tensorflow>=2.8.0
```

**Issue: "FileNotFoundError: data/database.csv"**
```bash
ls -la data/database.csv  # Verify file exists
export DATA_PATH=$(pwd)/data/database.csv  # Full path
python FinalCode.py
```

**Issue: "CUDA out of memory" (GPU systems)**
```bash
export CUDA_VISIBLE_DEVICES=-1  # Force CPU
python FinalCode.py
```

**Issue: Slow first execution**
- Normal behavior: LSTM graph compilation takes time
- Subsequent runs faster
- Use GPU if available for 2-3x speedup

### Documentation

- [Installation Guide](docs/INSTALL.md) - Detailed setup for all platforms
- [GitHub Issues](https://github.com/your-org/hybrid-ai-supply-chain/issues) - Report bugs
- [Zenodo](https://doi.org/10.5281/zenodo.17420367) - Cite this work

## Authors

Ricardo Coutinho Mello, Bo Chen, Felipe Schuler Fernandes, Daniela Barreiro Claro, Antônio Sérgio Araújo Fernandes, Rodrigo Ladeira

## Zenodo Archival

This code and manuscript are preserved via Zenodo:
- **DOI:** [10.5281/zenodo.17420367](https://doi.org/10.5281/zenodo.17420367)
- **Version:** v1.0.0
- **Archive Date:** October 2025
- **Citation:** Use the BibTeX format above for academic citations

---

**Version:** 1.0.0  
**Last Updated:** October 23, 2025  
**Status:** Peer-reviewed publication  
**License:** CC BY-NC-SA 4.0
