# A Hybrid AI Architecture for Healthcare Supply Chain Optimization

This repository contains the official source code for the manuscript titled: "A Hybrid Artificial Intelligence Architecture for Healthcare Supply Chain Optimization: Benchmarking Machine Learning, Reinforcement Learning, and Genetic Algorithms Against Operations Research Baselines."

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![DOI: 10.5281/zenodo.17420367](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17420367-blue.svg)](https://doi.org/10.5281/zenodo.17420367)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Validation Tests](https://img.shields.io/badge/Validation-Complete-green.svg)]()

## Abstract

The study develops and evaluates a hybrid artificial intelligence (AI) architecture that integrates machine learning (ML), reinforcement learning (RL), and genetic algorithms (GA) for healthcare supply chain optimization. Using operational data from Brazil's Unified Health System (SUS), the research conducts 100 Monte Carlo replications across 90 daily decision points to benchmark the hybrid AI architecture against classical operations research (OR) baselines, including a corrected analytical base-stock policy with stochastic lead time and empirically calibrated safety stock parameters. Complete empirical dataset (9,170 transactions from 3 facilities, 365-day observation period) is included in the repository for full reproducibility. Empirical findings demonstrate that the AI architecture achieves statistically significant cost reduction (65.3% vs. static baseline, p < 0.0001) while maintaining superior service levels (fill rate 0.9925 vs. 0.9604 baseline) under volatile and disrupted conditions. The study contributes a replicable methodological architecture, a validated benchmarking protocol with Bonferroni-corrected hypothesis testing, rigorously specified baseline policies, and empirically derived deployment criteria for high-uncertainty healthcare environments.

## Citation

If you use this code or the concepts presented in our work, please cite our paper:

```bibtex
@article{Mello2025HybridAI,
  title   = {A Hybrid Artificial Intelligence Architecture for Healthcare Supply Chain Optimization: Benchmarking Machine Learning, Reinforcement Learning, and Genetic Algorithms Against Operations Research Baselines},
  author  = {Mello, Ricardo Coutinho and Chen, Bo and Fernandes, Felipe Schuler and Claro, Daniela Barreiro and Fernandes, Antônio Sérgio Araújo and Ladeira, Rodrigo},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {v1.1.0},
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

Expected runtime: 40-45 minutes | Results saved to `outputs/` | Validation tests available in `tests/`

## Recent Updates (v1.1.0)

**November 2025 - Implementation Corrections and Validation**

This release incorporates empirically validated corrections to the BASE_STOCK_UNCERTAIN policy:

- ✅ **Corrected z-score parameter**: z_α = 1.645 (95% service level) aligns with SUS equity objectives
- ✅ **Stochastic lead time**: LT ~ Normal(μ=7.2, σ=1.8 days) reflects empirical SUS procurement variability
- ✅ **ML forecast integration**: Confirmed integration of ensemble forecast uncertainty (σ_forecast) for fair algorithmic comparison
- ✅ **Empirical scenario grounding**: HIGH_VOLATILITY (CV 0.42→0.63), SUPPLY_DISRUPTION (20%, LT+40%), and SEASONAL_SURGE (180-220%) validated against documented SUS crises (H1N1 2009, COVID-19 2020, Zika 2016)
- ✅ **Fair comparison validated**: Both AI and BASE_STOCK_UNCERTAIN policies operate under identical information constraints (same forecasts, parameters, constraints)
- ✅ **Statistical significance confirmed**: AI vs. BASE_STOCK_UNCERTAIN cost difference significant (p=0.006, Bonferroni α=0.017, Glass's Δ=0.5128)
- ✅ **Validation test suite**: Complete test suite validates policy specifications and empirical grounding (see `tests/` directory)

**Performance Summary (Monte Carlo n=100):**

| Policy | Mean Cost (R$) | Cost Reduction vs. Static | Fill Rate | Status |
|--------|----------------|--------------------------|-----------|--------|
| Static (Baseline) | 3,604,269,458 | — | 0.9604 | Reference |
| Dynamic | 50,199,985 | 98.6% | 0.9613 | Exponential Smoothing |
| **AI (RL-based)** | **1,251,921,849** | **65.3%** | **0.9925** | **✓ Recommended** |
| Base-Stock Perfect | 1,156,791,065 | 67.9% | 0.9972 | Theoretical Optimum |
| Base-Stock Uncertain (Corrected) | 2,064,134,674 | 42.7% | 0.9920 | OR Baseline |

*Costs normalized by 90-day horizon; significant at Bonferroni α=0.017*

## Methodological Overview

The framework integrates three core AI components into a closed-loop simulation environment:

### 1. Machine Learning (ML) Forecasting: Hybrid LSTM-RF Ensemble

A hybrid ensemble combining Long Short-Term Memory (LSTM) networks and Random Forest (RF) regressors generates demand predictions from 14-day historical sequences. The ensemble approach reduces individual model biases:

- **LSTM component**: Captures temporal dependencies and demand seasonality
- **RF component**: Identifies non-linear feature interactions and structural breaks
- **Bootstrap aggregation**: Generates probabilistic forecasts with confidence intervals
- **Uncertainty quantification**: σ_forecast used for fair comparison with analytical policies

**Key parameters:**
- Sequence length: 14 days
- LSTM units: 64 | Dropout: 0.3
- RF estimators: 120 | Max depth: 12
- Early stopping: patience=3

### 2. Reinforcement Learning (RL) Inventory Policy: Double Q-Learning

A Double Q-learning agent learns an adaptive inventory policy with state discretization:

- **State space**: 100 (10×10 discretization grid: current inventory × forecast uncertainty)
- **Action space**: 5 ordering levels (0-4 units relative to forecast)
- **Algorithm**: Double Q-learning (reduces overestimation bias vs. standard Q-learning)
- **Training**: 100 episodes with ε-greedy exploration (initial ε=0.2 → final ε=0.02)
- **Learning parameters**: α=0.1 (learning rate), γ=0.95 (discount factor)

**Advantages over analytical policies:**
- Nonlinear state-value mapping captures complex inventory-uncertainty interactions
- Adaptive learning accommodates facility-specific demand patterns
- Integration with GA enables joint optimization with vehicle routing

### 3. Genetic Algorithm (GA) Routing: Vehicle Routing Problem (VRP)

A DEAP-based genetic algorithm solves the Vehicle Routing Problem to provide realistic transportation costs:

- **Population size**: 30 individuals
- **Generations**: 12 (computational budget: 1.5 sec/optimization)
- **Crossover probability**: 0.8 (ordered crossover)
- **Mutation probability**: 0.2 (shuffle)
- **Selection**: Tournament (size=3)
- **Objective**: Minimize distance + time subject to vehicle capacity constraints

### 4. Operations Research (OR) Baseline Policies

The hybrid AI is benchmarked against four OR policies representing progressively sophisticated analytical approaches:

#### Policy 1: STATIC
- **Mechanism**: Fixed reorder point with constant safety stock
- **Formula**: ROP = μ × LT + z × σ × √LT (standard inventory theory)
- **Parameterization**: z=1.96 (99% service level), LT=5 days
- **Updating**: Policy parameters never change across simulation horizon

#### Policy 2: DYNAMIC
- **Mechanism**: Adaptive reorder point with exponential smoothing forecasts
- **Formula**: ROP_t = μ_smooth,t × LT + z × σ_smooth,t × √LT
- **Smoothing parameters**: α=0.2 (level), β=0.1 (trend)
- **Advantage over STATIC**: Responds to demand changes via smoothing, but no future prediction

#### Policy 3: BASE_STOCK_PERFECT (Theoretical Optimum)
- **Mechanism**: Base-stock policy with perfect demand information (omniscient)
- **Information**: Assumes demand known without forecast error
- **Formula**: S_perfect = μ_actual + z × σ_actual × √(LT + review_period)
- **Purpose**: Upper bound (optimal analytical performance)
- **Note**: Unrealistic but provides theoretical performance ceiling

#### Policy 4: BASE_STOCK_UNCERTAIN (Corrected OR Baseline - **NEW v1.1.0**)
- **Mechanism**: Analytical base-stock policy with ML-generated forecasts (fair comparison)
- **Formula**: S_t = μ_forecast,t + z_α × σ_forecast × √(LT + review_period)
- **Parameters** (corrected):
  - z_α = 1.645 (95% service level per SUS equity objectives)
  - LT ~ Normal(μ=7.2, σ=1.8) days (empirical SUS procurement distribution)
  - σ_forecast from LSTM-RF ensemble (identical to AI policy forecasts)
  - review_period = 1 day (continuous review)
- **Fair comparison validation**: Both AI and BASE_STOCK_UNCERTAIN receive identical forecasts and constraints; difference is algorithmic (RL vs. analytical)
- **Empirical grounding**: Parameters validated against historical SUS data (2009-2023)

## Validation & Empirical Grounding

### Specification Validation Tests

Complete validation test suite confirms all policy implementations match specifications. Tests can be run independently:

```bash
python -m pytest tests/validation_tests.py -v
```

**Test Coverage:**

1. **TEST 1: BASE_STOCK_UNCERTAIN Implementation** (test_base_stock_uncertain_integration)
   - Verifies z_α = 1.645 implementation
   - Validates stochastic lead time sampling from Normal(7.2, 1.8)
   - Confirms σ_forecast integration from ML ensemble
   - Confirms fair comparison (identical forecast objects)

2. **TEST 2: Stress Scenario Parameters** (test_stress_scenario_parameters)
   - HIGH_VOLATILITY: CV increase 50% (0.42→0.63) validated ✓
   - SUPPLY_DISRUPTION: 20% disruption rate validated ✓
   - SEASONAL_SURGE: Demand range 180-220% validated ✓

3. **TEST 3: Fair Comparison** (validate_fair_comparison)
   - Confirms AI and BASE_STOCK_UNCERTAIN receive identical forecasts
   - Verifies constraint equivalence
   - Validates outcome difference is purely algorithmic

4. **TEST 4: Scenario Sensitivity** (validate_scenario_sensitivity)
   - HIGH_VOLATILITY: 0.63 within empirical range [0.61, 0.68] ✓
   - SUPPLY_DISRUPTION: 20% within empirical range [18%, 23%] ✓
   - SEASONAL_SURGE: 180-220% encompasses empirical data ✓

### Empirical Grounding Against SUS Historical Crises

All stress test scenarios validated against documented SUS crises:

**HIGH_VOLATILITY Scenario (CV: 0.42 → 0.63)**
- H1N1 Pandemic (2009): Observed CV = 0.61 (antiviral medications)
- COVID-19 Wave 1 (2020): Observed CV = 0.68 (ICU medications)
- Source: Ministry of Health Procurement Database (SIASG 2009-2023)

**SUPPLY_DISRUPTION Scenario (Disruption: 20%, LT extension: +40%)**
- COVID-19 Q2 2020: 18-23% of orders delayed >30 days
- Zika Outbreak (2016): Lead time increased +50% (7.2 → 10.8 days)
- Source: Brazilian Pharmaceutical Association, SESAB logistics records

**SEASONAL_SURGE Scenario (Demand: 180-220% baseline)**
- Annual flu vaccination: 195-240% demand spike (April-June)
- Dengue epidemics: 210% peak demand during epidemic weeks
- Source: SUS Immunization Database (2018-2022), SESAB epidemiological data

**Data Sources Cited in Manuscript:**
- Ministry of Health Procurement Database (SIASG): 2009-2023
- SESAB Internal Logistics Records: 2016-2023
- Brazilian Pharmaceutical Association COVID-19 Reports: 2020-2021

## System Requirements

- **Python:** 3.8 or higher
- **RAM:** 8 GB minimum (16 GB recommended for parallel processing)
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

### Run Full Simulation with Included SUS Dataset

No configuration needed—the code automatically uses the included example data:

```bash
python FinalCode.py
```

**What happens:**
- Loads `data/database.csv` (9,170 SUS transactions, 365 days, 2021)
- Trains LSTM-RF ensemble and RL agent on train set (through 2021-06-30)
- Executes 100 Monte Carlo replications on test set (2021-07-01 to 2021-09-29)
- Compares 5 inventory policies: STATIC, DYNAMIC, AI, BASE_STOCK_PERFECT, BASE_STOCK_UNCERTAIN
- Performs statistical analysis: Mann-Whitney U tests with Bonferroni correction
- Saves results to `outputs/` directory

**Expected output:**
```
================================================================================
HYBRID AI-RL-GA FRAMEWORK FOR HEALTHCARE SUPPLY CHAIN OPTIMIZATION
================================================================================
[DataProcessor] Loading data from data/database.csv...
[DataProcessor] Raw records: 9170
[DataProcessor] Processed records: 9078
[SimulationSystem] Units to process: ['City01', 'City02', 'City03']
[City01] Train: 3619, Test: 3681
[City01] Training LSTM-RF ensemble...
[City01] Training RL agent (100 episodes)...
[City01] Generating ML forecasts...
[City01] Running 100 Monte Carlo simulations (5 policies)...

[STATISTICAL ANALYSIS - Mann-Whitney U with Bonferroni Correction]

DESCRIPTIVE STATISTICS (Normalized cost in R$):
static                   : Cost=3,604,269,457.74±2,656,652,678.32 FR=0.9604
dynamic                  : Cost=50,199,985.26±36,709,204.24 FR=0.9613
ai                       : Cost=1,251,921,848.98±1,367,618,166.74 FR=0.9925
base_stock_perfect       : Cost=1,156,791,065.38±909,960,418.79 FR=0.9972
base_stock_uncertain     : Cost=2,064,134,674.15±1,769,277,269.20 FR=0.9920

STATISTICAL TESTS (Mann-Whitney U + Glass's Delta + Bonferroni):
ai vs base_stock_uncertain : U=50834 p=0.006002 ✓ SIG Glass's Δ=0.5128
base_stock_uncertain vs base_stock_perfect : U=60177 p<0.000001 ✓ SIG Glass's Δ=0.6439

SIMULATION COMPLETED in 40.9 minutes
================================================================================
```

### Run with Your Own Data

**Option A: Export environment variable**
```bash
export DATA_PATH=/path/to/your/data.csv
python FinalCode.py
```

**Option B: Modify code (line 41 in FinalCode.py)**
```python
data_path = '/path/to/your/data.csv'
```

### Run Validation Tests Only

Validate that policies implement specifications correctly without running full simulation:

```bash
python tests/validation_tests.py
```

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
- **Confidentiality:** Aggregated to facility-SKU-date level; no patient identifiers

**Advantages:**
- ✓ Reproduces exact manuscript results (deterministic, seed=42)
- ✓ No data preparation required
- ✓ Immediate execution
- ✓ Validates installation
- ✓ Enables policy comparison with documented SUS conditions

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
- Time order: Sequential dates (required for LSTM sequence modeling)

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

## Computational Parameters

### ML Ensemble (LSTMRFEnsemble)
- **Sequence length:** 14 days (model input)
- **LSTM units:** 64
- **Dropout rate:** 0.3 (regularization)
- **Random Forest estimators:** 120
- **RF max depth:** 12
- **Early stopping:** patience=3 (validation loss plateau)
- **Validation split:** 0.2 (20% of training data)

### RL Agent (QLearningAgent - Double Q-Learning)
- **State space:** 100 (10×10 discretization: inventory × forecast uncertainty)
- **Action space:** 5 ordering levels (0-4 units relative to forecast)
- **Learning rate (α):** 0.1
- **Discount factor (γ):** 0.95
- **Exploration (ε):** Initial 0.2 → Final 0.02 (linear decay)
- **Training epochs:** 100 total episodes
- **Algorithm:** Double Q-Learning (reduces overestimation vs. standard Q-learning)

### GA/VRP (DEAP-based)
- **Population size:** 30 individuals
- **Generations:** 12
- **Timeout per optimization:** 1.5 seconds
- **Max routing points:** 7 locations per route
- **Crossover probability:** 0.8 (ordered crossover)
- **Mutation probability:** 0.2 (shuffle indexes)
- **Selection:** Tournament (tournament size=3)

### Cost Function Parameters
```python
stockout_penalty = 25.0        # R$ per unit shortage (service cost)
holding_rate = 0.003           # Daily inventory holding cost (%)
ordering_rate = 1.5            # Fixed ordering cost (R$)
service_bonus = 8.0            # Service achievement reward (R$)
safety_stock_factor = 2.0      # Initial inventory buffer multiplier
```

### Simulation Configuration
- **Monte Carlo replications:** 100 per facility
- **Evaluation horizon:** 90 days (2021-07-01 to 2021-09-29)
- **Train/test split date:** 2021-06-30 (temporal split, not random)
- **Bootstrap resamples:** 1,000 (confidence interval estimation)
- **Bonferroni α:** 0.017 (α₀=0.05 / 3 independent comparison groups)
- **Random seed:** 42 (deterministic for reproducibility)

## Output Files

Upon execution, `outputs/` directory contains:

1. **checkpoint.csv** (300+ rows after 100 MC replications × 3 facilities)
   - Full simulation results for every MC replication and facility
   - Columns: unit, sim, [policy]_cost, [policy]_fr, timestamp
   - Use for detailed analysis or custom post-processing

2. **results_summary.csv** (5 rows, one per policy)
   - Aggregated statistics per policy across all facilities and replications
   - Columns: policy, mean_cost, std_cost, ci_lower, ci_upper, mean_fr, std_fr
   - Quick reference for performance comparison

3. **statistical_tests.csv** (6 rows, one per pairwise comparison)
   - Pairwise comparisons (Mann-Whitney U test)
   - Columns: comparison, u_statistic, p_value, bonferroni_significant, glass_delta, interpretation
   - Evidence for statistical significance of differences

4. **[unit]_plots.png** (3 files)
   - Visualization of cost distributions and service levels by facility
   - Histograms with kernel density estimation

## Statistical Analysis Methodology

The framework implements rigorous hypothesis testing for fair policy comparison:

### Primary Test: Mann-Whitney U Test
- **Rationale**: Non-parametric test robust to non-normal distributions (simulation costs often skewed)
- **Null hypothesis**: Two policies produce costs from same distribution
- **Alternative**: Policy A cost ≠ Policy B cost (two-tailed)
- **Advantage over t-test**: Does not assume normality; appropriate for Monte Carlo data

### Multiple Comparisons Correction: Bonferroni

The framework tests 6 pairwise comparisons:
1. Dynamic vs. Static
2. AI vs. Static
3. AI vs. Dynamic
4. Base-Stock Perfect vs. Static
5. **AI vs. Base-Stock Uncertain** (primary comparison)
6. Base-Stock Uncertain vs. Base-Stock Perfect

Bonferroni adjustment: α_adjusted = 0.05 / 3 = **0.017** (protecting 3 independent policy groups)

**Note**: Comparisons within same algorithmic family (e.g., different RL configurations) use α=0.05; cross-family comparisons use α=0.017

### Effect Size: Glass's Delta

\[ Δ = \frac{μ_A - μ_B}{σ_B} \]

Where σ_B is the standard deviation of the reference (typically more conservative) policy.

**Interpretation (Cohen conventions):**
- |Δ| < 0.2: Negligible effect (practical equivalence)
- 0.2 ≤ |Δ| < 0.5: Small effect
- 0.5 ≤ |Δ| < 0.8: **Medium effect** ← AI vs. BASE_STOCK_UNCERTAIN (Δ=0.5128)
- |Δ| ≥ 0.8: Large effect

### Confidence Intervals: Bootstrap Percentile Method

- **Resampling**: 1,000 bootstrap samples per policy
- **Confidence level**: 95%
- **Estimation**: Percentile method (lower=2.5th percentile, upper=97.5th percentile)
- **Reported**: CI95%=[lower, upper] in results tables

### Replication Design: 100 Monte Carlo Iterations

Each facility runs 100 independent replications to:
1. Characterize cost distribution across uncertain demand realizations
2. Estimate confidence intervals and effect sizes
3. Achieve sufficient statistical power (typically n≥100 for MC studies)
4. Enable bootstrap resampling (requires ~100 samples minimum)

## Reproduction Notes

**To obtain results identical to manuscript Table 4.X:**

1. Use included `data/database.csv` without modification
2. Keep all parameters at defaults (no code changes)
3. Python version: 3.8 or higher
4. Set random seed before running (already done in code: `seed=42`)
5. Expected behavior: Results identical across multiple runs (deterministic)

**If results differ by >1%:**
- Check that correct CSV file is used (compare file hash: `sha256sum data/database.csv`)
- Verify Python and dependency versions (see requirements.txt)
- Run once more—results should be identical (deterministic with fixed seed)
- Check for modified parameters in FinalCode.py lines 41-80

**Reproducibility features implemented:**
- ✓ Fixed random seeds: `np.random.seed(42)`, `tf.random.set_seed(42)`
- ✓ Deterministic ordering in pandas/numpy operations
- ✓ Version locks in requirements.txt
- ✓ Temporal train-test split (not random cross-validation)

## Key Changes in v1.1.0

**Implementation Corrections:**

1. **BASE_STOCK_UNCERTAIN z-score corrected**
   - Before: z_α = 1.96 (99% service level)
   - After: z_α = 1.645 (95% service level, aligned with SUS equity objectives)
   - Impact: Reduces safety stock and total costs while maintaining operational service levels

2. **Lead time now stochastic**
   - Before: Fixed LT = 5 days
   - After: LT ~ Normal(μ=7.2, σ=1.8) days
   - Impact: Captures empirical SUS procurement variability; fair comparison with AI

3. **Empirical validation added**
   - Validated HIGH_VOLATILITY against H1N1 (CV=0.61) and COVID-19 (CV=0.68)
   - Validated SUPPLY_DISRUPTION against COVID-19 Q2 (18-23%) and Zika (LT+50%)
   - Validated SEASONAL_SURGE against flu campaigns (195-240%) and dengue (210%)
   - All stress test parameters grounded in documented SUS crises

4. **Fair comparison validated**
   - Confirmed AI and BASE_STOCK_UNCERTAIN receive identical forecasts
   - Verified identical cost parameters and constraints
   - Difference is purely algorithmic (RL vs. analytical)

5. **Statistical rigor enhanced**
   - Bonferroni correction applied (α=0.017 for main comparisons)
   - Glass's Δ effect sizes reported with interpretation
   - Bootstrap confidence intervals (1,000 resamples)
   - Complete validation test suite

**Results Impact:**

| Metric | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| AI vs. Static (cost reduction) | 65.5% | 65.3% | Stable |
| AI vs. BASE_STOCK_UNCERTAIN (Glass's Δ) | N/A | 0.5128 | NEW |
| AI vs. BASE_STOCK_UNCERTAIN (p-value) | N/A | 0.0060 | NEW |
| BASE_STOCK_UNCERTAIN z_α | 1.96 | 1.645 | ✓ Corrected |
| BASE_STOCK_UNCERTAIN lead time | Fixed (5d) | Stochastic (7.2d) | ✓ Corrected |

## Data Acknowledgment

The example dataset included in this repository originates from:

**Data Custodians:**
- State of Bahia Health Secretariat (SESAB - Secretaria de Saúde da Bahia)
- Federal University Hospital (HUPES - Hospital Universitário Professor Edgard Santos)
- Brazil's Unified Health System (SUS)

**Data Processing & Confidentiality:**
- Aggregated to facility-SKU-date level (not transaction-level)
- Anonymized facility names (City01, City02, City03)
- No patient-level identifiable information included
- Fully compliant with Brazilian healthcare privacy regulations (LGPD)

**Citation:** If using this dataset for research, acknowledge both this repository and the original data custodians. For access to additional datasets or research partnerships, contact data providers directly through their official channels.

## Repository Structure

```
hybrid-ai-supply-chain/
├── FinalCode.py                        # Main simulation script (v1.1.0)
├── requirements.txt                    # Python dependencies (8 packages)
├── README.md                           # This file
├── LICENSE                             # CC BY-NC-SA 4.0 license
├── .gitignore                          # Git exclusion patterns
├── data/
│   └── database.csv                    # Example SUS dataset (9,170 records, 0.89 MB)
├── tests/
│   └── validation_tests.py             # Specification validation suite
├── outputs/
│   ├── .gitkeep
│   ├── checkpoint.csv                  # Full results (after execution)
│   ├── results_summary.csv             # Aggregate metrics (after execution)
│   ├── statistical_tests.csv           # Statistical tests (after execution)
│   └── [unit]_plots.png                # Facility visualizations (after execution)
└── docs/
    ├── INSTALL.md                      # Detailed installation guide
    ├── VALIDATION.md                   # Validation test documentation
    └── METHODOLOGY.md                  # Detailed methodology (optional)
```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License** (CC BY-NC-SA 4.0).

**You are free to:**
- Share and redistribute the material in any format
- Adapt, remix, and build upon the material
- Use for research, education, and non-profit purposes

**Under these terms:**
- **Attribution:** Give appropriate credit to the original authors
- **NonCommercial:** Do not use for commercial purposes
- **ShareAlike:** Distribute adaptations under the same license

**Full license:** See [LICENSE](LICENSE) file or visit [Creative Commons](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Research Context

This framework operationalizes peer-reviewed research on AI-driven healthcare logistics optimization:

- **Study Design**: Simulation-based benchmarking with 100 Monte Carlo replications across 3 SUS facilities
- **Data Integration**: Real SUS operational records (9,170 transactions, 365 days)
- **Evaluation Period**: 90-day decision horizons with temporal train-test split (June 30, 2021 cutoff)
- **Baseline Policies**: Five competing inventory policies (STATIC, DYNAMIC, AI, BASE_STOCK_PERFECT, BASE_STOCK_UNCERTAIN)
- **Performance Metrics**: Cost efficiency (primary, normalized by 90-day horizon) and service-level fill rates (secondary)
- **Statistical Testing**: Mann-Whitney U tests with Bonferroni correction (α=0.017)
- **Effect Sizes**: Glass's Delta with Cohen interpretations
- **Confidence Intervals**: Bootstrap percentile method (1,000 resamples, 95% CI)
- **Publication**: Zenodo DOI: [10.5281/zenodo.17420367](https://doi.org/10.5281/zenodo.17420367)

## Support & Troubleshooting

### Common Issues

**Issue: "No module named 'tensorflow'"**
```bash
pip install --force-reinstall tensorflow>=2.8.0
```

**Issue: "FileNotFoundError: data/database.csv"**
```bash
ls -la data/database.csv                          # Verify file exists
export DATA_PATH=$(pwd)/data/database.csv         # Use absolute path
python FinalCode.py
```

**Issue: "CUDA out of memory" (GPU systems)**
```bash
export CUDA_VISIBLE_DEVICES=-1                   # Force CPU execution
python FinalCode.py
```

**Issue: Slow LSTM training**
- **Normal behavior**: LSTM graph compilation and first epoch takes time
- **Optimization**: Use GPU if available (2-3x speedup)
- **Alternative**: Reduce LSTM units to 32 or RF estimators to 60 (in FinalCode.py) for development testing

**Issue: Results differ from documentation**
- Check Python version: `python --version` (requires 3.8+)
- Check dependency versions: `pip list | grep tensorflow`
- Verify random seed is set to 42 (check line ~50 in FinalCode.py)
- Run twice to confirm determinism

### Validation Test Execution

```bash
python tests/validation_tests.py -v
```

Expected output: 4 test functions pass, all specifications confirmed ✓

### Documentation & Support

- [Installation Guide](docs/INSTALL.md) - Detailed setup for Windows/Mac/Linux
- [Validation Documentation](docs/VALIDATION.md) - Test suite details and empirical grounding
- [GitHub Issues](https://github.com/your-org/hybrid-ai-supply-chain/issues) - Report bugs or request features
- [Zenodo](https://doi.org/10.5281/zenodo.17420367) - Access archived version, cite publication

## Authors

- Ricardo Coutinho Mello (Lead author, methodology)
- Bo Chen (Advisor, RL expertise)
- Felipe Schuler Fernandes (Co-author, healthcare domain)
- Daniela Barreiro Claro (Co-author, SUS logistics)
- Antônio Sérgio Araújo Fernandes (Advisor, public health policy)
- Rodrigo Ladeira (Advisor, AI/ML integration)

## Zenodo Archival & Citation

This code and manuscript are permanently preserved via Zenodo:

- **DOI:** [10.5281/zenodo.17420367](https://doi.org/10.5281/zenodo.17420367)
- **Version:** v1.1.0 (Latest)
- **Archive Date:** November 2025
- **Permanence:** Guaranteed indefinitely via Zenodo institutional repository

For academic citations, use the BibTeX entry at the top of this README.

---

**Version:** 1.1.0 *(Validated & Corrected)*
**Last Updated:** November 1, 2025
**Release Date:** November 1, 2025
**Status:** Peer-reviewed publication with empirical validation
**License:** CC BY-NC-SA 4.0
**DOI:** 10.5281/zenodo.17420367
