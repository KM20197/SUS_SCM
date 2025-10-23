# A Hybrid AI Architecture for Healthcare Supply Chain Optimization

This repository contains the official source code for the manuscript titled: "A Hybrid Artificial Intelligence Architecture for Healthcare Supply Chain Optimization: Benchmarking Machine Learning, Reinforcement Learning, and Genetic Algorithms Against Operations Research Baselines."

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Abstract

This study develops and evaluates a hybrid artificial intelligence (AI) architecture that integrates machine learning (ML), reinforcement learning (RL), and genetic algorithms (GA) for healthcare supply chain optimization. Using operational data from Brazil's Unified Health System (SUS), the research conducts 100 Monte Carlo replications across 90 daily decision points to benchmark the hybrid AI architecture against classical operations research (OR) baselines. Empirical findings indicate that the AI architecture achieves superior performance under volatile or disrupted conditions, while classical OR policies remain more cost-efficient under stable demand. The study contributes a replicable methodological architecture, a validated benchmarking protocol, and empirically derived deployment criteria for high-uncertainty healthcare environments.

## Citation

If you use this code or the concepts presented in our work, please cite our paper:

```bibtex
@article{Mello2025HybridAI,
  title   = {A Hybrid Artificial Intelligence Architecture for Healthcare Supply Chain Optimization: Benchmarking Machine Learning, Reinforcement Learning, and Genetic Algorithms Against Operations Research Baselines},
  author  = {Mello, Ricardo Coutinho and Chen, Bo and Fernandes, Felipe Schuler and Claro, Daniela Barreiro and Ladeira, Rodrigo and Fernandes, Antônio Sérgio Araújo},
  journal = {Journal of Operations Research},
  year    = {2025},
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.17420367},
  url          = {https://doi.org/10.5281/zenodo.17420367}
}```
```

## Methodological Overview

The framework integrates three core components into a closed-loop simulation environment:
1.  **Machine Learning (ML) Forecasting:** A hybrid ensemble of a Long Short-Term Memory (LSTM) network and a Random Forest (RF) regressor generates demand predictions.
2.  **Reinforcement Learning (RL) Inventory Policy:** A Double Q-learning agent learns an adaptive inventory policy based on the ML forecasts.
3.  **Genetic Algorithm (GA) Routing:** A computationally-budgeted GA solves the Vehicle Routing Problem (VRP) to provide realistic transportation costs.

The performance of the AI framework is benchmarked against three other policies: a `STATIC` baseline, a `DYNAMIC` heuristic, and a classic OR `BASE_STOCK` policy.

## System Requirements

*   Python 3.9+
*   Key Python libraries: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `deap`, `statsmodels`, `scikit-optimize`.

## Installation

1.  Clone the repository:
    ```bash
    git clone [URL of your repository]
    cd [repository-name]
    ```
2.  It is highly recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install the required dependencies:
    ```bash
    pip install -q deap statsmodels scikit-optimize tensorflow pandas numpy
    ```

## Usage

The script is designed to be run directly from the command line.

```bash
python "FinalCode.py"
```

The script will execute the complete simulation and analysis pipeline. This is a computationally intensive process and may take several hours to complete, depending on the hardware.

## Data Requirements

The script operates in two modes:

1.  **Reproducibility Mode (Default):** By default, the script uses an internal synthetic data generator (`_generate_mock()`). This requires no external files and will exactly reproduce the results presented in the manuscript's reduced-scale experiment.

2.  **Empirical Mode (for use with real data):** To run the framework on a new, real-world dataset, you must:
    *   Modify the `DataProcessor` class in the script to load your local CSV file instead of calling `_generate_mock()`.
    *   Ensure your CSV file contains the following required columns with the specified data types: `DATA` (datetime), `ALMOXARIFADO` (string/object), `CODIGO` (string/object), `QUANTIDADE` (numeric), `UNITCOST` (numeric).

## Expected Output

Upon successful execution, the script will create an `outputs` directory containing the following artifacts:
*   `enhanced_final_report.txt`: A human-readable summary of the performance metrics and statistical comparisons.
*   `enhanced_raw_simulation_results.csv`: The raw, disaggregated data from every Monte Carlo replication.
*   `enhanced_scenario_summary.csv`: A summary table of the performance of each scenario.
*   `enhanced_statistical_tests.csv`: A detailed table of the statistical test results.
*   `parameters.csv`: A record of the economic parameters used in the simulation.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**. See the `LICENSE` file for details. You are free to share and adapt this work for non-commercial purposes, provided you give appropriate credit and distribute your contributions under the same license.
