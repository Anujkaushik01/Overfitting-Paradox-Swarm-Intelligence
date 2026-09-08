# The Overfitting Paradox in Swarm Intelligence

**Official code and data repository for the manuscript:** *The Overfitting Paradox in Swarm Intelligence: An Empirical Diagnostic of Friction-Aware Mixed-Integer Non-Linear Programming Portfolio Optimization.*

## Overview
This repository contains the complete computational framework, historical market datasets, and reproducibility scripts required to validate the empirical findings of the study. The experiment evaluates 5 metaheuristic architectures (GA, PSO, GWO, HHO, COA) across 5.1 million function evaluations on the NSE Nifty 50 index using a custom turnover-penalized MINLP formulation.

## Repository Structure
* `data_acquisition.py`: Downloads and preprocesses NSE Nifty 50 historical market data.
* `portfolio_optimization.py`: The core simulation engine executing the algorithms.
* `reproduce.py`: A unified script to extract statistical metrics and generate all manuscript figures directly from the saved workspace.
* `simulation_workspace.pkl`: The serialized state of all 51 independent simulation trials.
* `oos_equity_curves.csv`: Pre-calculated out-of-sample equity growth trajectories used for plotting.
* `statistical_report.txt`: The final exported statistical validation and performance metrics.
* `nifty49_*.csv`: The historical price, return, and covariance datasets.

## How to Reproduce the Results
1. Clone this repository to your local machine.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Execute the reproducibility script: `python reproduce.py`

This will instantly output the `statistical_report.txt` and all manuscript figures (`.png`) locally, completely bypassing the need to re-execute the 5.1 million function evaluations.
