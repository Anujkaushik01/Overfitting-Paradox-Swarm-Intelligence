"""
Reproducibility Script for:
"The Overfitting Paradox in Swarm Intelligence: An Empirical Diagnostic of Friction-Aware MINLP Portfolio Optimization"

Instructions for Reviewers:
1. Place 'Simulation_Workspace.pkl' and 'OOS_Equity_Curves.csv' in the same directory as this script.
2. Run this script to extract exact topological diagnostics, generate the statistical report, 
   and export the figures utilized in the manuscript.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from scipy.stats import friedmanchisquare, jarque_bera
from statsmodels.tsa.stattools import adfuller

WORKSPACE_FILE = 'simulation_workspace.pkl'
CSV_FILE = 'oos_equity_curves.csv'

if not os.path.exists(WORKSPACE_FILE) or not os.path.exists(CSV_FILE):
    print(f"Error: Missing {WORKSPACE_FILE} or {CSV_FILE}.")
    exit()

print("Loading validated workspace data...")
with open(WORKSPACE_FILE, 'rb') as f:
    ws = pickle.load(f)

ga_res, pso_res, gwo_res, hho_res, coa_res = ws['ga_res'], ws['pso_res'], ws['gwo_res'], ws['hho_res'], ws['coa_res']
ga_hist, pso_hist, gwo_hist, hho_hist, coa_hist = ws['ga_hist'], ws['pso_hist'], ws['gwo_hist'], ws['hho_hist'], ws['coa_hist']
allocations_df = ws['allocations_df']
wilcoxon_matrix = ws['wilcoxon_matrix']
backtest_results = ws['backtest_results']
algorithms = ['GA (1989)', 'PSO (1995)', 'GWO (2014)', 'HHO (2019)', 'COA (2023)']

# Calculate convergence
def get_stats(res_array):
    return np.max(res_array), np.min(res_array), np.mean(res_array), np.median(res_array), np.std(res_array)

ga_stats = get_stats(ga_res)
pso_stats = get_stats(pso_res)
gwo_stats = get_stats(gwo_res)
hho_stats = get_stats(hho_res)
coa_stats = get_stats(coa_res)

# 1. Friedman P-Value from the loaded workspace data
_, p_value_friedman = friedmanchisquare(ga_res, pso_res, gwo_res, hho_res, coa_res)

# 2. Data Diagnostics (JB & ADF) from the dataset
returns_df = pd.read_csv('nifty49_returns.csv', index_col=0, parse_dates=True)
market_proxy_returns = returns_df.mean(axis=1).dropna() 

_, jb_p_value = jarque_bera(market_proxy_returns)
adf_p_value = adfuller(market_proxy_returns)[1]

# 3. CPU times represent the author's specific hardware environment and are preserved here:
ga_time, pso_time, gwo_time, hho_time, coa_time = 4.74, 2.57, 2.95, 3.76, 3.36

# Topological calculation
weights_dict = {
    'GA (1989)': ws['ga_w'], 'PSO (1995)': ws['pso_w'], 'GWO (2014)': ws['gwo_w'],
    'HHO (2019)': ws['hho_w'], 'COA (2023)': ws['coa_w']
}

topo_metrics = []
for name, w in weights_dict.items():
    active_w = w[w > 0]
    hhi = np.sum(active_w**2)
    entropy = -np.sum(active_w * np.log(active_w))
    topo_metrics.append({'Algorithm': name, 'HHI': round(hhi, 4), 'Shannon Entropy': round(entropy, 4)})
topo_summary_df = pd.DataFrame(topo_metrics).set_index('Algorithm')

# 1. EXPORT REPORT TXT
print("\nExporting Statistical Report...")
with open("statistical_report.txt", "w") as f:
    f.write("BENCHMARK SUMMARY (51 RUNS | 20,000 FEs PER RUN | 5 ALGORITHMS)\n")
    f.write("="*75 + "\n\n")
    
    f.write("I. DATA DIAGNOSTICS (Market Index Proxy)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Jarque-Bera (Normality)   : p-value = {jb_p_value:.4e} (Non-normal if < 0.05)\n")
    f.write(f"ADF Test (Stationarity)   : p-value = {adf_p_value:.4e} (Stationary if < 0.05)\n\n")

    f.write("II. IN-SAMPLE ALGORITHMIC CONVERGENCE & COMPUTATIONAL COST\n")
    f.write("-" * 40 + "\n")
    f.write(f"1. GA  (1989): Best {ga_stats[0]:.4f} | Mean {ga_stats[2]:.4f} | Std {ga_stats[4]:.4f} | CPU/Run: {ga_time:.2f}s\n")
    f.write(f"2. PSO (1995): Best {pso_stats[0]:.4f} | Mean {pso_stats[2]:.4f} | Std {pso_stats[4]:.4f} | CPU/Run: {pso_time:.2f}s\n")
    f.write(f"3. GWO (2014): Best {gwo_stats[0]:.4f} | Mean {gwo_stats[2]:.4f} | Std {gwo_stats[4]:.4f} | CPU/Run: {gwo_time:.2f}s\n")
    f.write(f"4. HHO (2019): Best {hho_stats[0]:.4f} | Mean {hho_stats[2]:.4f} | Std {hho_stats[4]:.4f} | CPU/Run: {hho_time:.2f}s\n")
    f.write(f"5. COA (2023): Best {coa_stats[0]:.4f} | Mean {coa_stats[2]:.4f} | Std {coa_stats[4]:.4f} | CPU/Run: {coa_time:.2f}s\n\n")
    
    f.write("III. STATISTICAL VALIDATION (OMNIBUS & FULL PAIRWISE MATRIX)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Friedman Rank Test (Global): p-value = {p_value_friedman:.4e} (Significant if < 0.05)\n\n")
    f.write("Wilcoxon Signed-Rank Test Matrix (p-values):\n")
    f.write(wilcoxon_matrix.to_string(na_rep="---", float_format=lambda x: f"{x:.4e}"))
    f.write("\n\n")
    
    f.write("IV. TOPOLOGICAL DIAGNOSTICS (SYSTEMIC RISK DECENTRALIZATION)\n")
    f.write("-" * 40 + "\n")
    f.write(topo_summary_df.to_string())
    f.write("\n\nFINAL IN-SAMPLE ASSET ALLOCATION MATRIX (%):\n")
    f.write("-" * 40 + "\n")
    f.write(allocations_df.round(2).replace(0.0, "---").to_string())
    f.write("\n\n")
        
    f.write("="*75 + "\n")
    f.write("V. OUT-OF-SAMPLE BACKTESTING RESULTS (NET OF DYNAMIC FRICTION)\n")
    f.write("="*75 + "\n")
    f.write(backtest_results.to_string())

# 2. GENERATE FIGURES
print("Generating Figures...")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Fig: Convergence Curve
plt.figure(figsize=(8, 6))
plt.plot(ga_hist, color='#2ca02c', linewidth=2.0, linestyle='--', label="GA (1989)")
plt.plot(pso_hist, color='#d62728', linewidth=2.0, linestyle=':', label="PSO (1995)")
plt.plot(gwo_hist, color='#ff7f0e', linewidth=2.0, linestyle='-.', label="GWO (2014)")
plt.plot(hho_hist, color='#1f77b4', linewidth=2.0, linestyle='-', label="HHO (2019)")
plt.plot(coa_hist, color='#9467bd', linewidth=2.5, linestyle='-', label="COA (2023)")

plt.xlabel("Epochs", fontsize=14, fontweight='bold')
plt.ylabel("Fitness (Adjusted Sharpe Ratio)", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc="lower right", frameon=True, edgecolor='black')
plt.tight_layout()
plt.savefig("convergence_curve.png", dpi=600, bbox_inches='tight')

# Fig: Stability Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=[ga_res, pso_res, gwo_res, hho_res, coa_res], palette=['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4', '#9467bd'])
plt.xticks([0, 1, 2, 3, 4], ['GA\n(1989)', 'PSO\n(1995)', 'GWO\n(2014)', 'HHO\n(2019)', 'COA\n(2023)'], fontweight='bold')

plt.ylabel("Adjusted Sharpe Ratio", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("stability_boxplot.png", dpi=600, bbox_inches='tight')

# Fig: Topological Allocation Heatmap
plt.figure(figsize=(12, 8))
clean_tickers = [t.replace('.NS', '') for t in allocations_df.index]
sns.heatmap(allocations_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Capital Allocation (%)'},
            yticklabels=clean_tickers, linewidths=.5)

plt.xlabel("Metaheuristic Architecture", fontsize=14, fontweight='bold')
plt.ylabel("Active Nifty 50 Equities", fontsize=14, fontweight='bold')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("topological_heatmap.png", dpi=600, bbox_inches='tight')

# Fig: OOS Equity Curve
oos_df = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
STYLE_MAP = {
    'GA (1989)':          {'color': '#2ca02c', 'style': '--', 'width': 2.0},
    'PSO (1995)':         {'color': '#d62728', 'style': ':',  'width': 2.0},
    'GWO (2014)':         {'color': '#ff7f0e', 'style': '-.', 'width': 2.0},
    'HHO (2019)':         {'color': '#1f77b4', 'style': '-',  'width': 2.0},
    'COA (2023)':         {'color': '#9467bd', 'style': '-',  'width': 2.5},
    'Equal-Weight (1/N)': {'color': '#7f7f7f', 'style': '--', 'width': 1.5}
}

plt.figure(figsize=(12, 6))
for col in oos_df.columns:
    s = STYLE_MAP.get(col, {'color': 'black', 'style': '-', 'width': 1.0})
    plt.plot(pd.to_datetime(oos_df.index), oos_df[col], label=col, linestyle=s['style'], color=s['color'], linewidth=s['width'])

plt.xlabel("Date", fontsize=14, fontweight='bold')
plt.ylabel("Portfolio Value (Base = ₹100)", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="upper left", frameon=True, edgecolor='black')
plt.tight_layout()
plt.savefig("oos_equity_curve.png", dpi=600, bbox_inches='tight')

print("\n" + "="*80)
print("REPRODUCTION COMPLETE. ALL FIGURES & REPORT TXT EXTRACTED FROM WORKSPACE.")
print("="*80)