import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
from scipy.stats import wilcoxon, friedmanchisquare, jarque_bera
import warnings
from mealpy import FloatVar
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import HHO, PSO, GWO, COA
from statsmodels.tsa.stattools import adfuller

# Suppress minor warnings for clean console output
warnings.filterwarnings('ignore')

print("="*80)
print("INITIATING SIMULATION: EMPIRICAL MINLP DIAGNOSTIC (5.1M FEs)")
print("="*80)

# 1. LOAD DATA & DEFINE FINANCIAL CONSTANTS
try:
    returns_df = pd.read_csv('nifty49_returns.csv', index_col=0, parse_dates=True)
    cov_matrix_df = pd.read_csv('nifty49_covariance.csv', index_col=0)
except FileNotFoundError:
    print("Error: Could not find CSV files. Ensure 'nifty49_returns.csv' and 'nifty49_covariance.csv' is in directory.")
    exit()

tickers = cov_matrix_df.columns.values
N = len(tickers)

# In-Sample vs Out-of-Sample Split (80% Train, 20% Test)
SPLIT_RATIO = 0.8
split_idx = int(len(returns_df) * SPLIT_RATIO)

train_returns = returns_df.iloc[:split_idx]
test_returns = returns_df.iloc[split_idx:]

annual_returns_train = (train_returns.mean() * 252).values
cov_matrix_train = train_returns.cov().values * 252

# Financial Constants
RF_RATE = 0.0675  # 6.75% India 10-Year G-Sec yield proxy
TRANSACTION_FEE = 0.001  # 0.1% dynamic rebalancing cost
INITIAL_WEIGHTS = np.ones(N) / N  # Assuming initial 1/N allocation for turnover proxy

# 1.5 PRE-OPTIMIZATION DATA DIAGNOSTICS (Normality & Stationarity)
print("Executing Pre-Optimization Data Diagnostics...")
market_proxy_returns = returns_df.mean(axis=1).dropna() 

jb_stat, jb_p_value = jarque_bera(market_proxy_returns)

adf_result = adfuller(market_proxy_returns)
adf_stat, adf_p_value = adf_result[0], adf_result[1]

# 2. OBJECTIVE FUNCTION (IN-SAMPLE OPTIMIZATION)
def calculate_fitness(solution):
    weights = np.array(solution)
    K = 10
    
    # Enforce Cardinality Constraint (Top K selection)
    top_k_idx = np.argsort(weights)[-K:]
    final_weights = np.zeros(N)
    final_weights[top_k_idx] = weights[top_k_idx]
    
    sum_w = np.sum(final_weights)
    if sum_w == 0: 
        return -9999.0
    final_weights = final_weights / sum_w
    
    # Continuous Bounding Constraints (5% to 20%)
    penalty = 0.0
    for idx in top_k_idx:
        if final_weights[idx] < 0.05: 
            penalty += (0.05 - final_weights[idx]) * 50.0
        elif final_weights[idx] > 0.20: 
            penalty += (final_weights[idx] - 0.20) * 50.0
            
    port_return = np.dot(final_weights, annual_returns_train)
    port_variance = np.dot(final_weights.T, np.dot(cov_matrix_train, final_weights))
    
    # Friction Cost (Turnover-based penalty)
    turnover = np.sum(np.abs(final_weights - INITIAL_WEIGHTS))
    friction_cost = TRANSACTION_FEE * turnover
    
    if port_variance <= 0: 
        return -9999.0
    
    # Adjusted Financial Sharpe Ratio
    port_volatility = np.sqrt(port_variance)
    sharpe_ratio = (port_return - RF_RATE - friction_cost) / port_volatility
    
    return sharpe_ratio - penalty

problem_dict = {
    "bounds": FloatVar(lb=[0.0]*N, ub=[1.0]*N, name="portfolio_weights"),
    "obj_func": calculate_fitness,
    "minmax": "max",
}

# 3. OUT-OF-SAMPLE BACKTESTING ENGINE
def run_out_of_sample_backtest(ga_w, pso_w, gwo_w, hho_w, coa_w, test_data, rf_annual):
    print("\n" + "="*80)
    print("EXECUTING OUT-OF-SAMPLE BACKTESTING & RISK DIAGNOSTICS")
    print("="*80)
    
    eq_weights = np.ones(N) / N
    
    portfolios = {
        'GA (1989)': ga_w, 'PSO (1995)': pso_w, 'GWO (2014)': gwo_w,
        'HHO (2019)': hho_w, 'COA (2023)': coa_w, 'Equal-Weight (1/N)': eq_weights
    }
    
    metrics_summary = []
    oos_curves = {}
    rf_daily = rf_annual / 252.0
    
    # Pre-calculate benchmark (1/N) metrics for the Information Ratio
    eq_daily_ret = test_data.dot(eq_weights)
    trading_days = len(test_data)
    eq_cum_growth = (1 + eq_daily_ret).cumprod()
    eq_ann_return = ((eq_cum_growth.iloc[-1]) ** (252 / trading_days)) - 1
    
    plt.figure(figsize=(12, 6))
    
    for name, w in portfolios.items():
        turnover = np.sum(np.abs(w - INITIAL_WEIGHTS))
        initial_friction_drag = TRANSACTION_FEE * turnover
        
        daily_ret = test_data.dot(w)
        cum_growth = (1 + daily_ret).cumprod() * (100.0 * (1 - initial_friction_drag))
        oos_curves[name] = cum_growth.values
        
        total_cum_return = ((cum_growth.iloc[-1] / 100.0) - 1.0) * 100.0
        ann_return = ((cum_growth.iloc[-1] / 100.0) ** (252 / trading_days)) - 1
        ann_volatility = daily_ret.std() * np.sqrt(252)
        sharpe = (ann_return - rf_annual) / ann_volatility if ann_volatility > 0 else 0.0
        
        downside_diff = daily_ret[daily_ret < rf_daily] - rf_daily
        downside_std = np.sqrt(np.mean(downside_diff**2)) * np.sqrt(252)
        sortino = (ann_return - rf_annual) / downside_std if downside_std > 0 else 0.0
        
        max_drawdown = ((cum_growth - cum_growth.cummax()) / cum_growth.cummax()).min() * 100.0 
        
        # Calculate Information Ratio (Active Return / Tracking Error)
        if name != 'Equal-Weight (1/N)':
            active_return_daily = daily_ret - eq_daily_ret
            tracking_error = active_return_daily.std() * np.sqrt(252)
            info_ratio = (ann_return - eq_ann_return) / tracking_error if tracking_error > 0 else 0.0
        else:
            info_ratio = 0.0 # Benchmark relative to itself
            
        metrics_summary.append({
            'Portfolio Strategy': name,
            'Cum. Return (%)': round(total_cum_return, 2),
            'Ann. Return (%)': round(ann_return * 100, 2),
            'Ann. Volatility (%)': round(ann_volatility * 100, 2),
            'Sharpe Ratio': round(sharpe, 4),
            'Info Ratio': round(info_ratio, 4),
            'Sortino Ratio': round(sortino, 4),
            'Max Drawdown (%)': round(max_drawdown, 2)
        })
        
        if 'GA' in name: style, color, width = '--', '#2ca02c', 2.0
        elif 'PSO' in name: style, color, width = ':', '#d62728', 2.0
        elif 'GWO' in name: style, color, width = '-.', '#ff7f0e', 2.0
        elif 'HHO' in name: style, color, width = '-', '#1f77b4', 2.0
        elif 'COA' in name: style, color, width = '-', '#9467bd', 2.5
        else: style, color, width = '--', '#7f7f7f', 1.5
            
        plt.plot(pd.to_datetime(cum_growth.index), cum_growth.values, label=name, linestyle=style, color=color, linewidth=width)

    plt.xlabel("Date", fontsize=14, fontweight='bold')
    plt.ylabel("Portfolio Value (Base = ₹100)", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="upper left", frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.savefig("oos_equity_curve.png", dpi=600, bbox_inches='tight')
    
    oos_df = pd.DataFrame(oos_curves, index=test_data.index)
    oos_df.to_csv("oos_equity_curves.csv")
    
    return pd.DataFrame(metrics_summary).set_index('Portfolio Strategy')

# 4. EXECUTION (51 Runs, 20k FEs Per Run | 5.1M Total)
TOTAL_RUNS = 51
EPOCHS = 200
POPULATION = 100

def run_optimizer(model_class, model_name):
    print(f"\nDeploying {model_name} for {TOTAL_RUNS} runs (Budget: {TOTAL_RUNS * EPOCHS * POPULATION} FEs)...")
    results, best_fitness = [], -9999
    best_history, best_solution = None, None
    
    start_time = time.time()
    
    for i in range(TOTAL_RUNS):
        model = model_class(epoch=EPOCHS, pop_size=POPULATION, seed=i)
        best_sol = model.solve(problem_dict)
        fitness = best_sol.target.fitness
        results.append(fitness)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_history = model.history.list_global_best_fit
            best_solution = best_sol.solution
            
        if (i+1) % 10 == 0 or (i+1) == TOTAL_RUNS:
            print(f"  -> Completed run {i+1}/{TOTAL_RUNS}. Current Best Sharpe: {best_fitness:.4f}")
            
    end_time = time.time()
    mean_execution_time = (end_time - start_time) / TOTAL_RUNS
            
    return np.array(results), best_history, best_solution, mean_execution_time

# Executed chronologically
ga_res, ga_hist, ga_sol, ga_time   = run_optimizer(GA.BaseGA, "Genetic Algorithm (GA)")
pso_res, pso_hist, pso_sol, pso_time = run_optimizer(PSO.OriginalPSO, "Particle Swarm (PSO)")
gwo_res, gwo_hist, gwo_sol, gwo_time = run_optimizer(GWO.OriginalGWO, "Grey Wolf (GWO)")
hho_res, hho_hist, hho_sol, hho_time = run_optimizer(HHO.OriginalHHO, "Harris Hawks (HHO)")
coa_res, coa_hist, coa_sol, coa_time = run_optimizer(COA.OriginalCOA, "Coati (COA)")

# 5. POST-PROCESSING & STATISTICAL ANALYTICS
def extract_weights(solution_vector):
    w = np.array(solution_vector)
    top10 = np.argsort(w)[-10:]
    final = np.zeros(N)
    final[top10] = w[top10]
    return final / np.sum(final)

ga_w  = extract_weights(ga_sol)
pso_w = extract_weights(pso_sol)
gwo_w = extract_weights(gwo_sol)
hho_w = extract_weights(hho_sol)
coa_w = extract_weights(coa_sol)

def get_stats(res_array):
    return np.max(res_array), np.min(res_array), np.mean(res_array), np.median(res_array), np.std(res_array)

ga_stats  = get_stats(ga_res)
pso_stats = get_stats(pso_res)
gwo_stats = get_stats(gwo_res)
hho_stats = get_stats(hho_res)
coa_stats = get_stats(coa_res)

# A. Statistical Validation Matrix
friedman_stat, p_value_friedman = friedmanchisquare(ga_res, pso_res, gwo_res, hho_res, coa_res)

algorithms = ['GA', 'PSO', 'GWO', 'HHO', 'COA']
res_list = [ga_res, pso_res, gwo_res, hho_res, coa_res]
wilcoxon_matrix = pd.DataFrame(index=algorithms, columns=algorithms, dtype=float)

for i, alg1 in enumerate(algorithms):
    for j, alg2 in enumerate(algorithms):
        if i == j:
            wilcoxon_matrix.loc[alg1, alg2] = np.nan
        else:
            _, p_val = wilcoxon(res_list[i], res_list[j])
            wilcoxon_matrix.loc[alg1, alg2] = p_val

# B. Topological Diagnostics Matrix
weights_dict = {
    'GA (1989)': ga_w,
    'PSO (1995)': pso_w,
    'GWO (2014)': gwo_w,
    'HHO (2019)': hho_w,
    'COA (2023)': coa_w
}

# Create a master dataframe of all allocations
allocations_df = pd.DataFrame(weights_dict, index=tickers)

# Filter out assets ignored by ALL algorithms to keep the table clean
allocations_df = allocations_df.loc[(allocations_df != 0).any(axis=1)] * 100.0

# Calculate HHI and Entropy for all algorithms
topo_metrics = []
for name, w in weights_dict.items():
    active_w = w[w > 0]
    hhi = np.sum(active_w**2)
    entropy = -np.sum(active_w * np.log(active_w))
    topo_metrics.append({'Algorithm': name, 'HHI': round(hhi, 4), 'Shannon Entropy': round(entropy, 4)})

topo_summary_df = pd.DataFrame(topo_metrics).set_index('Algorithm')

backtest_results = run_out_of_sample_backtest(ga_w, pso_w, gwo_w, hho_w, coa_w, test_returns, RF_RATE)

# 6. EXPORT TEXT REPORT, VISUALIZATIONS, & PERSIST DATA
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
# Clean tickers for y-axis readability
clean_tickers = [t.replace('.NS', '') for t in allocations_df.index]
sns.heatmap(allocations_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Capital Allocation (%)'},
            yticklabels=clean_tickers, linewidths=.5)

plt.xlabel("Metaheuristic Architecture", fontsize=14, fontweight='bold')
plt.ylabel("Active Nifty 50 Equities", fontsize=14, fontweight='bold')
plt.yticks(rotation=0) # Make y-axis labels perfectly horizontal
plt.tight_layout()
plt.savefig("topological_heatmap.png", dpi=600, bbox_inches='tight')

# SAVE WORKSPACE STATE (PERSISTENCE)
print("Saving simulation workspace to disk...")
workspace_data = {
    'ga_res': ga_res, 'pso_res': pso_res, 'gwo_res': gwo_res, 'hho_res': hho_res, 'coa_res': coa_res,
    'ga_hist': ga_hist, 'pso_hist': pso_hist, 'gwo_hist': gwo_hist, 'hho_hist': hho_hist, 'coa_hist': coa_hist,
    'ga_w': ga_w, 'pso_w': pso_w, 'gwo_w': gwo_w, 'hho_w': hho_w, 'coa_w': coa_w,
    'backtest_results': backtest_results,
    'wilcoxon_matrix': wilcoxon_matrix,
    'allocations_df': allocations_df
}

with open('simulation_workspace.pkl', 'wb') as f:
    pickle.dump(workspace_data, f)

print("\n" + "="*80)
print("SIMULATION COMPLETE. ALL FIGURES, REPORTS, & WORKSPACE DATA SAVED.")
print("="*80)