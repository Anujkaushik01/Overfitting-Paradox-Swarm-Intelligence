import yfinance as yf
import pandas as pd
import numpy as np
import time

# 1. The 49 Constituents (Tata Motors removed for continuous listing integrity)
nifty49_tickers = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", 
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", 
    "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS", 
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", 
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", 
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", 
    "ITC.NS", "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", 
    "KOTAKBANK.NS", "LTM.NS", "LT.NS", "M&M.NS", 
    "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS", 
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", 
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATASTEEL.NS", 
    "TECHM.NS", "TITAN.NS", "UPL.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

start_date = "2022-01-01"
end_date = "2026-07-01"

print("Extracting Data\n...")

historical_data = {}
failed_downloads = []

# 2. Sequential Extraction
for i, ticker in enumerate(nifty49_tickers):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            # history() automatically adjusts for stock splits (vital for LTM)
            df = stock.history(start=start_date, end=end_date)
            
            if not df.empty and 'Close' in df.columns:
                historical_data[ticker] = df['Close']
                print(f"[{i+1}/49] ✅ {ticker} successful.")
                break
            else:
                if attempt < max_retries - 1:
                    time.sleep(2)
                elif attempt == max_retries - 1:
                    print(f"[{i+1}/49] ❌ {ticker} failed.")
                    failed_downloads.append(ticker)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
            elif attempt == max_retries - 1:
                print(f"[{i+1}/49] ❌ {ticker} failed.")
                failed_downloads.append(ticker)
        
    time.sleep(1) # Pause to prevent Yahoo Finance IP bans

# Combine into a single DataFrame
data = pd.DataFrame(historical_data)

print("\nExecuting mathematical data imputation...")

# 3. The Cleaning Pipeline
data = data.dropna(axis=1, how='all')
data = data.ffill()  # Forward-fill minor 1-day gaps (like trading holidays)
data = data.bfill()  # Backward-fill minor gaps at the very start of the series

# 4. Financial Mathematics
daily_returns = np.log(data / data.shift(1))
daily_returns = daily_returns.dropna(axis=0, how='all')

annualized_returns = daily_returns.mean() * 252
annualized_cov_matrix = daily_returns.cov() * 252

# 5. Export 
data.to_csv("nifty49_prices.csv")
daily_returns.to_csv("nifty49_returns.csv")
annualized_cov_matrix.to_csv("nifty49_covariance.csv")

print(f"\nDone! Data completely pre-processed.")
print(f"Number of assets: {len(data.columns)}")

if failed_downloads:
    print(f"\nAssets that failed: {failed_downloads}")