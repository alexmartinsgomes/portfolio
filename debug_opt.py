import pandas as pd
from engine import optimize_portfolio
from data_provider import fetch_data
import warnings

warnings.filterwarnings('ignore')

try:
    df = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2020-01-01', '2024-01-01')
    weights = optimize_portfolio(df, 'max_sharpe')
    print("Optimization success:", weights)
except Exception as e:
    print(f"ERROR: {type(e).__name__} - {e}")
