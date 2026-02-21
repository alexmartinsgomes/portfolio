from data_provider import fetch_data
from engine import optimize_portfolio
df = fetch_data(['AAPL', 'MSFT', 'GOOGL', 'SPY', 'TLT'], '2023-01-01', '2024-01-01')
optimize_portfolio(df, "max_sharpe")
