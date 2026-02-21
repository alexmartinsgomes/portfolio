import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Markdown: Header
md_header = """# Quantitative Equity Portfolio Architecture: Lecture Series

Welcome to the interactive module on **Quantitative Portfolio Optimization**, built around Python, PyPortfolioOpt, SciPy, and Pandas.

In this notebook, we decompose the complete structure of a modern quantitative application into educational segments, explaining the mathematical foundations behind each step.

---

### Key Learning Objectives
1. **Data Ingestion & Integrity:** Utilizing `pandas.DatetimeIndex` frameworks safely.
2. **Matrix Condition & Optimization:** How Ledoit-Wolf shrinkage stabilizes the Efficient Frontier matrix.
3. **Parametric Risk Measurement:** Implementing Value at Risk (VaR) and Conditional VaR (CVaR).
4. **Stochastic Processes:** Simulating Geometric Brownian Motion (GBM) dynamically via high-performance hardware-accelerated NumPy matrices avoiding traditional logic loops.
"""

# Code: Imports
code_imports = """import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from pypfopt import risk_models, expected_returns, EfficientFrontier
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore')
"""

# Markdown: Data Provider
md_data = """## 1. Data Integrity and Smart Caching

In financial engineering, preventing data leakage and minimizing I/O overhead on external APIs (like Yahoo Finance) is crucial.
The `fetch_data` function guarantees that multiple ticker symbols map properly to uniform `float64` boundaries. Missing values on holidays or weekends are forward/backward filled securely.
"""

code_data = '''CACHE_FILE = "price_cache.parquet"

def fetch_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches adjusted close prices using a local pyarrow/parquet cache.
    
    Mathematical/Data Schema:
    - Returns: pandas.DataFrame
    - Index: pandas.DatetimeIndex
    - Columns: Ticker symbols (str)
    - Values: Adjusted Close prices (float64) representing the asset value $S_t$.
    """
    tickers = [t.upper() for t in tickers]
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    if end_dt.date() == datetime.today().date():
         end_dt = end_dt + timedelta(days=1)

    cached_df = pd.DataFrame()
    if os.path.exists(CACHE_FILE):
        cached_df = pd.read_parquet(CACHE_FILE)

    tickers_to_fetch = []
    fetch_start, fetch_end = start_dt, end_dt

    if not cached_df.empty:
        cached_tickers = cached_df.columns.tolist()
        tickers_to_fetch = [t for t in tickers if t not in cached_tickers]
        
        cached_start, cached_end = cached_df.index.min(), cached_df.index.max()
        if start_dt < cached_start or end_dt > cached_end + timedelta(days=1):
             tickers_to_fetch = tickers
             fetch_start, fetch_end = start_dt, end_dt
    else:
        tickers_to_fetch = tickers

    if tickers_to_fetch:
        print(f"Fetching {tickers_to_fetch} from yfinance...")
        df = yf.download(tickers_to_fetch, start=fetch_start, end=fetch_end, auto_adjust=False)
        
        new_data = pd.DataFrame()
        if len(tickers_to_fetch) == 1 and 'Adj Close' in df.columns:
            new_data = df[['Adj Close']].copy()
            new_data.columns = [tickers_to_fetch[0]]
        elif df.columns.nlevels > 1 and 'Adj Close' in df.columns.levels[0]:
            new_data = df['Adj Close'].copy()
                
        if not new_data.empty:
            if new_data.index.tz is not None:
                new_data.index = new_data.index.tz_localize(None)
                
            if not cached_df.empty and tickers_to_fetch != tickers:
                cached_df = pd.concat([cached_df, new_data], axis=1)
            elif not cached_df.empty:
                # Transpose, groupby to deduplicate overlap, transpose back (support pandas 2.1+)
                cached_df = pd.concat([cached_df, new_data], axis=1).T.groupby(level=0).first().T
            else:
                cached_df = new_data
            
            cached_df.sort_index(inplace=True)
            cached_df.to_parquet(CACHE_FILE)

    if not cached_df.empty:
        mask = (cached_df.index >= pd.to_datetime(start_date)) & (cached_df.index <= pd.to_datetime(end_date))
        result_df = cached_df.loc[mask]
        result_df = result_df[[t for t in tickers if t in result_df.columns]].ffill().bfill().astype('float64')
        return result_df
    
    return pd.DataFrame()
'''

# Markdown: Math Theory 1
md_math1 = r"""## 2. Mathematical Engines: Portfolio Optimization

Let $\mu$ be the vector of exponentially-weighted historical returns, and $\Sigma$ the sample covariance matrix.
According to Markowitz' Modern Portfolio Theory (MPT), an optimizer maps weights array $w$ to define the Efficient Frontier.

**Maximum Sharpe Ratio:**
$$ \max_w \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}} $$

**Numerical Stability (Ledoit-Wolf Shrinkage):**
Financial datasets often yield poorly scaled covariance matrices $cond(\Sigma) \gg 1$.
To prevent optimization algorithms from failing, we "shrink" the sample covariance against a structured target $F$:
$$ \Sigma_{shrink} = (1-\delta)\Sigma_{sample} + \delta F $$
(Ledoit & Wolf, 2004).
"""

code_math1 = '''def get_returns_and_cov(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Calculates EMA returns and applies Ledoit-Wolf shrinkage to standard covariance."""
    mu = expected_returns.ema_historical_return(df)
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    return mu, S

def optimize_portfolio(df: pd.DataFrame, strategy: str = "max_sharpe", target: float = None) -> pd.Series:
    """Invokes convex optimization solver (via cvxpy backend)."""
    mu, S = get_returns_and_cov(df)
    ef = EfficientFrontier(mu, S)
    
    if strategy == "max_sharpe":
        weights = ef.max_sharpe()
    elif strategy == "min_volatility":
        weights = ef.min_volatility()
    elif strategy == "efficient_return":
        weights = ef.efficient_return(target_return=target or mu.mean())
    elif strategy == "max_sortino":
        semi_cov = risk_models.semicovariance(df)
        ef = EfficientFrontier(mu, semi_cov)
        weights = ef.max_sharpe()
        
    cleaned_weights = ef.clean_weights()
    return pd.Series(cleaned_weights).astype('float64')
'''

# Markdown: Math Theory 2
md_math2 = r"""## 3. Parametric Risk Estimation

**Value at Risk (VaR)** estimates the maximum potential loss at a given confidence interval $\alpha$ assuming normality.
$$ VaR_{\alpha} = - (w^T\mu - z_{\alpha} \sqrt{w^T\Sigma w}) $$

**Conditional Value at Risk (CVaR / Expected Shortfall)** is the expected loss *given* that the VaR threshold is breached.
$$ CVaR_{\alpha} = - \left( w^T\mu - \sqrt{w^T\Sigma w} \frac{\phi(z_{\alpha})}{\alpha} \right) $$
where $\phi$ is the normal PDF.
"""

code_math2 = '''def calculate_var_cvar(weights: pd.Series, df: pd.DataFrame, alpha: float = 0.05) -> tuple[float, float]:
    returns = df.pct_change().dropna()
    mu, S = get_returns_and_cov(df)
    
    # Ensuring weight padding to prevent alignment faults if optimizer pruned a ticker
    w = weights.reindex(returns.columns, fill_value=0.0).values
    
    mu_p = np.dot(w, mu)
    sigma_p = np.sqrt(np.dot(w.T, np.dot(S, w)))
    
    z_alpha = norm.ppf(1 - alpha)
    
    var_loss = -(mu_p - z_alpha * sigma_p)
    cvar_loss = -(mu_p - sigma_p * (norm.pdf(z_alpha) / alpha))
    
    return float(var_loss), float(cvar_loss)
'''

# Markdown: MC Simulation
md_mc = r"""## 4. Vectorized Geometric Brownian Motion

To assess deep stochastic pathways, we simulate the Geometric Brownian Motion (GBM).
Asset price $S_t$ evolves over bounded steps:
$$ S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z \right) $$

For $\Delta t = 1$ day, we draw $Z \sim \mathcal{N}(0,1)$ concurrently across $N$ simulations using hardware-optimized memory allocation avoiding native Python `for` loop bottlenecks.
"""

code_mc = '''def run_monte_carlo_gbm(weights: pd.Series, df: pd.DataFrame, days_ahead: int = 252, n_simulations: int = 10000) -> pd.DataFrame:
    mu, S = get_returns_and_cov(df)
    w = weights.reindex(df.columns, fill_value=0.0).values
    
    # Portfolio expected daily parameters scaling annual frequency
    mu_p_daily = np.dot(w, mu) / 252.0
    sigma_p_daily = np.sqrt(np.dot(w.T, np.dot(S, w))) / np.sqrt(252.0)
    
    dt = 1 
    Z = np.random.standard_normal((days_ahead, n_simulations))
    
    drift = (mu_p_daily - 0.5 * sigma_p_daily**2) * dt
    diffusion = sigma_p_daily * np.sqrt(dt) * Z
    
    daily_returns = np.exp(drift + diffusion)
    initial_value = np.ones((1, n_simulations))
    
    paths = np.vstack([initial_value, daily_returns])
    cumulative_paths = np.cumprod(paths, axis=0)[1:, :]
    
    return pd.DataFrame(cumulative_paths, index=np.arange(1, days_ahead + 1), dtype='float64')
'''

# Markdown: Execution
md_exec = """## 5. Execution Pipeline Example

Finally, we bind the functions together. Here we fetch data for a basket of equities, optimize weights for the **Maximum Sharpe Ratio**, and plot the Monte Carlo density curve!
"""

code_exec = '''# 1. Pipeline Start
tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'TLT']
start_date = '2020-01-01'
end_date = '2024-01-01'

df = fetch_data(tickers, start_date, end_date)
print(f"Loaded {df.shape[0]} business days.")

# 2. Optimize Portfolio
weights = optimize_portfolio(df, "max_sharpe")
print("\\nOptimal Weights (Max Sharpe):")
print(weights[weights > 0.01].round(4) * 100)

var, cvar = calculate_var_cvar(weights, df)
print(f"\\nParametric VaR (95%): {var*100:.2f}% | CVaR: {cvar*100:.2f}%")

# 3. Running GBM Simulation natively in Jupyter
simulations = run_monte_carlo_gbm(weights, df, days_ahead=252, n_simulations=5000)

fig = go.Figure()

# Plot first 100 paths
for i in range(100):
    fig.add_trace(go.Scatter(x=simulations.index, y=simulations.iloc[:, i], mode='lines', 
                             line=dict(color='gray', width=1), opacity=0.1, showlegend=False))

# Plot Percentiles Overlay
percentiles = {1: 'red', 50: 'green', 99: 'red'}
for p, color in percentiles.items():
    p_vals = simulations.apply(lambda x: np.percentile(x, p), axis=1)
    fig.add_trace(go.Scatter(x=simulations.index, y=p_vals, mode='lines', 
                             name=f'{p}th Pct', line=dict(color=color, width=2, dash='solid' if p == 50 else 'dash')))

fig.update_layout(title="Monte Carlo GBM Density Bounds", yaxis_title="Normalized Value", template="plotly_white")
fig.show()
'''

nb['cells'] = [
    nbf.v4.new_markdown_cell(md_header),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(md_data),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_markdown_cell(md_math1),
    nbf.v4.new_code_cell(code_math1),
    nbf.v4.new_markdown_cell(md_math2),
    nbf.v4.new_code_cell(code_math2),
    nbf.v4.new_markdown_cell(md_mc),
    nbf.v4.new_code_cell(code_mc),
    nbf.v4.new_markdown_cell(md_exec),
    nbf.v4.new_code_cell(code_exec)
]

with open("lecture_notebook.ipynb", "w") as f:
    nbf.write(nb, f)
