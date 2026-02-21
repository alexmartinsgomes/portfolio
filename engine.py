import numpy as np
import pandas as pd
from scipy.stats import norm
from pypfopt import risk_models, expected_returns, EfficientFrontier, base_optimizer

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Uniformly normalizes an array of weights such that sum(W_new) = 1.
    
    Expected Shape: (N,) where N is number of assets.
    """
    w_sum = np.sum(weights)
    if w_sum == 0:
        return np.ones(len(weights)) / len(weights)
    return weights / w_sum

def get_returns_and_cov(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calculates expected returns and covariance matrix from a DataFrame of prices.
    Applies Ledoit-Wolf shrinkage to the covariance matrix for numerical stability.
    
    Expected Shape:
    - Input: pandas.DataFrame (T, N) where T = periods, N = assets. Float64.
    - Output 1: pandas.Series (N,) of expected returns.
    - Output 2: pandas.DataFrame (N, N) of shrinkaged covariance. Float64.
    """
    # Using exponentially weighted historical returns (more responsive)
    mu = expected_returns.ema_historical_return(df)
    
    # Ledoit-Wolf Shrinkage for numerical stability:
    # Σ_shrink = (1-δ)Σ_sample + δF
    # This ensures cond(Σ) is well-behaved for optimizers.
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    
    return mu, S

def optimize_portfolio(df: pd.DataFrame, strategy: str = "max_sharpe", target_return: float = None) -> pd.Series:
    """
    Optimizes a portfolio based on a specific strategy.
    
    Expected Shape:
    - Input: pandas.DataFrame (T, N) of prices. Float64.
    - Output: pandas.Series (N,) of optimized weights. Float64.
    """
    mu, S = get_returns_and_cov(df)
    
    # Initialize PyPortfolioOpt Efficient Frontier
    ef = EfficientFrontier(mu, S)
    
    weights = {}
    if strategy == "max_sharpe":
        weights = ef.max_sharpe()
    elif strategy == "min_volatility":
        weights = ef.min_volatility()
    elif strategy == "efficient_return":
        if target_return is None:
            target_return = mu.mean() # Default to average return
        weights = ef.efficient_return(target_return=target_return)
    elif strategy == "max_sortino":
        # Sortino requires Semi-Covariance which pypfopt handles via SemiVariance
        semi_cov = risk_models.semicovariance(df)
        ef = EfficientFrontier(mu, semi_cov)
        weights = ef.max_sharpe() # Max Sharpe on semicovariance is implicitly Max Sortino
    else:
         raise ValueError(f"Unknown strategy: {strategy}")
         
    # Clean weights (remove tiny artifacts like 1e-16)
    cleaned_weights = ef.clean_weights()
    
    # Return as pandas Series aligned with asset names (float64)
    return pd.Series(cleaned_weights).astype('float64')

def calculate_var_cvar(weights: pd.Series, df: pd.DataFrame, alpha: float = 0.05) -> tuple[float, float]:
    """
    Calculates Parametric Value at Risk (VaR) and Conditional VaR (CVaR).
    
    Formula:
    VaR = mu_p - z_alpha * sigma_p
    CVaR = mu_p - sigma_p * (phi(z_alpha) / (1 - alpha))
    
    Expected Shape:
    - Input weights: pandas.Series (N,). Float64.
    - Input df: pandas.DataFrame (T, N) of prices.
    - Output: tuple of (VaR, CVaR) as float64. Units are percent (e.g. 0.05 = 5%).
    """
    returns = df.pct_change().dropna()
    mu, S = get_returns_and_cov(df)
    
    # Ensure alignment by padding missing zero-weights
    w = weights.reindex(returns.columns, fill_value=0.0).values
    
    mu_p = np.dot(w, mu)
    sigma_p = np.sqrt(np.dot(w.T, np.dot(S, w)))
    
    z_alpha = norm.ppf(1 - alpha)
    
    # We return the absolute loss value. A VaR of 0.05 means a 5% loss.
    # Formulae usually yield negative numbers for loss, we take absolute mapping.
    # VaR_alpha = - (mu - Z * sigma)
    # CVaR_alpha = - (mu - sigma * (phi(Z) / alpha))
    var_loss = -(mu_p - z_alpha * sigma_p)
    cvar_loss = -(mu_p - sigma_p * (norm.pdf(z_alpha) / alpha))
    
    # Empirical calculation for robustness check (not returned but good to have)
    port_returns = returns.dot(w)
    historical_var = -np.percentile(port_returns, alpha * 100)
    
    return float(var_loss), float(cvar_loss)

def run_monte_carlo_gbm(weights: pd.Series, df: pd.DataFrame, days_ahead: int = 252, n_simulations: int = 10000) -> pd.DataFrame:
    """
    Simulates portfolio paths using Geometric Brownian Motion (GBM).
    Uses high-performance NumPy vectorization avoiding generic Python loops.
    
    Formula:
    S_{t+dt} = S_t * exp((mu - sigma^2/2)*dt + sigma * sqrt(dt) * Z)
    where Z ~ N(0, 1).
    
    Expected Shape:
    - Outputs: pandas.DataFrame (days_ahead, n_simulations). Float64.
    - Index: Integer days from 1 to days_ahead.
    - Data Units: Normalized portfolio value (starts at 1.0).
    """
    returns = df.pct_change().dropna()
    mu, S = get_returns_and_cov(df)
    
    w = weights.reindex(returns.columns, fill_value=0.0).values
    
    # Portfolio expected daily return and volatility
    mu_p_daily = np.dot(w, mu) / 252.0
    sigma_p_daily = np.sqrt(np.dot(w.T, np.dot(S, w))) / np.sqrt(252.0)
    
    # Vectorized GBM
    dt = 1 # 1 day step
    
    # Pre-generate random standard normal shocks
    Z = np.random.standard_normal((days_ahead, n_simulations))
    
    # Calculate daily periodic returns
    drift = (mu_p_daily - 0.5 * sigma_p_daily**2) * dt
    diffusion = sigma_p_daily * np.sqrt(dt) * Z
    
    daily_returns = np.exp(drift + diffusion)
    
    # Initial portfolio value normalized to 1.0
    initial_value = np.ones((1, n_simulations))
    
    # Combine and calculate cumulative paths
    paths = np.vstack([initial_value, daily_returns])
    cumulative_paths = np.cumprod(paths, axis=0)
    
    # Drop the initial 0th day to yield exactly 'days_ahead' periods
    cumulative_paths = cumulative_paths[1:, :]
    
    return pd.DataFrame(cumulative_paths, index=np.arange(1, days_ahead + 1), dtype='float64')

def calculate_marginal_contribution(weights: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    Calculates the Marginal Contribution to Risk (MCR) for each asset.
    
    Formula: MCR_i = (Sigma * w)_i / sigma_p
    
    Expected Shape:
    - Output: pandas.Series (N,). Float64.
    """
    mu, S = get_returns_and_cov(df)
    w = weights.reindex(df.columns, fill_value=0.0).values
    
    sigma_p = np.sqrt(np.dot(w.T, np.dot(S, w)))
    
    # Marginal contribution: (Σw) / σ_p
    mcr = np.dot(S, w) / sigma_p
    
    return pd.Series(mcr, index=df.columns, dtype='float64')
