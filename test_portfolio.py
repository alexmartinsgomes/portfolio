import pytest
import numpy as np
import pandas as pd
from engine import normalize_weights, get_returns_and_cov, optimize_portfolio, calculate_var_cvar, run_monte_carlo_gbm, calculate_marginal_contribution

@pytest.fixture
def mock_price_data():
    """Generates synthetic log-normal price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq='B')
    
    # 3 assets with different volatilities and drift
    # Increase variance to avoid condition number issues with CVXPY Solvers
    returns = np.random.multivariate_normal(
        mean=[0.005, 0.002, 0.001], 
        cov=[[0.005, 0.002, 0.001], 
             [0.002, 0.004, 0.0015], 
             [0.001, 0.0015, 0.006]], 
        size=100
    )
    
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=['AAPL', 'MSFT', 'GOOGL'])
    return df.astype('float64')

def test_normalize_weights():
    w = np.array([0.5, 0.5, 1.0])
    w_norm = normalize_weights(w)
    
    assert np.isclose(np.sum(w_norm), 1.0)
    assert np.allclose(w_norm, [0.25, 0.25, 0.5])
    
    # Test zero sum
    w_zero = np.array([0.0, 0.0])
    w_norm_zero = normalize_weights(w_zero)
    assert np.isclose(np.sum(w_norm_zero), 1.0)
    assert np.allclose(w_norm_zero, [0.5, 0.5])

def test_get_returns_and_cov(mock_price_data):
    mu, S = get_returns_and_cov(mock_price_data)
    
    # Check shapes and types
    assert isinstance(mu, pd.Series)
    assert isinstance(S, pd.DataFrame)
    
    assert len(mu) == 3
    assert S.shape == (3, 3)
    
    assert mu.dtype == 'float64'
    assert S.dtypes.iloc[0] == 'float64'

def test_optimize_portfolio(mock_price_data):
    weights_sharpe = optimize_portfolio(mock_price_data, strategy="max_sharpe")
    
    assert isinstance(weights_sharpe, pd.Series)
    assert weights_sharpe.dtype == 'float64'
    assert np.isclose(weights_sharpe.sum(), 1.0)
    
    weights_vol = optimize_portfolio(mock_price_data, strategy="min_volatility")
    assert np.isclose(weights_vol.sum(), 1.0)

def test_calculate_var_cvar(mock_price_data):
    weights = pd.Series([0.4, 0.4, 0.2], index=['AAPL', 'MSFT', 'GOOGL'])
    
    var, cvar = calculate_var_cvar(weights, mock_price_data, alpha=0.05)
    
    assert isinstance(var, float)
    assert isinstance(cvar, float)
    assert cvar > var # CVaR should be strictly greater than VaR for normal distribution

def test_run_monte_carlo_gbm(mock_price_data):
    weights = pd.Series([0.4, 0.4, 0.2], index=['AAPL', 'MSFT', 'GOOGL'])
    
    simulations = run_monte_carlo_gbm(weights, mock_price_data, days_ahead=20, n_simulations=100)
    
    assert isinstance(simulations, pd.DataFrame)
    assert simulations.shape == (20, 100)
    assert simulations.dtypes.iloc[0] == 'float64'

def test_calculate_marginal_contribution(mock_price_data):
    weights = pd.Series([0.4, 0.4, 0.2], index=['AAPL', 'MSFT', 'GOOGL'])
    
    mcr = calculate_marginal_contribution(weights, mock_price_data)
    
    assert isinstance(mcr, pd.Series)
    assert mcr.dtype == 'float64'
    assert len(mcr) == 3
