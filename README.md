<div align="center">
  
# 📈 Quantitative Equity Portfolio Architect

> A high-performance, mathematically rigorous quantitative portfolio visualization and optimization suite built with Python and Gradio.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://gradio.app/)
[![PyPortfolioOpt](https://img.shields.io/badge/PyPortfolioOpt-Optimization-green.svg)](https://pyportfolioopt.readthedocs.io/)
[![uv](https://img.shields.io/badge/uv-Dependency%20Manager-purple.svg)](https://github.com/astral-sh/uv)

</div>

<hr/>

## 📖 Overview

The **Quantitative Equity Portfolio Architect** is an advanced interactive Web application designed for Financial Engineers, Quants, and Portfolio Managers. 

Powered by **PyPortfolioOpt** and mathematically rigorous structural constraints (strict `float64` floating point precision, native NumPy vectorization), it allows users to effortlessly fetch historical equity data, calculate Efficient Frontier optimizers, assess downside risk (Parametric VaR & CVaR), and run tens of thousands of Geometric Brownian Motion (GBM) Monte Carlo paths asynchronously without locking the UI.

## 🌟 Key Features

### 📡 1. Robust Data Fetching & Caching (`data_provider.py`)
- **Smart `.parquet` Caching**: Minimizes `yfinance` network calls. Only queries the API for out-of-bound Date range misses or uncached asset tickers.
- **Strict Data Schemas**: Assures `pandas.DataFrame` schemas with explicit `pd.DatetimeIndex` alignments.

### 🧮 2. Mathematical Engines (`engine.py`)
- **Efficient Frontier Optimizations**: Maximum Sharpe, Minimum Volatility, Efficient Return, and Maximum Sortino strategies.
- **Ledoit-Wolf Shrinkage**: Guarantees optimization solver stability by applying shrinkage estimators to ill-conditioned Covariance Matrices.
- **Risk Assessment**: Parametric Value at Risk (VaR) and Conditional Value at Risk (CVaR).
- **Stochastic Simulations**: High-performance Geometric Brownian Motion paths utilizing purely parallel NumPy vectorization. No native Python `for` loops constraint allows `10,000+` path distributions resolving instantly.

### 💻 3. Interactive User Interface (`app.py`)
- **Decoupled Architecture**: Strict separation of mathematical logic and UI execution (Gradio `Blocks` and `Tabs`).
- **Dynamic Charting**: Integrated Plotly visualization for interactive Heatmaps and logarithmic Monte Carlo bounds overlaying real-time percentile metrics (1st, 5th, median, 95th, etc).
- **Statistical Charts**: Seaborn and Matplotlib outputs mapping Historical Monthly Returns Distributions out-of-the-box.

---

## 🛠 Installation & Setup

We strictly utilize [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package manager written in Rust, ensuring environment purity.

**1. Clone the repository**
```bash
git clone https://github.com/your-username/quantitative-portfolio-architect.git
cd quantitative-portfolio-architect
```

**2. Synchronize Dependencies with `uv`**
All dependencies are cleanly locked in `uv.lock`. 
```bash
uv sync
```
*(Packages include: `numpy`, `pandas`, `scipy`, `scikit-learn`, `cvxpy`, `pyportfolioopt`, `yfinance`, `gradio`, `plotly`, `pyarrow`, `matplotlib`, `seaborn`)*

---

## 🚀 Usage

Execute the graphical application locally across your predefined environment.

```bash
uv run app.py
```

The server natively launches. Navigate your browser to:
`http://127.0.0.1:7860/`

**Workflow Steps within the GUI:**
1. **Data & Setup**: Input comma-separated tickers (e.g., `AAPL, MSFT, GOOGL`) and adjust your temporal lookback horizon. Click *Fetch*. Wait for the Correlation Heatmap to render.
2. **Optimization**: Navigate to the second tab. Select an objective function (like `max_sharpe`). Let the PyPortfolioOpt solvers map optimal weights, highlighting Marginal Risk Contributions and Parametric VaR indices.
3. **Simulation**: Head to the Monte Carlo tab. Scale up your periods ahead and simulate `$S_t$` future geometric pathways overlaid with bounds.

---

## 🧪 Scientific Validation & Testing

Every complex algorithm inside the `engine.py` wrapper respects mathematical integrity with zero-tolerance for rounding deviations. The suite mandates strict tests auditing dimensionality matrices.

Execute the Pytest framework to validate structural integrities:
```bash
uv run pytest test_portfolio.py -v
```

### Mathematical Formulation
The codebase is structured around core econometric axioms. For instance, the simulated Asset Price dynamics govern:

$S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z \right)$

where $Z \sim \mathcal{N}(0,1)$.

---
*Created strictly decoupled ensuring that core optimization mechanics are instantly portable to API microservices without inheriting UI-blocking scopes.*
