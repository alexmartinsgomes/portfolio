import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from data_provider import fetch_data
from engine import (
    normalize_weights, 
    optimize_portfolio, 
    calculate_var_cvar, 
    run_monte_carlo_gbm, 
    calculate_marginal_contribution,
    get_returns_and_cov
)

def load_data(tickers_str, start_date, end_date):
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        return None, "Please enter at least one ticker."
    
    try:
        df = fetch_data(tickers, start_date, end_date)
        if df.empty:
            return None, "No data fetched. Check tickers and dates."
        
        # Create a preview table
        preview_df = df.tail(5).reset_index()
        preview_df['Date'] = preview_df['Date'].dt.strftime('%Y-%m-%d')
        # Round the numeric columns to 2 decimal places for cleaner UI
        numeric_cols = preview_df.select_dtypes(include=['float64', 'float32']).columns
        preview_df[numeric_cols] = preview_df[numeric_cols].round(2)
        
        msg = f"Successfully loaded data for {len(df.columns)} assets. {len(df)} trading days."
        return df, msg, preview_df
    except Exception as e:
        return None, f"Error loading data: {str(e)}", pd.DataFrame()

def run_optimization(df, strategy, target_return_str):
    if df is None or df.empty:
        return None, "No data loaded.", None, None
        
    try:
        target_return = None
        if strategy == "efficient_return" and target_return_str:
            target_return = float(target_return_str) / 100.0
            
        weights = optimize_portfolio(df, strategy, target_return)
        
        # Format weights for display
        w_df = pd.DataFrame({
            'Asset': weights.index,
            'Weight (%)': (weights.values * 100).round(2)
        })
        w_df = w_df[w_df['Weight (%)'] > 0.01].sort_values('Weight (%)', ascending=False)
        
        # Calculate Risk Metrics
        var, cvar = calculate_var_cvar(weights, df)
        risk_msg = f"Parametric VaR (95%): {var*100:.2f}%\nParametric CVaR (95%): {cvar*100:.2f}%"
        
        # Calculate MCR
        mcr = calculate_marginal_contribution(weights, df)
        mcr_df = pd.DataFrame({
            'Asset': mcr.index,
            'Marginal Contribution to Risk': mcr.values.round(4)
        })
        mcr_df = mcr_df.set_index('Asset').loc[w_df['Asset']].reset_index() # Align with non-zero weights
        
        return weights, w_df, risk_msg, mcr_df
        
    except Exception as e:
        return None, pd.DataFrame({"Error": [str(e)]}), "Optimization failed.", pd.DataFrame()

def plot_correlation(df):
    if df is None or df.empty:
        return None
    
    corr = df.pct_change().corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    fig.update_layout(title="Asset Correlation Heatmap", height=400)
    return fig

def plot_monte_carlo(df, weights, days_ahead, n_simulations, use_log_scale):
    if df is None or df.empty or weights is None:
        return None
        
    try:
        simulations = run_monte_carlo_gbm(weights, df, int(days_ahead), int(n_simulations))
        
        fig = go.Figure()
        
        # Plot a subset of paths to avoid crashing the browser (e.g. max 100 lines)
        paths_to_plot = min(100, int(n_simulations))
        for i in range(paths_to_plot):
            fig.add_trace(go.Scatter(
                x=simulations.index, 
                y=simulations.iloc[:, i],
                mode='lines',
                line=dict(width=1, color='rgba(0, 100, 255, 0.05)'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
        # Plot percentiles
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        colors = ['red', 'orange', 'lightgreen', 'green', 'lightgreen', 'orange', 'red']
        
        for p, color in zip(percentiles, colors):
            p_values = simulations.apply(lambda x: np.percentile(x, p), axis=1)
            fig.add_trace(go.Scatter(
                x=simulations.index,
                y=p_values,
                mode='lines',
                name=f'{p}th Percentile',
                line=dict(width=2, color=color, dash='dash' if p != 50 else 'solid')
            ))
            
        fig.update_layout(
            title=f"Monte Carlo Simulation ({n_simulations} paths, {days_ahead} days)",
            xaxis_title="Days Ahead",
            yaxis_title="Portfolio Value (Normalized to 1.0)",
            yaxis_type="log" if use_log_scale else "linear",
            height=600
        )
        return fig
    except Exception as e:
        print(f"MC Error: {e}")
        return None

def plot_boxplot(df):
    if df is None or df.empty:
        return None
        
    monthly_returns = df.resample('ME').last().pct_change().dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=monthly_returns, ax=ax, palette="Set3")
    ax.set_title("Monthly Returns Distribution")
    ax.set_ylabel("Return")
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

# --- Gradio UI Definition ---

with gr.Blocks(title="Quantitative Equity Portfolio Architect", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Quantitative Equity Portfolio Architect")
    
    # State variables
    stored_data = gr.State(None)
    current_weights = gr.State(None)
    
    with gr.Tabs():
        
        # --- Data / Setup Tab ---
        with gr.Tab("1. Data & Setup"):
            with gr.Row():
                with gr.Column():
                    tickers_input = gr.Textbox(label="Tickers (comma separated)", value="AAPL, MSFT, GOOGL, SPY, TLT")
                    start_date_input = gr.Textbox(label="Start Date (YYYY-MM-DD)", value=(datetime.today() - timedelta(days=365*3)).strftime('%Y-%m-%d'))
                    end_date_input = gr.Textbox(label="End Date (YYYY-MM-DD)", value=datetime.today().strftime('%Y-%m-%d'))
                    load_btn = gr.Button("Fetch / Load Data", variant="primary")
                    
                with gr.Column():
                    status_text = gr.Textbox(label="Status", interactive=False)
                    data_preview = gr.Dataframe(label="Data Preview (Last 5 Days)")
                    
            with gr.Row():
                corr_plot = gr.Plot(label="Correlation Heatmap")
                
            load_btn.click(
                fn=load_data,
                inputs=[tickers_input, start_date_input, end_date_input],
                outputs=[stored_data, status_text, data_preview]
            ).then(
                fn=plot_correlation,
                inputs=[stored_data],
                outputs=[corr_plot]
            )
            
        # --- Optimization Tab ---
        with gr.Tab("2. Optimization & Risk"):
            with gr.Row():
                with gr.Column():
                    strategy_dropdown = gr.Dropdown(
                        choices=["max_sharpe", "min_volatility", "efficient_return", "max_sortino"],
                        value="max_sharpe",
                        label="Optimization Strategy"
                    )
                    target_return_input = gr.Textbox(
                        label="Target Return % (for efficient_return only)", 
                        visible=False
                    )
                    
                    def toggle_target(choice):
                        return gr.update(visible=choice == "efficient_return")
                        
                    strategy_dropdown.change(fn=toggle_target, inputs=strategy_dropdown, outputs=target_return_input)
                    
                    opt_btn = gr.Button("Run Optimization", variant="primary")
                    
                with gr.Column():
                    weights_table = gr.Dataframe(label="Optimal Weights")
                    risk_metrics = gr.Textbox(label="Risk Metrics (VaR / CVaR)", interactive=False, lines=2)
                    mcr_table = gr.Dataframe(label="Marginal Contribution to Risk")
            
            with gr.Row():
                box_plot = gr.Plot(label="Monthly Returns Boxplot")
                
            opt_btn.click(
                fn=run_optimization,
                inputs=[stored_data, strategy_dropdown, target_return_input],
                outputs=[current_weights, weights_table, risk_metrics, mcr_table]
            ).then(
                fn=plot_boxplot,
                inputs=[stored_data],
                outputs=[box_plot]
            )
            
        # --- Simulation Tab ---
        with gr.Tab("3. Monte Carlo Simulation"):
            with gr.Row():
                with gr.Column(scale=1):
                    days_ahead = gr.Number(label="Days Ahead", value=252, precision=0)
                    n_sims = gr.Number(label="Number of Simulations", value=10000, precision=0)
                    log_scale = gr.Checkbox(label="Logarithmic Y-Axis", value=False)
                    sim_btn = gr.Button("Run Simulation", variant="primary")
                    sim_warning = gr.Markdown("*Note: Requires an optimized portfolio from Tab 2.*")
                    
                with gr.Column(scale=3):
                    mc_plot = gr.Plot(label="Geometric Brownian Motion Paths")
                    
            sim_btn.click(
                fn=plot_monte_carlo,
                inputs=[stored_data, current_weights, days_ahead, n_sims, log_scale],
                outputs=[mc_plot]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
