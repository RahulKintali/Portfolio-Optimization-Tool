import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Fetch historical data
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):  # Handle multi-index columns
        data = data['Close']  # Extract only the "Close" prices
    st.write("Processed data columns:", data.columns)  # Debugging step
    returns = data.pct_change().dropna()
    return data, returns

# Portfolio metrics calculation
def portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio


# Portfolio optimization
def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # No shorting allowed

    result = minimize(lambda weights: -portfolio_metrics(weights, *args)[2],  # Maximize Sharpe Ratio
                      x0=[1 / num_assets] * num_assets,  # Initial weights
                      bounds=bounds, constraints=constraints)
    return result


# Plot efficient frontier
def plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate):
    results = []
    weights_list = []
    for _ in range(5000):  # Generate random portfolios
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_metrics(weights, mean_returns, cov_matrix,
                                                                                 risk_free_rate)
        results.append((portfolio_volatility, portfolio_return, sharpe_ratio))
        weights_list.append(weights)
    results = np.array(results)

    # Find optimal portfolio
    max_sharpe_idx = results[:, 2].argmax()
    max_sharpe = results[max_sharpe_idx]

    # Plot the efficient frontier
    fig, ax = plt.subplots()
    scatter = ax.scatter(results[:, 0], results[:, 1], c=results[:, 2], cmap='viridis', marker='o')
    ax.scatter(max_sharpe[0], max_sharpe[1], c='red', s=50, label='Max Sharpe Ratio')
    plt.colorbar(scatter, label="Sharpe Ratio")
    ax.set_xlabel('Volatility (Risk)')
    ax.set_ylabel('Return')
    ax.set_title('Efficient Frontier')
    ax.legend()
    st.pyplot(fig)

    return weights_list[max_sharpe_idx], max_sharpe


# Streamlit app
st.title("Portfolio Optimization Tool")
st.sidebar.header("Inputs")

# Input fields
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated):", "AAPL,MSFT,GOOG").split(',')
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
risk_free_rate = st.sidebar.slider("Risk-Free Rate (in %)", 0.0, 5.0, 2.0) / 100  # Convert to decimal

if st.sidebar.button("Optimize"):
    # Fetch data
    try:
        data, returns = fetch_data(tickers, start_date, end_date)
        mean_returns = returns.mean() * 252  # Annualized return
        cov_matrix = returns.cov() * 252  # Annualized covariance

        # Optimize portfolio
        optimal_result = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
        optimal_weights = optimal_result.x

        # Plot efficient frontier
        max_weights, max_sharpe = plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate)

        # Display results
        st.subheader("Optimal Portfolio Weights")
        for ticker, weight in zip(tickers, max_weights):
            st.write(f"{ticker}: {weight:.2%}")

        st.write(f"**Portfolio Return:** {max_sharpe[1]:.2f}")
        st.write(f"**Portfolio Volatility (Risk):** {max_sharpe[0]:.2f}")
        st.write(f"**Sharpe Ratio:** {max_sharpe[2]:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
