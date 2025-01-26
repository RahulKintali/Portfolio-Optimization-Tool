import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Function to fetch historical data
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna()
    return data, returns

# Function to calculate portfolio metrics
def portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Function to optimize the portfolio
def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # No shorting allowed

    result = minimize(lambda weights: -portfolio_metrics(weights, *args)[2],  # Maximize Sharpe Ratio
                      x0=[1 / num_assets] * num_assets,  # Initial weights
                      bounds=bounds, constraints=constraints)
    return result

# Function to calculate Beta and Alpha
def calculate_beta_alpha(asset_returns, benchmark_returns):
    X = sm.add_constant(benchmark_returns)
    model = sm.OLS(asset_returns, X).fit()
    beta = model.params[1]
    alpha = model.params[0]
    return beta, alpha

# Function to plot the efficient frontier
def plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate):
    results = []
    for _ in range(10000):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility, _ = portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate)
        results.append((portfolio_volatility, portfolio_return))
    results = np.array(results)

    plt.scatter(results[:, 0], results[:, 1], c=results[:, 1] / results[:, 0], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    plt.show()


def plot_efficient_frontier_with_optimal(mean_returns, cov_matrix, risk_free_rate, optimal_weights):
    results = []
    weights_list = []
    for _ in range(10000):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_metrics(weights, mean_returns, cov_matrix,
                                                                                 risk_free_rate)
        results.append((portfolio_volatility, portfolio_return, sharpe_ratio))
        weights_list.append(weights)
    results = np.array(results)

    # Find the portfolio with the max Sharpe Ratio
    max_sharpe_idx = results[:, 2].argmax()
    max_sharpe = results[max_sharpe_idx]
    max_weights = weights_list[max_sharpe_idx]

    plt.scatter(results[:, 0], results[:, 1], c=results[:, 2], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(max_sharpe[0], max_sharpe[1], c='red', s=50, label='Max Sharpe Ratio')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.show()
    return max_weights


# Main execution
# Main execution
if __name__ == "__main__":
    # Input data
    tickers = ['AAPL', 'MSFT', 'GOOG']  # Replace with your desired tickers
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    risk_free_rate = 0.02  # Example risk-free rate (2%)

    # Fetch data and calculate returns
    data, returns = fetch_data(tickers, start_date, end_date)
    mean_returns = returns.mean() * 252  # Annualized return
    cov_matrix = returns.cov() * 252  # Annualized covariance

    # Optimize portfolio
    optimized_result = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
    optimal_weights = optimized_result.x

    # Plot efficient frontier with optimal portfolio
    max_weights = plot_efficient_frontier_with_optimal(mean_returns, cov_matrix, risk_free_rate, optimal_weights)

    # Print results
    portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_metrics(max_weights, mean_returns, cov_matrix, risk_free_rate)
    print("Optimal Portfolio Weights:", max_weights)
    print(f"Return: {portfolio_return:.2f}")
    print(f"Volatility (Risk): {portfolio_volatility:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
