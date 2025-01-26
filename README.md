# Portfolio Optimization Tool

This is a Python-based tool that optimizes a portfolio of financial assets using **Modern Portfolio Theory (MPT)**. The tool leverages historical stock data to calculate optimal asset allocations, maximizing returns for a given level of risk. It also visualizes the **Efficient Frontier**, helping investors make data-driven decisions.

---

## **Features**
- Fetches historical data from **Yahoo Finance** or **Alpha Vantage**.
- Implements **mean-variance optimization** for portfolio allocation.
- Calculates key metrics:
  - **Expected Returns**
  - **Portfolio Risk**
  - **Sharpe Ratio**
  - **Beta** and **Alpha**
- Visualizes the **Efficient Frontier** using **Matplotlib** or **Plotly**.
- Provides a detailed summary of the optimal portfolio weights.

---

## **Tech Stack**
- **Python**: Core programming language.
- **Libraries**:
  - `numpy`: Numerical calculations.
  - `pandas`: Data manipulation.
  - `matplotlib`/`plotly`: Data visualization.
  - `yfinance` or `alpha_vantage`: Fetching historical data.
  - `scipy`: Optimization algorithms.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/portfolio-optimization-tool.git
   cd portfolio-optimization-tool
