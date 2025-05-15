# Trading Strategy

## Overview
Trading Strategy is a Python package designed to implement and evaluate real-world investment strategies. It computes portfolio signals based on asset prices and a predefined investment universe, providing a structured framework for systematic trading. The package includes a suite of financial tools for portfolio construction, performance measurement, backtesting, and data visualization. By integrating asset weighting techniques, risk metrics, and benchmarking, it helps investors analyze their strategies efficiently.

## Features
- **Market Data Retrieval**: Fetches asset price data from Yahoo Finance.
- **Financial Metrics**: Computes returns, volatility, Sharpe ratio, Sortino ratio, Value at Risk, and more.
- **Portfolio Construction**: Implements asset weighting and rebalancing techniques.
- **Backtesting**: Simulates portfolio performance using historical data.
- **Strategy Evaluation**: Provides benchmark comparisons and macroeconomic overlays.

## Installation
Ensure Python (>=3.6) is installed. Clone this repository:

```sh
git clone https://github.com/BuzLclair/Trading_strategy.git
cd Trading_strategy
pip install numpy pandas seaborn matplotlib yfinance statsmodels tqdm
```

## Usage
### Run Strategy Analysis:
Execute the following script to compute asset returns and benchmark portfolio performance:

``` sh
python strategy_results.py
```
