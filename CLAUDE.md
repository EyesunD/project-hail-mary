# Project Hail Mary — Claude Guidance

## Project Overview

Python-based quantitative backtesting and analytics platform. The user does quantitative analysis on multi-factor models and needs beautiful, interactive visualisations of backtest results.

## Package layout

```
src/hailmary/
├── data/           # Abstract market-data platform + providers
│   ├── base.py     # DataProvider ABC, Bar, FundamentalData, Timeframe
│   ├── cache.py    # Parquet disk cache
│   ├── registry.py # ProviderRegistry with priority-based fallback
│   └── providers/  # Yahoo, Alpaca, Polygon, CSV
├── models/         # Multi-factor model framework
│   ├── factors/    # Base Factor ABC + Momentum, Value, Quality, Volatility
│   ├── multi_factor.py  # MultiFactorModel, ICStats
│   └── portfolio.py     # FactorPortfolio (quantile / score-weighted / optimised)
├── backtest/       # Backtesting engine
│   ├── engine.py   # BacktestEngine, Strategy ABC, BacktestResult
│   ├── portfolio.py # Portfolio, Position, Trade
│   └── execution.py # ExecutionModel, SlippageModel, CommissionModel
├── analytics/      # Risk & performance analytics
│   ├── metrics.py  # PerformanceMetrics (Sharpe, Sortino, Calmar, VaR, …)
│   ├── risk.py     # RiskAnalytics (covariance, risk contribution, factor decomp)
│   └── statistics.py # FactorAnalytics (IC, quintile returns, decay)
├── viz/            # Plotly visualisations
│   ├── theme.py    # House theme (dark, high-contrast)
│   ├── performance_charts.py  # PerformanceCharts + tearsheet
│   └── factor_charts.py       # FactorCharts + factor tearsheet
└── cli/            # Click CLI (hailmary fetch, clear-cache, info)
```

## Conventions

- **Python 3.11+**, typed with mypy strict.
- **Formatting**: ruff (line length 100). Run `ruff check . && ruff format .` before committing.
- **Tests**: pytest with synthetic fixtures in `tests/conftest.py`. No real network calls in tests — mock providers or use CSVProvider with fixture data.
- **Data layer**: always accept `list[str]` for symbols; return MultiIndex DataFrames `(symbol, timestamp)` for bar data, wide DataFrames `(date, symbol)` for prices/returns.
- **No comments** that explain what the code does. Comments only for non-obvious *why*.
- **Visualisations**: always use `hailmary.viz.theme.apply_theme()` so all charts share the dark house style. Return `go.Figure` objects — never call `.show()` inside library code.
- **New providers**: subclass `DataProvider`, implement `get_bars` and `get_latest_bars`, add optional import guard, register in `providers/__init__.py`.
- **New factors**: subclass `Factor`, set `name` and `description`, implement `compute()` returning a `FactorScore`.

## Available skills

- `/init` — regenerate this CLAUDE.md if the project structure changes significantly.
- `/review` — review a pull request.
- `/security-review` — security review of pending branch changes.
- `/simplify` — review changed code for quality and simplify where possible.

## Install

```bash
pip install -e ".[dev,notebooks]"         # core + dev tools + Jupyter
pip install -e ".[alpaca]"               # Alpaca provider
pip install -e ".[polygon]"              # Polygon provider
pip install -e ".[dash]"                 # Dash interactive dashboard
```

## Running tests

```bash
pytest                    # all tests with coverage
pytest tests/models/ -v   # specific module
```

## CLI

```bash
hailmary fetch AAPL MSFT --start 2022-01-01
hailmary clear-cache
hailmary info
```
