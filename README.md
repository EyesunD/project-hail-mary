# Project Hail Mary

Quantitative backtesting and analytics platform built around time-series signal strategies.
Currently implements a moving-average trend signal on crypto assets, with the architecture
designed to support multiple signals as the system grows.

## Features

| Layer | What it does |
|---|---|
| **Data** | Abstract `DataProvider` with Yahoo Finance, Alpaca, Polygon, and CSV backends. Priority-based registry with automatic failover. Parquet disk cache. |
| **Signals** | `TrendSignal` — generates look-ahead-safe signal columns (`ma`, `signal_open`, `enter`, `exit`, `cycle`, `signal_age`) from OHLCV bar data. |
| **Backtest** | Two paths: `BarBacktest` for fast per-symbol simulation; `BacktestEngine` + `TrendSignalStrategy` for full portfolio rebalancing with NAV and trade log. Fill-aware returns in both mark-to-close and conservative conventions. |
| **Analytics** | `SignalTradePerformance` (each trade as one iid bet — win rate, expectancy, distribution, paths) and `SignalAllocationPerformance` (deployed-capital lens — per-symbol buy-and-hold + portfolio NAV under three capital models). `PerformanceMetrics` (Sharpe, Sortino, Calmar, VaR, CVaR, drawdown). `RiskAnalytics` (covariance, risk contribution, factor decomposition). |
| **Factor model** | `MovingAverageTrendFactor` + `MultiFactorModel` for cross-sectional scoring. `FactorPortfolio` (quantile / score-weighted / mean-variance optimised). |
| **Visualisation** | Interactive Plotly charts with a dark house theme. Performance tearsheet, equity curves, drawdown, monthly return heatmap, factor charts. |
| **CLI** | `hailmary fetch / clear-cache / info` |

## Install

```bash
pip install -e ".[dev,notebooks]"
jupyter lab notebooks/examples/
```

Optional extras:

```bash
pip install -e ".[alpaca]"    # Alpaca Markets provider
pip install -e ".[polygon]"   # Polygon.io provider
pip install -e ".[dash]"      # Dash interactive dashboard
```

## Quick start — bar-level backtest

```python
import pandas as pd
from hailmary.data.providers import YahooFinanceProvider
from hailmary.models import TrendSignal
from hailmary.backtest.signal_backtest import BarBacktest
from hailmary.analytics.signal_analytics import (
    SignalAllocationPerformance,
    SignalTradePerformance,
)

signal = TrendSignal(ma_window=200)
yahoo  = YahooFinanceProvider()

start, end = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")

# Fetch warmup bars so MA-200 is fully computed from day one of the backtest
fetch_start = start - pd.offsets.BDay(signal.warmup)
bars = yahoo.get_bars(["BTC-USD", "ETH-USD", "SOL-USD"], start=fetch_start, end=end)

signal_df = signal.run(bars, trim_start=start)
bt_result = BarBacktest().run(signal_df)

# Trade-level lens — each entry→exit cycle as one iid bet
SignalTradePerformance(bt_result).trade_summary()  # win rate, expectancy, profit factor, …

# Allocation-level lens — capital deployed and compounded
alloc = SignalAllocationPerformance(bt_result)
alloc.summary()                      # per-symbol returns, entries, % invested
alloc.portfolio_equity().plot()      # equal-weight portfolio NAV
alloc.portfolio_metrics().summary()  # Sharpe, max drawdown, CAGR, …
```

## Quick start — full portfolio engine

```python
from hailmary.backtest import BacktestEngine, TrendSignalStrategy
from hailmary.viz.performance_charts import PerformanceCharts

# signal_df and bars from above (with warmup fetch)
close_df = bars["close"].unstack("symbol").loc[start:]

engine = BacktestEngine(
    prices=close_df,
    strategy=TrendSignalStrategy(signal_df),
    rebalance_frequency="D",
)
result = engine.run()
PerformanceCharts(result).tearsheet().show()
```

## Notebooks

| Notebook | What it covers |
|---|---|
| `01_data_providers.ipynb` | Fetching OHLCV data via `YahooFinanceProvider`, cache, multi-symbol |
| `04_ma200_trend_signal.ipynb` | `TrendSignal` — signal columns, look-ahead safety, entry/exit visualisation |
| `05_bar_backtest_analytics.ipynb` | `BarBacktest` + `SignalAllocationPerformance` — returns, equity curves, drawdown, trade duration |

## Project structure

```
src/hailmary/
├── data/              # Market data abstraction (providers, cache, registry)
├── models/            # TrendSignal, MovingAverageTrendFactor, MultiFactorModel, FactorPortfolio
├── backtest/          # BarBacktest, TrendSignalStrategy, BacktestEngine, Portfolio, ExecutionModel
├── analytics/         # SignalTradePerformance, SignalAllocationPerformance, PerformanceMetrics, RiskAnalytics, FactorAnalytics
├── viz/               # Plotly charts (dark theme, tearsheet, factor charts)
└── cli/               # hailmary fetch / clear-cache / info

tests/                 # pytest suite — no real network calls, synthetic fixtures
notebooks/examples/    # End-to-end example notebooks
```

## Running tests

```bash
pytest                       # all tests with coverage
pytest tests/backtest/ -v    # specific module
```
