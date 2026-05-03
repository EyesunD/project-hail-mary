# Project Hail Mary — Claude Guidance

## Project Overview

Python-based quantitative backtesting and analytics platform focused on time-series signal
strategies (currently MA-200 trend on crypto). The architecture has two backtest paths:

- **Bar-level path** (`TrendSignal → BarBacktest → SignalTradePerformance / SignalAllocationPerformance`):
  fast per-symbol signal simulation with fill-aware returns. Two evaluation lenses on the
  same `BarBacktestResult` — `SignalTradePerformance` treats each entry-to-exit cycle as one
  iid bet (no capital allocation), while `SignalAllocationPerformance` deploys capital and
  compounds it (per-symbol buy-and-hold + portfolio NAV under three capital models).
- **Engine path** (`TrendSignalStrategy → BacktestEngine → BacktestResult → PerformanceCharts`):
  full portfolio rebalancing with NAV, trade log, and tearsheet.

Both paths share the same `TrendSignal` signal columns. The cross-sectional factor path
(`MultiFactorModel → FactorPortfolio`) is kept for future multi-signal expansion.

## Package layout

```
src/hailmary/
├── data/           # Abstract market-data platform + providers
│   ├── base.py     # DataProvider ABC, Bar, FundamentalData, Timeframe
│   ├── cache.py    # Parquet disk cache
│   ├── registry.py # ProviderRegistry with priority-based fallback
│   └── providers/  # Yahoo, Alpaca, Polygon, CSV
├── models/         # Signal and factor model framework
│   ├── factors/    # Factor ABC + MovingAverageTrendFactor (cross-sectional)
│   ├── signals.py  # TrendSignal — generates signal columns from OHLCV bars
│   ├── multi_factor.py  # MultiFactorModel, ICStats
│   └── portfolio.py     # FactorPortfolio (quantile / score-weighted / optimised)
├── backtest/       # Backtesting engines
│   ├── engine.py        # BacktestEngine, Strategy ABC, BacktestResult
│   ├── strategies.py    # TrendSignalStrategy — wires TrendSignal into BacktestEngine
│   ├── signal_backtest.py # BarBacktest, BarBacktestResult — bar-level signal execution
│   ├── portfolio.py     # Portfolio, Position, Trade
│   └── execution.py     # ExecutionModel, SlippageModel, CommissionModel
├── analytics/      # Risk & performance analytics
│   ├── metrics.py        # PerformanceMetrics (Sharpe, Sortino, Calmar, VaR, …)
│   ├── signal_analytics.py # SignalTradePerformance (iid bets) + SignalAllocationPerformance (deployed capital)
│   ├── risk.py           # RiskAnalytics (covariance, risk contribution, factor decomp)
│   └── statistics.py     # FactorAnalytics (IC, quintile returns, decay)
├── viz/            # Plotly visualisations
│   ├── theme.py    # House theme (dark, high-contrast)
│   ├── performance_charts.py  # PerformanceCharts + tearsheet (for BacktestResult)
│   └── factor_charts.py       # FactorCharts + factor tearsheet
└── cli/            # Click CLI (hailmary fetch, clear-cache, info)
```

## End-to-end pipeline (bar-level)

```python
bars       = YahooFinanceProvider().get_bars(symbols, start=start, end=end)
signal_df  = TrendSignal(ma_window=200).run(bars)          # signal columns
bt_result  = BarBacktest().run(signal_df)                   # execution + returns

# Trade-level lens — each entry→exit cycle as one iid bet
trade_perf = SignalTradePerformance(bt_result)
trade_perf.trade_summary()                                  # win rate, expectancy, profit factor, …
trade_perf.trade_paths()                                    # per-trade cumulative-return paths

# Allocation-level lens — capital deployed and compounded
alloc      = SignalAllocationPerformance(bt_result)
alloc.summary()                                             # per-symbol returns, drawdown, % invested
alloc.portfolio_equity()                                    # equal-weight portfolio NAV
alloc.portfolio_metrics().summary()                         # Sharpe, drawdown, etc.
```

## End-to-end pipeline (engine / portfolio)

```python
bars      = YahooFinanceProvider().get_bars(symbols, start=start, end=end)
signal_df = TrendSignal(ma_window=200).run(bars)
close_df  = bars["close"].unstack("symbol")
engine    = BacktestEngine(prices=close_df, strategy=TrendSignalStrategy(signal_df),
                           rebalance_frequency="D")
result    = engine.run()                                   # BacktestResult
PerformanceCharts(result).tearsheet().show()
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
- **New signals**: extend `TrendSignal` or add a new signal class with `.run(bars) -> pd.DataFrame`. Signal classes produce signal columns only — no return/execution columns. Those are added by `BarBacktest`.
- **New strategies**: subclass `Strategy` in `backtest/strategies.py`, implement `generate_weights()`. Wire into `BacktestEngine`.

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
