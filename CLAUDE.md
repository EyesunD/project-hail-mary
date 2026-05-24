# Project Hail Mary — Claude Guidance

## Project Overview

Python-based quantitative backtesting and analytics platform focused on time-series signal
strategies (currently MA-200 trend on crypto).

The active development path is the **bar-level path**
(`TrendSignal → BarBacktest → SignalTradePerformance / SignalAllocationPerformance`):
fast per-symbol signal simulation with fill-aware returns, evaluated through two lenses on
the same `BarBacktestResult` — `SignalTradePerformance` treats each entry-to-exit cycle as
one iid bet (no capital allocation), while `SignalAllocationPerformance` deploys capital
and compounds it (per-symbol buy-and-hold + portfolio NAV under three capital models).

A full portfolio-rebalancing engine path (`TrendSignalStrategy → BacktestEngine →
BacktestResult → PerformanceCharts`) also exists in `hailmary.backtest` but is not
currently exercised by the example notebooks — leave it alone for now; we'll come back to
it once the bar-level surface is settled. The cross-sectional factor path
(`MultiFactorModel → FactorPortfolio`) is similarly retained for future multi-signal
expansion.

A separate **allocation diagnostic path** (`hailmary.allocation`) ingests real Stashaway
PDF statements, role-tags portfolios, reconstructs returns via proxies through the
existing `DataProvider` registry, and renders a self-contained HTML report covering
combined-book exposure, correlation/redundancy, risk contribution, and Sharpe deltas vs
`MANAGED_BENCHMARK`-tagged portfolios. Phase 1 only; FX conversion and longer-horizon
Sharpe re-runs are deferred.

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
├── allocation/     # Stashaway book ingestion + allocation diagnostic (Phase 1)
│   ├── universe.py     # STASHAWAY_UNIVERSE: Stashaway asset → tradeable ticker + metadata
│   ├── statements.py   # PDF parser (pdfplumber) + JSON fallback + parquet cache
│   ├── portfolios.py   # Portfolio, Holding, Role (CUSTOM/MANAGED_BENCHMARK/PROTECTED/HOLDING)
│   ├── returns.py      # portfolio_returns — reconstruct series from current weights
│   ├── diagnostic.py   # combined_exposure, correlation_matrix, redundancy_pairs, risk_contribution, benchmark_comparison, render_html_report
│   └── book_config.py  # Centralised role-tag mapping for the user's actual book
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
- **Engine / strategy work is paused.** `backtest/engine.py`, `backtest/strategies.py`, and the related portfolio/execution code stay in the repo and the tests still run, but no notebooks exercise them right now. Don't add new strategies or extend the engine path until the user explicitly comes back to it.

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
pip install -e ".[allocation]"           # Stashaway PDF parsing + HTML report (pdfplumber, jinja2)
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
