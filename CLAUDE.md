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

A separate **allocation diagnostic path** (`hailmary.allocation`) ingests the user's
Stashaway book directly from `data/holding.xlsx` (sheet-driven, see `holdings_book.load_book`;
the legacy PDF parser in `statements.py` is kept for spot-checks but unwired from the main
flow). It role-tags portfolios via `book_config.ROLES`, reconstructs returns via
`DataProvider` proxies, and renders two HTML reports:

- **`reports/allocation_diagnostic.html`** (via `diagnostic.render_html_report`) — combined-book
  exposure, correlation/redundancy, risk contribution, benchmark deltas vs `MANAGED_BENCHMARK`
  sleeves, per-holding drilldown.
- **`reports/reconcile.html`** (via `reconcile.render_reconcile_report`) — per-sleeve
  trust-check vs the app. Multi-anchor BH+DR model anchored at each `PortfolioValue` snapshot,
  with mid-period deposits auto-injected at their date. Top table = sleeve totals (Gap =
  our_model − app_end); drilldown = per-holding × per-anchor timeline with 1M / cumulative
  deltas in native + SGD, plus BH-vs-Observed and BH-vs-DR drift rows.
- A `book_health_check()` runs at the top of every reconcile and flags integrity gaps
  (untagged portfolios, unknown tickers, weight-sum violations, FX staleness, mid-period
  deposits) as a visible banner.

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
├── allocation/     # Stashaway book ingestion + allocation diagnostic + reconcile
│   ├── universe.py        # STASHAWAY_UNIVERSE: Yahoo ticker → AssetMetadata
│   ├── holdings_book.py   # load_book(holding.xlsx, as_of) — main sheet-driven ingestion
│   ├── statements.py      # Legacy PDF parser — kept for spot-checks, unwired from main flow
│   ├── portfolios.py      # Portfolio, Holding, Role (HOLDING/MANAGED_BENCHMARK/PROTECTED/CUSTOM)
│   ├── returns.py         # portfolio_returns — daily-rebalance reconstruction (for diagnostic)
│   ├── diagnostic.py      # render_html_report + combined_exposure, correlation, redundancy, risk, benchmarks
│   ├── reconcile.py       # render_reconcile_report — multi-anchor BH+DR + deposit injection + book_health_check
│   ├── scenarios.py       # Scenario edit helpers + scenario_compare + render_scenario_report
│   ├── etf_explorer.py    # ETF universe browser (standalone)
│   └── book_config.py     # ROLES + MGMT_FEES_ANNUAL for the user's actual book
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
