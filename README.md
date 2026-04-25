# Project Hail Mary

Quantitative backtesting and multi-factor analytics platform.

## Features

| Layer | What it does |
|---|---|
| **Data** | Abstract `DataProvider` with Yahoo Finance, Alpaca, Polygon, and CSV backends. Priority-based registry with automatic failover. Parquet disk cache. |
| **Factors** | Composable `Factor` ABC — Momentum (12-1, residual), Value (B/M, E/P, composite), Quality (ROE, margins, gross profitability), Volatility (realised, beta, idiosyncratic). |
| **Multi-factor model** | `MultiFactorModel` with equal, custom, or IC-weighted combination. IC analysis, signal decay curves. |
| **Backtest** | `BacktestEngine` + `Strategy` ABC. Portfolio with position tracking, configurable slippage + commission models, full trade log. |
| **Analytics** | `PerformanceMetrics` (Sharpe, Sortino, Calmar, VaR, CVaR, drawdown, capture). `RiskAnalytics` (Ledoit-Wolf / OAS covariance, risk contribution, factor variance decomposition). `FactorAnalytics` (IC, quintile returns, decay, turnover). |
| **Visualisation** | Interactive Plotly charts with a dark house theme. Full performance tearsheet. Factor tearsheet (IC series, decay, quintile returns). Monthly return heatmap. |
| **CLI** | `hailmary fetch / clear-cache / info` |

## Quick start

```bash
pip install -e ".[dev,notebooks]"
cp .env.example .env        # fill in API keys if using Alpaca / Polygon
jupyter lab notebooks/examples/
```

## Project structure

```
src/hailmary/
├── data/           # Market data abstraction layer
├── models/         # Factor models & portfolio construction
├── backtest/       # Backtesting engine
├── analytics/      # Risk & performance analytics
├── viz/            # Plotly visualisation suite
└── cli/            # Command-line interface

tests/              # Pytest test suite
notebooks/examples/ # Example Jupyter notebooks
configs/            # Strategy configuration templates
data/cache/         # Local data cache (git-ignored)
```

## Example

```python
from hailmary.data.providers import YahooFinanceProvider
from hailmary.models.factors import MomentumFactor, VolatilityFactor
from hailmary.models.multi_factor import MultiFactorModel
from hailmary.backtest.engine import BacktestEngine
from hailmary.viz.performance_charts import PerformanceCharts

yahoo = YahooFinanceProvider()
close = yahoo.get_bars(universe, "2019-01-01", "2024-01-01")["close"].unstack(0)

model = MultiFactorModel([MomentumFactor(), VolatilityFactor()])
engine = BacktestEngine(prices=close, strategy=my_strategy)
result = engine.run()

PerformanceCharts(result).tearsheet().show()
```

See `notebooks/examples/` for end-to-end walkthroughs.
