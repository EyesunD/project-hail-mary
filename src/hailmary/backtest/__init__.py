"""Vectorised backtesting engine."""

from hailmary.backtest.engine import BacktestEngine, BacktestResult
from hailmary.backtest.execution import ExecutionModel, SlippageModel
from hailmary.backtest.portfolio import Portfolio
from hailmary.backtest.signal_backtest import BarBacktest, BarBacktestResult
from hailmary.backtest.strategies import TrendSignalStrategy

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BarBacktest",
    "BarBacktestResult",
    "ExecutionModel",
    "Portfolio",
    "SlippageModel",
    "TrendSignalStrategy",
]
