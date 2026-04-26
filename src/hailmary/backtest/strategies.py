"""Concrete Strategy implementations for use with BacktestEngine."""

from __future__ import annotations

import pandas as pd

from hailmary.backtest.engine import Strategy
from hailmary.backtest.portfolio import Portfolio


class TrendSignalStrategy(Strategy):
    """Equal-weight the symbols where ``signal_open = 1`` at each rebalance bar.

    Consumes the DataFrame produced by :meth:`~hailmary.models.TrendSignal.run`
    and exposes it as a :class:`~hailmary.backtest.engine.Strategy` so it can
    be plugged into :class:`~hailmary.backtest.engine.BacktestEngine` for
    portfolio-level NAV, Sharpe, tearsheet, etc.

    Usage::

        bars      = yahoo.get_bars(symbols, start=start, end=end)
        signal_df = TrendSignal(ma_window=200).run(bars)

        engine = BacktestEngine(
            TrendSignalStrategy(signal_df),
            bars=bars,
        )
        result = engine.run()
        PerformanceCharts(result).tearsheet().show()
    """

    def __init__(self, signal_df: pd.DataFrame) -> None:
        # Pre-unstack to a wide (timestamp × symbol) boolean matrix for O(1) lookups.
        self._signals: pd.DataFrame = signal_df["signal_open"].unstack(level="symbol")

    def generate_weights(
        self,
        prices: pd.DataFrame,
        timestamp: pd.Timestamp,
        portfolio: Portfolio,
        **context: object,
    ) -> pd.Series:
        """Return fixed-fractional weights for all symbols where ``signal_open = 1``.

        Each in-signal symbol receives ``1 / N_universe`` regardless of how many
        other symbols are active.  Uninvested capital sits in cash — a symbol
        dropping out of signal never increases the weight of the remaining ones.

        If the exact *timestamp* is not in the signal index (e.g. weekend, holiday),
        falls back to the most recent prior bar.  Returns an empty Series when no
        symbols are in signal so the engine holds cash.
        """
        if timestamp in self._signals.index:
            row = self._signals.loc[timestamp]
        else:
            avail = self._signals.index[self._signals.index <= timestamp]
            if avail.empty:
                return pd.Series(dtype=float)
            row = self._signals.loc[avail[-1]]

        on = row[row == 1].index
        if on.empty:
            return pd.Series(dtype=float)
        weight = 1.0 / len(self._signals.columns)
        return pd.Series(weight, index=on, name="weight")
