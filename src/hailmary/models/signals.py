"""Time-series trend signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _signal_age(s: pd.Series) -> pd.Series:
    """Count bars into the current trade, including the exit bar.

    Returns 0 on flat bars and 1, 2, 3, … on init / held / exit bars.
    """
    active = s.fillna(0).eq(1) | ((s.fillna(0).shift() == 1) & (s.fillna(0) == 0))
    return (
        active.astype(int)
        .groupby((~active).cumsum())
        .cumsum()
        .astype("Int64")
    )


class TrendSignal:
    """MA-crossover trend signal.

    Annotates a (symbol, timestamp) MultiIndex bar DataFrame with signal
    columns ready for consumption by :class:`~hailmary.backtest.BarBacktest`.

    Long when ``close > MA(ma_window)``, flat otherwise.  The signal is
    observed after the close and shifted +1 day so positions open at the
    *next* bar's open — no look-ahead bias.

    Added columns:

    ================  ====================================================
    ``ma``            Rolling ``ma_window``-day average of close
    ``signal_close``  1 when ``close > MA`` (after-close, not tradeable)
    ``signal_open``   ``signal_close`` shifted +1 day (tradeable)
    ``trade_direction``  Daily change in ``signal_open`` (+1 / -1 / 0)
    ``turnover``      Absolute change in ``signal_open``
    ``enter``         1 on the first bar of each new long position
    ``exit``          1 on the last bar of each long position
    ``cycle``         ``init`` / ``held`` / ``exit`` / ``None``
    ``signal_age``    Bars elapsed since trade entry (incl. exit bar)
    ================  ====================================================

    Args:
        ma_window: Look-back window for the moving average (default 200).
    """

    def __init__(self, ma_window: int = 200) -> None:
        self.ma_window = ma_window

    @property
    def warmup(self) -> int:
        """Minimum number of bars needed before the MA is valid.

        Fetch at least this many bars *before* your intended backtest start so
        that ``signal_open`` is populated from day one::

            fetch_start = start - pd.offsets.BDay(signal.warmup)
            bars        = provider.get_bars(symbols, start=fetch_start, end=end)
            signal_df   = signal.run(bars, trim_start=start)
        """
        return self.ma_window

    def run(self, bars: pd.DataFrame, *, trim_start: pd.Timestamp | None = None) -> pd.DataFrame:
        """Annotate *bars* with signal columns and return the result.

        Args:
            bars: MultiIndex DataFrame from ``DataProvider.get_bars()`` with
                  at least a ``close`` column.
            trim_start: If provided, drop all rows whose timestamp is earlier
                than this value after computing the signal.  Use this together
                with a pre-fetched warm-up window so the backtest period starts
                with a fully-computed MA.

        Returns:
            Input DataFrame with signal columns appended (warm-up rows
            excluded when *trim_start* is set).
        """
        # Phase 1 — needs the full warmup window for MA and the cross-bar shift
        # that produces signal_open.
        base = (
            bars
            .assign(
                ma=lambda df: df.groupby(level="symbol")["close"].transform(
                    lambda s: s.rolling(self.ma_window).mean()
                ),
            )
            .assign(
                signal_close=lambda df: (df["close"] > df["ma"]).astype(int),
            )
            .assign(
                signal_open=lambda df: df.groupby(level="symbol")["signal_close"].transform(
                    lambda s: s.shift(1).fillna(0).astype(int)
                ),
            )
        )

        if trim_start is not None:
            base = base[base.index.get_level_values("timestamp") >= trim_start]

        # Phase 2 — all columns derived from signal_open.  Computing these on
        # the trimmed DataFrame means shift(1).fillna(0) treats the first bar as
        # the start of the world: a position already on at trim_start gets
        # enter=1 / trade_direction=+1 / turnover=1 / signal_age=1, consistent
        # with BarBacktest using open as the reference price for that bar.
        return (
            base
            .assign(
                trade_direction=lambda df: df.groupby(level="symbol")["signal_open"].transform(
                    lambda s: s.diff().fillna(s)
                ),
                turnover=lambda df: df.groupby(level="symbol")["signal_open"].transform(
                    lambda s: s.diff().abs().fillna(s.abs())
                ),
                enter=lambda df: df.groupby(level="symbol")["signal_open"].transform(
                    lambda s: np.where(~s.isin([0]) & s.shift(1).fillna(0).isin([0]), 1, 0)
                ),
                exit=lambda df: df.groupby(level="symbol")["signal_open"].transform(
                    lambda s: np.where(s.isin([0]) & ~s.shift(1).fillna(0).isin([0]), 1, 0)
                ),
            )
            .assign(
                cycle=lambda df: np.select(
                    [df["enter"].eq(1), df["exit"].eq(1), df["signal_open"].eq(1)],
                    ["init", "exit", "held"],
                    default="None",
                ),
            )
            .assign(
                signal_age=lambda df: df.groupby(level="symbol")["signal_open"].transform(
                    _signal_age
                ),
            )
        )
