"""Bar-level backtest for binary trend signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _new_trade_boundary(x: pd.Series) -> pd.Series:
    """True on the first bar of each new trade cycle or flat period."""
    sign_change_exit = np.logical_and(
        np.sign(x) != np.sign(x.shift(1)),
        ~x.shift(1).ffill().isin([0]),
    )
    flat_before_entry: pd.Series = x.isin([0]) & ~x.shift(-1).ffill().isin([0])
    return sign_change_exit | flat_before_entry


def _apply_entry_offset(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Delay each trade's entry by *k* bars; drop trades that don't survive.

    For each contiguous in-trade run (consecutive bars with ``cycle != "None"``):

    - Run length ``≤ k + 1``: reset entirely to flat (the delayed entry would
      coincide with or follow the original exit).
    - Otherwise: bars 0..k-1 of the run become flat, bar k becomes the new
      entry (``cycle="init"``, ``enter=1``, ``signal_open=0``); the original
      held / exit bars are left untouched.
    """
    if k <= 0:
        return df

    out = df.copy()
    sym_idx = out.index.get_level_values("symbol")
    in_trade = out["cycle"] != "None"

    # A new trade always starts at an `enter==1` bar.  Using `cycle != "None"`
    # transitions would merge back-to-back trades (one trade's exit on day N
    # followed by another's init on day N+1 with no flat bar between) into a
    # single run, which would let only the first one carry the offset shift.
    new_run = out["enter"] == 1
    run_id = new_run.astype(int).groupby(sym_idx).cumsum()
    # Pool flat bars under run_id=0 so they don't merge with adjacent trades.
    run_id_eff = run_id.where(in_trade, 0)

    grp = out.groupby([sym_idx, run_id_eff])
    pos = grp.cumcount()
    run_len = grp["cycle"].transform("count")

    surviving = in_trade & (run_len > k + 1)
    flat_rows = in_trade & (
        (run_len <= k + 1)         # short trade — drop entirely
        | (surviving & (pos < k))   # first k bars of survivors
    )
    new_init = surviving & (pos == k)

    out.loc[flat_rows, "cycle"] = "None"
    out.loc[flat_rows, "enter"] = 0
    out.loc[flat_rows, "exit"] = 0
    out.loc[flat_rows, "signal_open"] = 0

    out.loc[new_init, "cycle"] = "init"
    out.loc[new_init, "enter"] = 1
    out.loc[new_init, "signal_open"] = 0

    return out


@dataclass
class BarBacktestResult:
    """Fully annotated bar DataFrame produced by :class:`BarBacktest`.

    The ``data`` DataFrame contains all columns from :class:`~hailmary.models.TrendSignal`
    plus the execution columns added by this backtest:

    ================================  ==========================================
    ``return_mark_to_close``          Fill-aware daily return (best-case timing)
    ``return_conservative``           Fill-aware daily return (worst-case timing)
    ``return_net``                    MTC return minus round-trip cost on entry/exit
    ``position_start``                1 if capital is deployed at bar open
    ``position_end``                  1 if capital is deployed at bar close
    ``strategy_equity_mark_to_close`` Cumulative product of MTC returns
    ``strategy_equity_conservative``  Cumulative product of conservative returns
    ``strategy_equity_net``           Cumulative product of net returns
    ``trade_cycle_id``                Incrementing ID per trade or flat period
    ================================  ==========================================
    """

    data: pd.DataFrame


class BarBacktest:
    """Simulate execution of a binary signal on OHLCV bars.

    Consumes the annotated DataFrame from :meth:`~hailmary.models.TrendSignal.run`
    and adds fill-aware return attribution and position tracking.

    Return accounting by cycle:

    ========  ===================  ========================  ==========================
    Cycle     Mark-to-close        Conservative (worst fill) Net (MTC minus cost)
    ========  ===================  ========================  ==========================
    init      open → close         high → close              open → close minus cost
    held      prev-close → close   prev-close → close        prev-close → close
    exit      prev-close → open    prev-close → low          prev-close → open minus cost
    ========  ===================  ========================  ==========================

    Args:
        cost_bps: One-way transaction cost in basis points applied on entry and exit bars.
                  ``return_net`` equals ``return_mark_to_close`` when this is 0 (default).

    Example::

        signal_df = TrendSignal(200).run(bars)
        result    = BarBacktest(cost_bps=5).run(signal_df)
    """

    def __init__(self, cost_bps: float = 0.0) -> None:
        self.cost_bps = cost_bps

    def run(
        self, signal_df: pd.DataFrame, entry_offset: int = 0
    ) -> BarBacktestResult:
        """Run the execution simulation on a signal-annotated bar DataFrame.

        Args:
            signal_df: Output of :meth:`~hailmary.models.TrendSignal.run`.
            entry_offset: Delay each trade's entry by this many bars — only
                take the trade if the signal is still on after the offset.
                Trades that don't survive the offset (run length ≤
                ``entry_offset + 1`` bars) are dropped entirely.  Default
                ``0`` enters on the original signal flip.

        Returns:
            :class:`BarBacktestResult` with the fully annotated DataFrame.
        """
        if entry_offset > 0:
            signal_df = _apply_entry_offset(signal_df, entry_offset)
        return BarBacktestResult(data=self._build(signal_df))

    # ------------------------------------------------------------------ private

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_close = df.groupby(level="symbol")["close"].shift(1)
        cost = self.cost_bps / 10_000

        return (
            df
            .assign(
                return_mark_to_close=lambda d: np.select(
                    [d["enter"].eq(1), d["exit"].eq(1), d["signal_open"].eq(1)],
                    [
                        d["close"] / d["open"] - 1,
                        d["open"] / prev_close - 1,
                        d["close"] / prev_close - 1,
                    ],
                    default=0.0,
                ),
                return_conservative=lambda d: np.select(
                    [d["enter"].eq(1), d["exit"].eq(1), d["signal_open"].eq(1)],
                    [
                        d["close"] / d["high"] - 1,
                        d["low"] / prev_close - 1,
                        d["close"] / prev_close - 1,
                    ],
                    default=0.0,
                ),
            )
            .assign(
                return_net=lambda d: d["return_mark_to_close"] - np.where(
                    d["enter"].eq(1) | d["exit"].eq(1), cost, 0.0
                ),
                position_start=lambda d: np.where(d["cycle"].isin(["exit", "held"]), 1, 0),
                position_end=lambda d: np.where(d["cycle"].isin(["init", "held"]), 1, 0),
                strategy_equity_mark_to_close=lambda d: (
                    (1 + d["return_mark_to_close"]).groupby(level="symbol").cumprod()
                ),
                strategy_equity_conservative=lambda d: (
                    (1 + d["return_conservative"]).groupby(level="symbol").cumprod()
                ),
                strategy_equity_net=lambda d: (
                    (1 + d["return_net"]).groupby(level="symbol").cumprod()
                ),
            )
            .assign(
                trade_cycle_id=lambda d: d.groupby(level="symbol")["position_start"].transform(
                    lambda s: np.where(_new_trade_boundary(s), 1, 0).cumsum()
                ),
            )
        )
