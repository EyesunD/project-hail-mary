"""Performance analytics for bar-level signal backtests."""

from __future__ import annotations

import pandas as pd

from hailmary.analytics.metrics import PerformanceMetrics
from hailmary.backtest.signal_backtest import BarBacktestResult


class SignalAnalytics:
    """Compute performance statistics from a :class:`~hailmary.backtest.BarBacktestResult`.

    Provides both *per-symbol* views (``summary``, ``equity``) and a combined
    *portfolio* view that equal-weights all in-signal symbols at each bar
    (``portfolio_equity``, ``portfolio_returns``, ``portfolio_metrics``).

    Args:
        result: Output of :meth:`~hailmary.backtest.BarBacktest.run`.

    Example::

        signal_df = TrendSignal(200).run(bars)
        bt        = BarBacktest().run(signal_df)
        analytics = SignalAnalytics(bt)

        # Per-symbol
        analytics.summary()
        analytics.equity()

        # Combined portfolio
        analytics.portfolio_equity().plot()
        analytics.portfolio_metrics().summary()
    """

    def __init__(self, result: BarBacktestResult) -> None:
        self._data = result.data

    # ----------------------------------------------------------------- per-symbol

    def summary(self) -> pd.DataFrame:
        """Per-symbol aggregate: returns, trade count, and time invested.

        Returns:
            DataFrame indexed by symbol with columns ``return_mtc``,
            ``return_conservative``, ``entries``, ``exits``,
            ``invested_days``, ``pct_invested``.
        """
        total_bars = self._data.groupby(level="symbol").size()
        return self._data.groupby(level="symbol").agg(
            return_mtc=("strategy_equity_mark_to_close", lambda x: x.iloc[-1] - 1),
            return_conservative=("strategy_equity_conservative", lambda x: x.iloc[-1] - 1),
            return_net=("strategy_equity_net", lambda x: x.iloc[-1] - 1),
            entries=("enter", "sum"),
            exits=("exit", "sum"),
            invested_days=("cycle", lambda x: x.ne("None").sum()),
        ).assign(pct_invested=lambda df: df["invested_days"] / total_bars * 100)

    def equity(self, method: str = "mtc") -> pd.DataFrame:
        """Wide equity curve DataFrame (index=timestamp, columns=symbol).

        Args:
            method: ``"mtc"`` for mark-to-close (default) or
                    ``"conservative"`` for worst-fill.
        """
        col = {
            "mtc": "strategy_equity_mark_to_close",
            "conservative": "strategy_equity_conservative",
            "net": "strategy_equity_net",
        }[method]
        return self._data[col].unstack(level="symbol")

    # ----------------------------------------------------------------- portfolio

    def portfolio_equity(self, method: str = "mtc", reinvest: bool = False) -> pd.Series:
        """Equal-weight portfolio equity across all symbols (normalised to 1.0).

        Args:
            method: ``"mtc"`` (default), ``"conservative"``, or ``"net"``.
            reinvest: If ``False`` (default), rebalances back to ``1/N_universe``
                      each bar — positions that outperform are trimmed daily.
                      If ``True``, each symbol starts at ``1/N_universe`` and
                      compounds freely without rebalancing — equivalent to entering
                      each position with a fixed dollar amount and letting it run.

        Returns:
            pd.Series indexed by timestamp, normalised to 1.0 at the first bar.
        """
        if reinvest:
            return self._portfolio_equity_reinvested(method)
        return (1 + self._portfolio_returns_rebalanced(method)).cumprod().rename(
            f"portfolio_equity_{method}"
        )

    def portfolio_returns(self, method: str = "mtc", reinvest: bool = False) -> pd.Series:
        """Daily portfolio returns.

        Args:
            method: ``"mtc"`` (default), ``"conservative"``, or ``"net"``.
            reinvest: See :meth:`portfolio_equity`.

        Returns:
            pd.Series of daily returns indexed by timestamp.
        """
        if reinvest:
            return self._portfolio_equity_reinvested(method).pct_change().fillna(0)
        return self._portfolio_returns_rebalanced(method)

    def portfolio_metrics(
        self, method: str = "mtc", risk_free_rate: float = 0.0, reinvest: bool = False
    ) -> PerformanceMetrics:
        """Return a :class:`~hailmary.analytics.PerformanceMetrics` for the combined portfolio.

        Args:
            method: ``"mtc"`` (default), ``"conservative"``, or ``"net"``.
            risk_free_rate: Annualised risk-free rate (default 0).
            reinvest: See :meth:`portfolio_equity`.

        Returns:
            :class:`PerformanceMetrics` instance; call ``.summary()`` for a full table.
        """
        return PerformanceMetrics(
            self.portfolio_returns(method=method, reinvest=reinvest),
            risk_free_rate=risk_free_rate,
        )

    # ----------------------------------------------------------------- private

    def _portfolio_returns_rebalanced(self, method: str) -> pd.Series:
        """Daily returns assuming the portfolio rebalances to 1/N_universe each bar."""
        ret_col = {
            "mtc": "return_mark_to_close",
            "conservative": "return_conservative",
            "net": "return_net",
        }[method]
        in_trade = (self._data["cycle"] != "None").astype(int).unstack(level="symbol")
        returns_wide = self._data[ret_col].unstack(level="symbol")
        weights = in_trade / len(in_trade.columns)
        return (weights * returns_wide).sum(axis=1).rename(f"portfolio_returns_{method}")

    def _portfolio_equity_reinvested(self, method: str) -> pd.Series:
        """Portfolio equity when each position compounds freely from its 1/N_universe entry.

        Equivalent to the mean of the per-symbol equity curves: each symbol starts
        at 1/N and drifts with its own cumulative return.  No intra-period rebalancing.
        """
        equity_col = {
            "mtc": "strategy_equity_mark_to_close",
            "conservative": "strategy_equity_conservative",
            "net": "strategy_equity_net",
        }[method]
        equity_wide = self._data[equity_col].unstack(level="symbol")
        return equity_wide.mean(axis=1).rename(f"portfolio_equity_{method}_reinvested")
