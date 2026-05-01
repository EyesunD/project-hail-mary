"""Performance analytics for bar-level signal backtests."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from hailmary.analytics.metrics import PerformanceMetrics
from hailmary.backtest.signal_backtest import BarBacktestResult

PortfolioMode = Literal["rebalanced", "buy_and_hold", "fixed_stake"]


def _intra_drawdown(returns: pd.Series) -> float:
    # Prepend 1.0 so the entry price is the initial peak — without this,
    # a trade that drops on bar 1 would show zero drawdown.
    eq = pd.concat([pd.Series([1.0]), (1 + returns).cumprod()])
    return float((eq / eq.cummax() - 1).min())


class SignalAnalytics:
    """Compute performance statistics from a :class:`~hailmary.backtest.BarBacktestResult`.

    Provides both *per-symbol* views (``summary``, ``equity``, ``trade_stats``,
    ``trade_summary``) and a combined *portfolio* view that equal-weights all
    in-signal symbols at each bar (``portfolio_equity``, ``portfolio_returns``,
    ``portfolio_metrics``).

    Args:
        result: Output of :meth:`~hailmary.backtest.BarBacktest.run`.

    Example::

        signal_df = TrendSignal(200).run(bars)
        bt        = BarBacktest().run(signal_df)
        analytics = SignalAnalytics(bt)

        # Per-symbol
        analytics.summary()
        analytics.equity()

        # Combined portfolio — three capital models
        analytics.portfolio_equity(mode="rebalanced").plot()
        analytics.portfolio_equity(mode="buy_and_hold").plot()
        analytics.portfolio_equity(mode="fixed_stake", amount_per_entry=1_000).plot()
        analytics.portfolio_metrics(mode="rebalanced").summary()
    """

    def __init__(self, result: BarBacktestResult) -> None:
        self._data = result.data

    # ----------------------------------------------------------------- per-symbol

    def summary(self) -> pd.DataFrame:
        """Per-symbol aggregate: returns, drawdown, trade count, and time invested.

        Returns:
            DataFrame indexed by symbol with columns ``return_mtc``,
            ``return_conservative``, ``return_net``,
            ``max_drawdown_mtc``, ``avg_drawdown_mtc``,
            ``max_drawdown_conservative``, ``avg_drawdown_conservative``,
            ``entries``, ``exits``, ``invested_days``, ``pct_invested``.
        """
        total_bars = self._data.groupby(level="symbol").size()
        base = self._data.groupby(level="symbol").agg(
            return_mtc=("strategy_equity_mark_to_close", lambda x: x.iloc[-1] - 1),
            return_conservative=("strategy_equity_conservative", lambda x: x.iloc[-1] - 1),
            return_net=("strategy_equity_net", lambda x: x.iloc[-1] - 1),
            entries=("enter", "sum"),
            exits=("exit", "sum"),
            invested_days=("cycle", lambda x: x.ne("None").sum()),
        ).assign(pct_invested=lambda df: df["invested_days"] / total_bars * 100)
        return base.join(self._drawdown_summary())

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

    def trade_stats(self) -> pd.DataFrame:
        """One row per entry-to-exit cycle across all symbols.

        Returns:
            DataFrame with columns ``symbol``, ``entry_date``, ``exit_date``,
            ``duration`` (bars), ``return_net``, ``return_mtc``,
            ``return_conservative`` (all compound), and
            ``max_intra_drawdown_net``, ``max_intra_drawdown_mtc``,
            ``max_intra_drawdown_conservative`` (worst peak-to-trough within the
            cycle for each fill method; always ≤ 0).
        """
        in_trade = self._data[self._data["cycle"] != "None"].reset_index()
        return (
            in_trade.groupby(["symbol", "trade_cycle_id"])
            .agg(
                entry_date=("timestamp", "first"),
                exit_date=("timestamp", "last"),
                duration=("timestamp", "count"),
                return_net=("return_net", lambda x: (1 + x).prod() - 1),
                return_mtc=("return_mark_to_close", lambda x: (1 + x).prod() - 1),
                return_conservative=("return_conservative", lambda x: (1 + x).prod() - 1),
                max_intra_drawdown_net=("return_net", _intra_drawdown),
                max_intra_drawdown_mtc=("return_mark_to_close", _intra_drawdown),
                max_intra_drawdown_conservative=("return_conservative", _intra_drawdown),
            )
            .reset_index()
            .drop(columns="trade_cycle_id")
            .sort_values(["symbol", "entry_date"])
            .reset_index(drop=True)
        )

    def trade_summary(self, method: str = "net") -> pd.DataFrame:
        """Per-symbol signal-quality stats treating each entry-to-exit cycle as an independent bet.

        Args:
            method: Return column — ``"net"`` (default), ``"mtc"``, or ``"conservative"``.

        Returns:
            DataFrame indexed by symbol with columns ``n_trades``, ``win_rate``,
            ``avg_win``, ``avg_loss``, ``expectancy``, ``profit_factor``,
            ``avg_duration``, ``avg_intra_drawdown``, ``max_intra_drawdown``.
        """
        trades = self.trade_stats()
        ret_col = {
            "net": "return_net",
            "mtc": "return_mtc",
            "conservative": "return_conservative",
        }[method]
        dd_col = f"max_intra_drawdown_{method}"

        def _agg(g: pd.DataFrame) -> pd.Series:
            ret = g[ret_col]
            dd = g[dd_col]
            wins = ret[ret > 0]
            losses = ret[ret < 0]
            n = len(ret)
            win_rate = len(wins) / n if n > 0 else float("nan")
            avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
            loss_sum = float(losses.sum())
            expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
            ret_ex_top = ret.drop(ret.idxmax()) if len(ret) > 1 else ret
            return pd.Series({
                "n_trades": n,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "expectancy": expectancy,
                "expectancy_ex_top": float(ret_ex_top.mean()),
                "median_return": float(ret.median()),
                "profit_factor": float(wins.sum()) / abs(loss_sum) if loss_sum != 0 else float("inf"),
                "max_win": float(wins.max()) if len(wins) > 0 else 0.0,
                "max_loss": float(losses.min()) if len(losses) > 0 else 0.0,
                "skewness": float(ret.skew()),
                "avg_duration": float(g["duration"].mean()),
                "avg_intra_drawdown": float(dd.mean()),
                "max_intra_drawdown": float(dd.min()),
            })

        cols = [ret_col, dd_col, "duration"]
        return trades.groupby("symbol")[cols].apply(_agg)

    # ----------------------------------------------------------------- portfolio

    def portfolio_equity(
        self,
        method: str = "mtc",
        mode: PortfolioMode = "rebalanced",
        amount_per_entry: float = 1_000.0,
    ) -> pd.Series:
        """Equal-weight portfolio equity across all symbols (normalised to 1.0).

        Three capital models are available via ``mode``:

        - ``"rebalanced"`` *(default)*: weights reset to ``1/N_universe`` each bar —
          outperforming positions are trimmed daily so the portfolio stays equal-weight.
        - ``"buy_and_hold"``: each symbol starts at ``1/N_universe`` and compounds
          freely with no intra-period rebalancing.
        - ``"fixed_stake"``: a fixed ``amount_per_entry`` is deployed on every entry;
          realised P&L accumulates as cash and does not affect future entry sizes.

        Args:
            method: ``"mtc"`` (default), ``"conservative"``, or ``"net"``.
            mode: Capital model — see above.
            amount_per_entry: Dollar amount per entry (only used when ``mode="fixed_stake"``).

        Returns:
            pd.Series indexed by timestamp, normalised to 1.0 at the first bar.
        """
        if mode == "buy_and_hold":
            return self._portfolio_equity_buy_and_hold(method)
        if mode == "fixed_stake":
            return self._portfolio_equity_fixed_stake(method, amount_per_entry)
        return (1 + self._portfolio_returns_rebalanced(method)).cumprod().rename(
            f"portfolio_equity_{method}_rebalanced"
        )

    def portfolio_returns(
        self,
        method: str = "mtc",
        mode: PortfolioMode = "rebalanced",
        amount_per_entry: float = 1_000.0,
    ) -> pd.Series:
        """Daily portfolio returns.

        Args:
            method: ``"mtc"`` (default), ``"conservative"``, or ``"net"``.
            mode: Capital model — see :meth:`portfolio_equity`.
            amount_per_entry: Dollar amount per entry (only used when ``mode="fixed_stake"``).

        Returns:
            pd.Series of daily returns indexed by timestamp.
        """
        if mode == "buy_and_hold":
            return self._portfolio_equity_buy_and_hold(method).pct_change().fillna(0)
        if mode == "fixed_stake":
            return self._portfolio_equity_fixed_stake(method, amount_per_entry).pct_change().fillna(0)
        return self._portfolio_returns_rebalanced(method)

    def portfolio_metrics(
        self,
        method: str = "mtc",
        risk_free_rate: float = 0.0,
        mode: PortfolioMode = "rebalanced",
        amount_per_entry: float = 1_000.0,
    ) -> PerformanceMetrics:
        """Return a :class:`~hailmary.analytics.PerformanceMetrics` for the combined portfolio.

        Args:
            method: ``"mtc"`` (default), ``"conservative"``, or ``"net"``.
            risk_free_rate: Annualised risk-free rate (default 0).
            mode: Capital model — see :meth:`portfolio_equity`.
            amount_per_entry: Dollar amount per entry (only used when ``mode="fixed_stake"``).

        Returns:
            :class:`PerformanceMetrics` instance; call ``.summary()`` for a full table.
        """
        return PerformanceMetrics(
            self.portfolio_returns(method=method, mode=mode, amount_per_entry=amount_per_entry),
            risk_free_rate=risk_free_rate,
        )

    # ----------------------------------------------------------------- private

    def _drawdown_summary(self) -> pd.DataFrame:
        """Per-symbol max and average drawdown for MTC and conservative equity curves."""
        result: dict[str, pd.Series] = {}
        for col, suffix in [
            ("strategy_equity_mark_to_close", "mtc"),
            ("strategy_equity_conservative", "conservative"),
        ]:
            equity = self._data[col].unstack(level="symbol")
            dd = equity / equity.cummax() - 1
            result[f"max_drawdown_{suffix}"] = dd.min()
            in_dd = dd[dd < 0]
            result[f"avg_drawdown_{suffix}"] = in_dd.mean().where(in_dd.count() > 0, 0.0)
        return pd.DataFrame(result)

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
        return (weights * returns_wide).sum(axis=1).rename(f"portfolio_returns_{method}_rebalanced")

    def _portfolio_equity_buy_and_hold(self, method: str) -> pd.Series:
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
        return equity_wide.mean(axis=1).rename(f"portfolio_equity_{method}_buy_and_hold")

    def _portfolio_equity_fixed_stake(self, method: str, amount_per_entry: float) -> pd.Series:
        """NAV curve for the fixed-dollar-per-entry capital model.

        Within each trade the position compounds from amount_per_entry.
        Realised P&L from closed trades accumulates as cash and does not
        affect the entry size of future trades.
        """
        ret_col = {
            "mtc": "return_mark_to_close",
            "conservative": "return_conservative",
            "net": "return_net",
        }[method]

        df = self._data[[ret_col, "cycle", "trade_cycle_id"]]
        in_trade_mask = df["cycle"] != "None"
        exit_mask = df["cycle"] == "exit"

        # Within-trade cumulative equity: (1+r_1)(1+r_2)…(1+r_t) per (symbol, trade)
        in_trade_df = df[in_trade_mask]
        sym_level = in_trade_df.index.get_level_values("symbol")
        within_trade_equity = (
            in_trade_df[ret_col]
            .groupby([sym_level, in_trade_df["trade_cycle_id"]])
            .transform(lambda x: (1 + x).cumprod())
            .reindex(df.index)  # NaN on flat bars
        )

        # Unrealised P&L per (symbol, bar): amount × (equity − 1) when in trade, 0 when flat
        unrealised_pnl = (within_trade_equity - 1).fillna(0) * amount_per_entry

        # Total unrealised across symbols at each timestamp
        total_unrealised = unrealised_pnl.unstack(level="symbol").fillna(0).sum(axis=1)

        # Realised P&L: capture unrealised on exit bars, shift +1 so it starts after exit
        exit_pnl = unrealised_pnl.where(exit_mask).fillna(0)
        realized_cumsum = (
            exit_pnl.unstack(level="symbol").fillna(0).sum(axis=1)
            .shift(1).fillna(0).cumsum()
        )

        n_universe = df.index.get_level_values("symbol").nunique()
        initial_capital = amount_per_entry * n_universe
        nav = initial_capital + total_unrealised + realized_cumsum
        return (nav / initial_capital).rename(f"portfolio_equity_{method}_fixed_stake")
