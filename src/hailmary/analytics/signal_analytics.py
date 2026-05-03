"""Performance analytics for bar-level signal backtests.

Two evaluation lenses, both consuming a :class:`~hailmary.backtest.BarBacktestResult`:

- :class:`SignalTradePerformance` — every entry-to-exit cycle is treated as one
  independent $1 bet.  No capital allocation, no compounding between trades.
  Win rate, expectancy, profit factor, distribution shape, intra-trade timing,
  and per-trade quality flags live here.

- :class:`SignalAllocationPerformance` — capital is deployed and compounds.
  Per-symbol buy-and-hold equity curves and portfolio NAV/returns/metrics
  under explicit capital models (rebalanced, buy-and-hold, fixed stake).
"""

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


class SignalTradePerformance:
    """Trade-level performance: every entry-to-exit cycle as one iid bet.

    No capital allocation enters these calculations — each trade is one $1 bet
    and trades do not compound into each other.  Use this lens to characterise
    the *signal itself*: edge, distribution shape, timing, quality flags.

    Args:
        result: Output of :meth:`~hailmary.backtest.BarBacktest.run`.

    Example::

        signal_df = TrendSignal(200).run(bars)
        bt        = BarBacktest().run(signal_df)
        trades    = SignalTradePerformance(bt)

        trades.trade_summary()                  # per-symbol edge / distribution stats
        trades.trade_stats()                    # one row per entry→exit cycle
        paths = trades.trade_paths()            # per-trade cumulative-return paths
        SignalTradePerformance.d5_stats("BTC-USD", "net", paths)
        SignalTradePerformance.quality_flags(expectancy=0.05, expectancy_ex_top=0.01,
                                             median_return=0.02, skewness=0.3)
    """

    def __init__(self, result: BarBacktestResult) -> None:
        self._data = result.data

    # ----------------------------------------------------------------- per-trade

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

    # ----------------------------------------------------------------- paths / timing / flags

    def trade_paths(self) -> list[dict]:
        """Build one dict per entry-to-exit cycle with cumulative return paths and timing fields.

        Returns:
            List of dicts, one per trade, each containing:

            - ``label`` — e.g. "BTC-USD T3"
            - ``sym`` — symbol string
            - ``cum_net`` / ``cum_con`` — cumulative return Series starting at 0.0
            - ``entry_dt`` / ``exit_dt`` — Timestamps
            - ``duration`` — number of bars in the trade
            - ``final_net`` / ``final_con`` — compound return over the full trade
            - ``d5_net`` / ``d5_con`` — compound return at bar 5 (clamped to duration)
            - ``d5_to_exit_net`` / ``d5_to_exit_con`` — geometric return from bar 5 to close
            - ``max_dd_net`` / ``max_dd_con`` — worst peak-to-trough within the trade
        """
        df = self._data
        trade_stats = self.trade_stats()
        paths: list[dict] = []
        sym_counters: dict[str, int] = {}

        for (sym, _trade_id), group in df[df["cycle"] != "None"].groupby(
            ["symbol", "trade_cycle_id"]
        ):
            sym_counters[sym] = sym_counters.get(sym, 0) + 1
            label    = f"{sym} T{sym_counters[sym]}"
            rets_net = group["return_net"].reset_index(drop=True)
            rets_con = group["return_conservative"].reset_index(drop=True)
            cum_net  = pd.concat([pd.Series([0.0]), (1 + rets_net).cumprod() - 1]).reset_index(drop=True)
            cum_con  = pd.concat([pd.Series([0.0]), (1 + rets_con).cumprod() - 1]).reset_index(drop=True)
            entry_dt = group.index.get_level_values("timestamp")[0]
            exit_dt  = group.index.get_level_values("timestamp")[-1]
            ts_row   = trade_stats[
                (trade_stats["symbol"] == sym) & (trade_stats["entry_date"] == entry_dt)
            ]
            d5        = min(5, len(cum_net) - 1)
            d5_net    = float(cum_net.iloc[d5])
            d5_con    = float(cum_con.iloc[d5])
            final_net = float(cum_net.iloc[-1])
            final_con = float(cum_con.iloc[-1])
            paths.append({
                "label":          label,
                "sym":            sym,
                "cum_net":        cum_net,
                "cum_con":        cum_con,
                "entry_dt":       entry_dt,
                "exit_dt":        exit_dt,
                "duration":       len(rets_net),
                "final_net":      final_net,
                "final_con":      final_con,
                "d5_net":         d5_net,
                "d5_con":         d5_con,
                "d5_to_exit_net": (1 + final_net) / (1 + d5_net) - 1,
                "d5_to_exit_con": (1 + final_con) / (1 + d5_con) - 1,
                "max_dd_net":     float(ts_row["max_intra_drawdown_net"].iloc[0])           if len(ts_row) else float("nan"),
                "max_dd_con":     float(ts_row["max_intra_drawdown_conservative"].iloc[0])  if len(ts_row) else float("nan"),
            })
        return paths

    @staticmethod
    def d5_stats(sym: str, method: str, paths: list[dict]) -> dict:
        """Timing statistics at the 5-bar mark for one symbol.

        Args:
            sym: Symbol to filter on.
            method: ``"net"`` or ``"conservative"``.
            paths: Output of :meth:`trade_paths`.

        Returns:
            Dict with keys ``wr_d5``, ``wr_d5_n``, ``wr_d5_nwin``,
            ``wr_tail``, ``wr_tail_n``, ``wr_tail_nwin``, ``capture_med``.
            Win-rate fields are ``nan`` when no qualifying trades exist.
        """
        sym_paths = [p for p in paths if p["sym"] == sym]
        if not sym_paths:
            return {
                "wr_d5": float("nan"), "wr_d5_n": 0, "wr_d5_nwin": 0,
                "wr_tail": float("nan"), "wr_tail_n": 0, "wr_tail_nwin": 0,
                "capture_med": float("nan"),
            }
        d5_col   = "d5_net"         if method == "net" else "d5_con"
        tail_col = "d5_to_exit_net" if method == "net" else "d5_to_exit_con"

        # WR@5d: trades that actually reached bar 5
        d5_paths   = [p for p in sym_paths if p["duration"] >= 5]
        # 5d→Exit WR and capture: trades with a tail segment beyond bar 5
        tail_paths = [p for p in sym_paths if p["duration"] > 5]

        d5_vals   = [p[d5_col]   for p in d5_paths]
        tail_vals = [p[tail_col] for p in tail_paths]

        n_d5     = len(d5_vals)
        n_win_d5 = sum(1 for v in d5_vals if v > 0)
        n_tail   = len(tail_vals)
        n_win_t  = sum(1 for v in tail_vals if v > 0)

        return {
            "wr_d5":        n_win_d5 / n_d5 if n_d5 > 0 else float("nan"),
            "wr_d5_n":      n_d5,
            "wr_d5_nwin":   n_win_d5,
            "wr_tail":      n_win_t / n_tail if n_tail > 0 else float("nan"),
            "wr_tail_n":    n_tail,
            "wr_tail_nwin": n_win_t,
        }

    @staticmethod
    def quality_flags(
        expectancy: float,
        expectancy_ex_top: float,
        median_return: float,
        skewness: float,
        net_expectancy: float | None = None,
        net_median: float | None = None,
    ) -> dict[str, bool]:
        """Signal-quality warning flags as a plain boolean mapping.

        Args:
            expectancy: Expected return per trade.
            expectancy_ex_top: Expectancy after removing the best trade.
            median_return: Median trade return.
            skewness: Return distribution skewness.
            net_expectancy: Net-fill expectancy (supply when computing conservative flags).
            net_median: Net-fill median (supply when computing conservative flags).

        Returns:
            Dict mapping flag name → ``True`` if the condition is triggered:

            - ``median_negative`` — expectancy > 0 but median < 0
            - ``top_trade_outlier`` — expectancy > 0 and best trade accounts for > 30% of it
            - ``skewed_right`` / ``skewed_left`` — |skewness| > 1.5
            - ``edge_reversed`` — conservative expectancy ≤ 0 while net > 0
            - ``fill_halves_edge`` — conservative < 50% of net expectancy
            - ``median_flips`` — net median ≥ 0 but conservative median < 0
        """
        top_share = (
            abs(expectancy - expectancy_ex_top) / abs(expectancy)
            if expectancy > 0 else 0.0
        )
        return {
            "median_negative":   expectancy > 0 and median_return < 0,
            "top_trade_outlier": expectancy > 0 and top_share > 0.3,
            "skewed_right":      skewness > 1.5,
            "skewed_left":       skewness < -1.5,
            "edge_reversed":     net_expectancy is not None and net_expectancy > 0 and expectancy <= 0,
            "fill_halves_edge":  net_expectancy is not None and net_expectancy > 0 and 0 < expectancy < net_expectancy * 0.5,
            "median_flips":      net_median is not None and net_median >= 0 and median_return < 0,
        }


class SignalAllocationPerformance:
    """Allocation-level performance: capital is deployed and compounds.

    Provides per-symbol buy-and-hold views (``summary``, ``equity``) and
    portfolio NAV/returns/metrics under three explicit capital models
    (``portfolio_equity``, ``portfolio_returns``, ``portfolio_metrics``).
    Per-symbol buy-and-hold is the N=1 special case of the portfolio.

    Args:
        result: Output of :meth:`~hailmary.backtest.BarBacktest.run`.

    Example::

        signal_df = TrendSignal(200).run(bars)
        bt        = BarBacktest().run(signal_df)
        alloc     = SignalAllocationPerformance(bt)

        # Per-symbol buy-and-hold
        alloc.summary()
        alloc.equity()

        # Combined portfolio — three capital models
        alloc.portfolio_equity(mode="rebalanced").plot()
        alloc.portfolio_equity(mode="buy_and_hold").plot()
        alloc.portfolio_equity(mode="fixed_stake", amount_per_entry=1_000).plot()
        alloc.portfolio_metrics(mode="rebalanced").summary()
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
