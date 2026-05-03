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

FillMethod = Literal["net", "mtc", "conservative"]
PortfolioMode = Literal["rebalanced", "buy_and_hold", "fixed_stake"]

_RETURN_COL: dict[str, str] = {
    "net": "return_net",
    "mtc": "return_mark_to_close",
    "conservative": "return_conservative",
}
_EQUITY_COL: dict[str, str] = {
    "net": "strategy_equity_net",
    "mtc": "strategy_equity_mark_to_close",
    "conservative": "strategy_equity_conservative",
}

# ---------------------------------------------------------------------------
# Column / flag documentation — single source of truth shared across viz
# (HTML tearsheet table notes) and notebook (markdown render of method docs).
# Each entry is ``(display_label, description)``.  When a column changes or is
# dropped, update the entry here and both surfaces pick it up.
# ---------------------------------------------------------------------------

ColumnDoc = tuple[str, str]

TRADE_STATS_DOCS: dict[str, ColumnDoc] = {
    "label":      ("Label",    "Display label like 'BTC-USD T3' (T-numbered per symbol)."),
    "symbol":     ("Symbol",   "Symbol the trade was on."),
    "entry_date": ("Entry",    "Bar timestamp at entry."),
    "exit_date":  ("Exit",     "Bar timestamp at exit."),
    "duration":   ("Duration", "Number of bars the trade was held."),
    "return_net":          ("Return (Net)", "Compound net return (after fills + costs)."),
    "return_mtc":          ("Return (MTC)", "Compound MTC return (no fill modelling)."),
    "return_conservative": ("Return (Conservative)", "Compound return, worst-fill assumption."),
    "max_intra_drawdown_net": ("Max DD (Net)", "Worst peak-to-trough (≤ 0)."),
    "max_intra_drawdown_mtc": ("Max DD (MTC)", "Worst peak-to-trough, MTC fill."),
    "max_intra_drawdown_conservative": (
        "Max DD (Conservative)",
        "Worst peak-to-trough, conservative fill.",
    ),
    "d5_net":          ("5d (Net)", "Net return over first 5 bars (or full trade if shorter)."),
    "d5_mtc":          ("5d (MTC)", "MTC return over first 5 bars."),
    "d5_conservative": ("5d (Conservative)", "Conservative return over first 5 bars."),
    "d5_to_exit_net":          ("5d → Exit (Net)", "Net return bar 5 → close (0 if duration ≤ 5)."),
    "d5_to_exit_mtc":          ("5d → Exit (MTC)", "MTC return bar 5 → close."),
    "d5_to_exit_conservative": ("5d → Exit (Conservative)", "Conservative return bar 5 → close."),
}

TRADE_SUMMARY_DOCS: dict[str, ColumnDoc] = {
    "n_trades":           ("N Trades",      "Number of entry-to-exit cycles."),
    "win_rate":           ("Win Rate",      "Fraction of trades that closed positive."),
    "avg_win":            ("Avg Win",       "Mean return across winning trades."),
    "avg_loss":           ("Avg Loss",      "Mean return across losing trades (negative)."),
    "expectancy":        ("Expectancy", "win_rate × avg_win + (1 − win_rate) × avg_loss."),
    "expectancy_ex_top": ("Exp ex-Top", "Expectancy after dropping the best trade."),
    "median_return":     ("Median",     "50th-pct return; Median ≪ Expectancy = right-skewed."),
    "profit_factor":      ("Profit Factor", "Σ wins / |Σ losses|; > 1 earns more than it loses."),
    "max_win":            ("Max Win",       "Best single-trade return."),
    "max_loss":           ("Max Loss",      "Worst single-trade return."),
    "skewness":           ("Skewness",      "> +1: rare large wins; < −1: rare large losses."),
    "avg_duration":       ("Avg Duration",  "Mean bars held per trade."),
    "avg_intra_drawdown": ("Avg DD",        "Mean of per-trade max intra-drawdowns (≤ 0)."),
    "max_intra_drawdown": ("Worst DD",      "Worst per-trade intra-drawdown observed (≤ 0)."),
}

QUALITY_FLAG_DOCS: dict[str, ColumnDoc] = {
    "median_negative":   ("Median < 0",       "Expectancy > 0 but median < 0."),
    "top_trade_outlier": ("Top Trade > 30%",  "Expectancy > 0 and best trade > 30% of it."),
    "skewed_right":      ("Skewed Right",     "Skewness > 1.5 — rare large wins."),
    "skewed_left":       ("Skewed Left",      "Skewness < −1.5 — rare large losses."),
    "edge_reversed":     ("Edge Reversed",    "Net expectancy > 0 but this fill ≤ 0."),
    "fill_halves_edge":  ("Fill Halves Edge", "This fill's expectancy < 50% of net."),
    "median_flips":      ("Median Flips",     "Net median ≥ 0 but this fill's median < 0."),
}

D5_STATS_DOCS: dict[str, ColumnDoc] = {
    "wr_d5":        ("WR @ 5d",      "Win rate at bar 5 — trades ≥ 5 bars only."),
    "wr_d5_n":      ("N (≥5d)",      "Number of trades that reached bar 5."),
    "wr_d5_nwin":   ("Wins (5d)",    "Number of those trades positive at bar 5."),
    "wr_tail":      ("WR 5d → Exit", "Win rate from bar 5 to close — trades > 5 bars only."),
    "wr_tail_n":    ("N (>5d)",      "Number of trades with a tail beyond bar 5."),
    "wr_tail_nwin": ("Wins (tail)",  "Number of those trades whose tail segment was positive."),
}

FILL_METHOD_DOCS: dict[str, ColumnDoc] = {
    "net":          ("Net",          "MTC fill minus round-trip cost — realistic execution."),
    "mtc":          ("MTC",          "Mark-to-close — frictionless benchmark, no fill modelling."),
    "conservative": ("Conservative", "Worst-fill: high entry / low exit — bear-case execution."),
}


def docs_markdown(docs: dict[str, ColumnDoc], title: str | None = None) -> str:
    """Render a docs dict as a Markdown bullet list."""
    lines = [f"**{title}**", ""] if title else []
    for label, desc in docs.values():
        lines.append(f"- **{label}** — {desc}")
    return "\n".join(lines)


def _compound(x: pd.Series) -> float:
    return float((1 + x).prod() - 1)


def _compound_first_5(x: pd.Series) -> float:
    return _compound(x.iloc[:5])


def _compound_after_5(x: pd.Series) -> float:
    return _compound(x.iloc[5:])


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

        ts = trades.trade_stats()                           # one row per trade
        trades.trade_summary(trade_stats=ts)                # per-symbol edge stats
        paths = trades.trade_paths(trade_stats=ts)          # per-bar cum returns
        SignalTradePerformance.d5_stats("BTC-USD", "net", trade_stats=ts)

    The expensive primitive is :meth:`trade_stats`; every other method accepts
    it as an optional ``trade_stats`` keyword so callers can compute it once
    and thread it through.
    """

    def __init__(self, result: BarBacktestResult) -> None:
        self._data = result.data

    # ----------------------------------------------------------------- per-trade

    def trade_stats(self) -> pd.DataFrame:
        """One row per entry-to-exit cycle — the canonical per-trade scalar table.

        Returns:
            DataFrame with columns ``symbol``, ``label`` (e.g. ``"BTC-USD T3"``),
            ``entry_date``, ``exit_date``, ``duration`` (bars), and for each
            fill method ``m`` ∈ {``net``, ``mtc``, ``conservative``}:

            - ``return_{m}`` — compound return over the trade
            - ``max_intra_drawdown_{m}`` — worst peak-to-trough within the
              cycle (always ≤ 0)
            - ``d5_{m}`` — compound return over the first 5 bars (or the full
              trade if shorter)
            - ``d5_to_exit_{m}`` — compound return from bar 5 to close
              (0 when ``duration ≤ 5``)
        """
        in_trade = self._data[self._data["cycle"] != "None"].reset_index()

        aggs: dict[str, tuple] = {
            "entry_date": ("timestamp", "first"),
            "exit_date":  ("timestamp", "last"),
            "duration":   ("timestamp", "count"),
        }
        for method, ret_col in _RETURN_COL.items():
            aggs[f"return_{method}"]             = (ret_col, _compound)
            aggs[f"max_intra_drawdown_{method}"] = (ret_col, _intra_drawdown)
            aggs[f"d5_{method}"]                 = (ret_col, _compound_first_5)
            aggs[f"d5_to_exit_{method}"]         = (ret_col, _compound_after_5)

        out = (
            in_trade.groupby(["symbol", "trade_cycle_id"])
            .agg(**aggs)
            .reset_index()
            .drop(columns="trade_cycle_id")
            .sort_values(["symbol", "entry_date"])
            .reset_index(drop=True)
        )
        counter = out.groupby("symbol").cumcount() + 1
        out.insert(1, "label", out["symbol"] + " T" + counter.astype(str))
        return out

    def trade_summary(
        self,
        method: FillMethod = "net",
        *,
        trade_stats: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Per-symbol signal-quality stats treating each entry-to-exit cycle as an independent bet.

        Args:
            method: Fill method — ``"net"`` (default), ``"mtc"``, or ``"conservative"``.
            trade_stats: Precomputed :meth:`trade_stats` frame to avoid recomputing.

        Returns:
            DataFrame indexed by symbol with columns ``n_trades``, ``win_rate``,
            ``avg_win``, ``avg_loss``, ``expectancy``, ``expectancy_ex_top``,
            ``median_return``, ``profit_factor``, ``max_win``, ``max_loss``,
            ``skewness``, ``avg_duration``, ``avg_intra_drawdown``,
            ``max_intra_drawdown``.
        """
        ts = trade_stats if trade_stats is not None else self.trade_stats()
        ret_col = f"return_{method}"
        dd_col  = f"max_intra_drawdown_{method}"

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
            # "Expectancy excluding the best trade" needs at least 2 trades to be meaningful.
            expectancy_ex_top = float(ret.drop(ret.idxmax()).mean()) if n > 1 else float("nan")
            return pd.Series({
                "n_trades": n,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "expectancy": expectancy,
                "expectancy_ex_top": expectancy_ex_top,
                "median_return": float(ret.median()),
                "profit_factor": float(wins.sum()) / abs(loss_sum) if loss_sum != 0 else float("inf"),
                "max_win": float(wins.max()) if len(wins) > 0 else 0.0,
                "max_loss": float(losses.min()) if len(losses) > 0 else 0.0,
                "skewness": float(ret.skew()),
                "avg_duration": float(g["duration"].mean()),
                "avg_intra_drawdown": float(dd.mean()),
                "max_intra_drawdown": float(dd.min()),
            })

        return ts.groupby("symbol")[[ret_col, dd_col, "duration"]].apply(_agg)

    # ----------------------------------------------------------------- paths / timing / flags

    def trade_paths(self, *, trade_stats: pd.DataFrame | None = None) -> list[dict]:
        """Per-trade cumulative-return paths plus identifying scalars.

        The scalar fields (``final_*``, ``d5_*``, ``max_dd_*``, ``duration``,
        ``entry_dt``, ``exit_dt``) are sourced from :meth:`trade_stats` rather
        than recomputed.  This method's unique contribution is the per-bar
        ``cum_net`` / ``cum_con`` Series used by chart helpers.

        Args:
            trade_stats: Precomputed :meth:`trade_stats` frame to avoid recomputing.

        Returns:
            List of dicts, one per trade, each containing:

            - ``label`` — e.g. ``"BTC-USD T3"``
            - ``sym`` — symbol string
            - ``cum_net`` / ``cum_con`` — cumulative return Series starting at 0.0
            - ``entry_dt`` / ``exit_dt`` — Timestamps
            - ``duration`` — number of bars in the trade
            - ``final_net`` / ``final_con`` — compound return over the full trade
            - ``d5_net`` / ``d5_con`` — compound return at bar 5 (clamped to duration)
            - ``d5_to_exit_net`` / ``d5_to_exit_con`` — return from bar 5 to close
            - ``max_dd_net`` / ``max_dd_con`` — worst peak-to-trough within the trade
        """
        ts = trade_stats if trade_stats is not None else self.trade_stats()
        ts_by_label = ts.set_index("label")

        df = self._data
        paths: list[dict] = []
        sym_counter: dict[str, int] = {}
        for (sym, _trade_id), group in df[df["cycle"] != "None"].groupby(
            ["symbol", "trade_cycle_id"]
        ):
            sym_counter[sym] = sym_counter.get(sym, 0) + 1
            label    = f"{sym} T{sym_counter[sym]}"
            rets_net = group["return_net"].reset_index(drop=True)
            rets_con = group["return_conservative"].reset_index(drop=True)
            cum_net  = pd.concat([pd.Series([0.0]), (1 + rets_net).cumprod() - 1]).reset_index(drop=True)
            cum_con  = pd.concat([pd.Series([0.0]), (1 + rets_con).cumprod() - 1]).reset_index(drop=True)
            row      = ts_by_label.loc[label]
            paths.append({
                "label":          label,
                "sym":            sym,
                "cum_net":        cum_net,
                "cum_con":        cum_con,
                "entry_dt":       row["entry_date"],
                "exit_dt":        row["exit_date"],
                "duration":       int(row["duration"]),
                "final_net":      float(row["return_net"]),
                "final_con":      float(row["return_conservative"]),
                "d5_net":         float(row["d5_net"]),
                "d5_con":         float(row["d5_conservative"]),
                "d5_to_exit_net": float(row["d5_to_exit_net"]),
                "d5_to_exit_con": float(row["d5_to_exit_conservative"]),
                "max_dd_net":     float(row["max_intra_drawdown_net"]),
                "max_dd_con":     float(row["max_intra_drawdown_conservative"]),
            })
        return paths

    @staticmethod
    def d5_stats(
        sym: str,
        method: FillMethod,
        *,
        trade_stats: pd.DataFrame,
    ) -> dict:
        """Timing statistics at the 5-bar mark for one symbol.

        Args:
            sym: Symbol to filter on.
            method: ``"net"``, ``"mtc"``, or ``"conservative"``.
            trade_stats: :meth:`trade_stats` DataFrame.

        Returns:
            Dict with keys ``wr_d5``, ``wr_d5_n``, ``wr_d5_nwin``,
            ``wr_tail``, ``wr_tail_n``, ``wr_tail_nwin``.  Win-rate fields are
            ``nan`` when no qualifying trades exist.
        """
        sym_rows = trade_stats[trade_stats["symbol"] == sym]
        if sym_rows.empty:
            return {
                "wr_d5": float("nan"), "wr_d5_n": 0, "wr_d5_nwin": 0,
                "wr_tail": float("nan"), "wr_tail_n": 0, "wr_tail_nwin": 0,
            }

        d5_col   = f"d5_{method}"
        tail_col = f"d5_to_exit_{method}"

        d5_rows   = sym_rows[sym_rows["duration"] >= 5]
        tail_rows = sym_rows[sym_rows["duration"] > 5]

        n_d5     = len(d5_rows)
        n_tail   = len(tail_rows)
        n_win_d5 = int((d5_rows[d5_col] > 0).sum())
        n_win_t  = int((tail_rows[tail_col] > 0).sum())

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
        row: pd.Series,
        *,
        net_row: pd.Series | None = None,
    ) -> dict[str, bool]:
        """Signal-quality warning flags for a single row of :meth:`trade_summary`.

        Args:
            row: A row of :meth:`trade_summary` — must have ``expectancy``,
                ``expectancy_ex_top``, ``median_return``, ``skewness``.
            net_row: Corresponding net-fill row (with ``expectancy`` and
                ``median_return``).  Supply when ``row`` is a non-net method
                so the cross-fill flags can fire.

        Returns:
            Dict mapping flag name → ``True`` if the condition is triggered:

            - ``median_negative`` — expectancy > 0 but median < 0
            - ``top_trade_outlier`` — expectancy > 0 and best trade accounts for > 30% of it
            - ``skewed_right`` / ``skewed_left`` — |skewness| > 1.5
            - ``edge_reversed`` — net expectancy > 0 while this row's ≤ 0
            - ``fill_halves_edge`` — this row's expectancy < 50% of net
            - ``median_flips`` — net median ≥ 0 but this row's median < 0
        """
        expectancy        = float(row["expectancy"])
        expectancy_ex_top = float(row["expectancy_ex_top"])
        median_return     = float(row["median_return"])
        skewness          = float(row["skewness"])

        # NaN-safe: with n=1 expectancy_ex_top is NaN and the comparison resolves to False.
        top_share = (
            abs(expectancy - expectancy_ex_top) / abs(expectancy)
            if expectancy > 0 and pd.notna(expectancy_ex_top) else 0.0
        )
        flags = {
            "median_negative":   expectancy > 0 and median_return < 0,
            "top_trade_outlier": expectancy > 0 and top_share > 0.3,
            "skewed_right":      skewness > 1.5,
            "skewed_left":       skewness < -1.5,
            "edge_reversed":     False,
            "fill_halves_edge":  False,
            "median_flips":      False,
        }
        if net_row is not None:
            net_exp = float(net_row["expectancy"])
            net_med = float(net_row["median_return"])
            flags["edge_reversed"]    = net_exp > 0 and expectancy <= 0
            flags["fill_halves_edge"] = net_exp > 0 and 0 < expectancy < net_exp * 0.5
            flags["median_flips"]     = net_med >= 0 and median_return < 0
        return flags

    # ----------------------------------------------------------------- display-shaped data

    def quality_table(
        self,
        methods: tuple[FillMethod, ...] = ("net", "conservative"),
        *,
        trade_stats: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Wide per-(symbol, method) quality DataFrame for downstream display.

        Combines :meth:`trade_summary` metrics with :meth:`d5_stats` timing and
        :meth:`quality_flags` into one frame indexed by ``(symbol, method)``.

        When ``"net"`` is in ``methods``, non-net rows compare against net via
        ``net_row`` so the cross-fill flags (``edge_reversed``,
        ``fill_halves_edge``, ``median_flips``) populate.

        Args:
            methods: Fill methods to include, default ``("net", "conservative")``.
            trade_stats: Precomputed :meth:`trade_stats` frame to avoid recomputing.

        Returns:
            DataFrame indexed by ``(symbol, method)`` with the per-symbol
            summary columns plus ``wr_d5``, ``wr_d5_n``, ``wr_d5_nwin``,
            ``wr_tail``, ``wr_tail_n``, ``wr_tail_nwin``, and a ``flags``
            column holding the raw ``dict[str, bool]`` from
            :meth:`quality_flags`.
        """
        ts = trade_stats if trade_stats is not None else self.trade_stats()
        summaries = {m: self.trade_summary(m, trade_stats=ts) for m in methods}
        ts_net = summaries.get("net")

        all_symbols: set[str] = set()
        for s in summaries.values():
            all_symbols.update(s.index)

        rows: list[dict] = []
        for sym in sorted(all_symbols):
            for method in methods:
                summary = summaries[method]
                if sym not in summary.index:
                    continue
                row = summary.loc[sym].to_dict()
                row["symbol"] = sym
                row["method"] = method
                row.update(self.d5_stats(sym, method, trade_stats=ts))
                net_row = (
                    ts_net.loc[sym]
                    if method != "net" and ts_net is not None and sym in ts_net.index
                    else None
                )
                row["flags"] = self.quality_flags(summary.loc[sym], net_row=net_row)
                rows.append(row)

        return pd.DataFrame(rows).set_index(["symbol", "method"])


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
            return_mtc=(_EQUITY_COL["mtc"], lambda x: x.iloc[-1] - 1),
            return_conservative=(_EQUITY_COL["conservative"], lambda x: x.iloc[-1] - 1),
            return_net=(_EQUITY_COL["net"], lambda x: x.iloc[-1] - 1),
            entries=("enter", "sum"),
            exits=("exit", "sum"),
            invested_days=("cycle", lambda x: x.ne("None").sum()),
        ).assign(pct_invested=lambda df: df["invested_days"] / total_bars * 100)
        return base.join(self._drawdown_summary())

    def equity(self, method: FillMethod = "mtc") -> pd.DataFrame:
        """Wide equity curve DataFrame (index=timestamp, columns=symbol).

        Args:
            method: ``"mtc"`` (default), ``"conservative"``, or ``"net"``.
        """
        return self._data[_EQUITY_COL[method]].unstack(level="symbol")

    # ----------------------------------------------------------------- portfolio

    def portfolio_equity(
        self,
        method: FillMethod = "mtc",
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
        method: FillMethod = "mtc",
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
        method: FillMethod = "mtc",
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
        for method in ("mtc", "conservative"):
            equity = self._data[_EQUITY_COL[method]].unstack(level="symbol")
            dd = equity / equity.cummax() - 1
            result[f"max_drawdown_{method}"] = dd.min()
            in_dd = dd[dd < 0]
            result[f"avg_drawdown_{method}"] = in_dd.mean().where(in_dd.count() > 0, 0.0)
        return pd.DataFrame(result)

    def _portfolio_returns_rebalanced(self, method: FillMethod) -> pd.Series:
        """Daily returns assuming the portfolio rebalances to 1/N_universe each bar."""
        in_trade = (self._data["cycle"] != "None").astype(int).unstack(level="symbol")
        returns_wide = self._data[_RETURN_COL[method]].unstack(level="symbol")
        weights = in_trade / len(in_trade.columns)
        return (weights * returns_wide).sum(axis=1).rename(f"portfolio_returns_{method}_rebalanced")

    def _portfolio_equity_buy_and_hold(self, method: FillMethod) -> pd.Series:
        """Portfolio equity when each position compounds freely from its 1/N_universe entry.

        Equivalent to the mean of the per-symbol equity curves: each symbol starts
        at 1/N and drifts with its own cumulative return.  No intra-period rebalancing.
        """
        equity_wide = self._data[_EQUITY_COL[method]].unstack(level="symbol")
        return equity_wide.mean(axis=1).rename(f"portfolio_equity_{method}_buy_and_hold")

    def _portfolio_equity_fixed_stake(
        self, method: FillMethod, amount_per_entry: float
    ) -> pd.Series:
        """NAV curve for the fixed-dollar-per-entry capital model.

        Within each trade the position compounds from amount_per_entry.
        Realised P&L from closed trades accumulates as cash and does not
        affect the entry size of future trades.
        """
        ret_col = _RETURN_COL[method]
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
