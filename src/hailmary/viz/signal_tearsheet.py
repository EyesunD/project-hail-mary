"""Standalone HTML tearsheet for a single signal backtest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hailmary.analytics.signal_analytics import SignalAnalytics
from hailmary.viz.theme import PALETTE, apply_theme

_P = PALETTE

_CSS = """
*, *::before, *::after { box-sizing: border-box; }
body {
    background: #0d1117; color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 13px; line-height: 1.5;
    margin: 0; padding: 28px 40px 56px; max-width: 1600px;
}
h1 { font-size: 1.4rem; font-weight: 600; margin: 0 0 4px; }
h2 {
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #8b949e;
    margin: 36px 0 14px; padding-bottom: 6px; border-bottom: 1px solid #30363d;
}
.meta { font-size: 0.8rem; color: #8b949e; margin-bottom: 28px; }
.cards {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 10px; margin-bottom: 16px;
}
.card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; }
.card-label { font-size: 0.67rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.07em; }
.card-value { font-size: 1.05rem; font-weight: 700; margin-top: 2px; }
.pos { color: #3fb950; }
.neg { color: #f85149; }
.neu { color: #e6edf3; }
.table-wrap { overflow-x: auto; margin-bottom: 8px; }
.table-wrap table { border-collapse: collapse; font-size: 0.77rem; }
.table-wrap th, .table-wrap td {
    padding: 4px 8px; border: 1px solid #30363d;
    text-align: center; white-space: nowrap;
}
.table-wrap th { background: #161b22 !important; color: #e6edf3 !important; font-weight: 600; }
.table-wrap td { background: #0d1117; color: #e6edf3; }
.note { font-size: 0.71rem; color: #8b949e; line-height: 1.8; padding-top: 6px; border-top: 1px solid #30363d; }
"""

_TABLE_NOTES = [
    "<b>Win Rate</b> — fraction of trades that closed with a positive return.",
    "<b>Expectancy</b> — win_rate × avg_win + (1 − win_rate) × avg_loss; expected return per bet.",
    "<b>Exp ex-Top</b> — expectancy after removing the single best trade. Large gap = outlier-driven edge.",
    "<b>Median</b> — 50th-percentile return; Median ≪ Expectancy = right-skewed distribution.",
    "<b>Profit Factor</b> — &Sigma; wins / |&Sigma; losses|; &gt; 1 earns more than it loses. &infin; = no losses.",
    "<b>Skewness</b> — &gt; +1: rare large wins; &lt; &minus;1: rare large losses.",
    "<b>Avg DD / Worst DD</b> — intra-trade peak-to-trough from entry price. Always &le; 0.",
    "<b>Net</b> = MTC fill minus round-trip cost. "
    "<b>Conservative</b> = worst-case fill (high entry / low exit).",
]


def _bg(val: float, bound: float) -> str:
    if pd.isna(val) or bound == 0:
        return ""
    t = max(-1.0, min(1.0, float(val) / bound))
    if t >= 0:
        r = b = round(255 * (1 - 0.75 * t))
        g = round(255 * (1 - 0.15 * t))
    else:
        t = -t
        r = round(255 * (1 - 0.15 * t))
        g = b = round(255 * (1 - 0.75 * t))
    return f"background-color: rgb({r},{g},{b}); color: #111;"


def _bg_red(val: float, bound: float) -> str:
    if pd.isna(val) or bound == 0:
        return ""
    t = min(1.0, abs(float(val) / bound))
    r = round(255 * (1 - 0.15 * t))
    g = b = round(255 * (1 - 0.75 * t))
    return f"background-color: rgb({r},{g},{b}); color: #111;"


class SignalTearsheet:
    """Self-contained HTML tearsheet for a single signal backtest.

    Combines a portfolio summary, per-trade quality table, aligned trade paths,
    and return distribution into a single ``.html`` file that can be opened in
    any browser or shared without Jupyter.

    Args:
        analytics: :class:`~hailmary.analytics.SignalAnalytics` instance.
        title: Heading shown at the top of the report.
        method: Primary fill method — ``"net"`` (default), ``"mtc"``, or
            ``"conservative"``.

    Example::

        from hailmary.viz.signal_tearsheet import SignalTearsheet

        analytics = SignalAnalytics(bt_result)
        SignalTearsheet(analytics, title="MA-200 — BTC/ETH/SOL").save("reports/ma200.html")
    """

    def __init__(
        self,
        analytics: SignalAnalytics,
        title: str = "Signal Tearsheet",
        method: str = "net",
    ) -> None:
        self._a = analytics
        self._title = title
        self._method = method

    # ----------------------------------------------------------------- public

    def save(self, path: str | Path) -> Path:
        """Write the tearsheet to *path* and return the resolved path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.html(), encoding="utf-8")
        return p.resolve()

    def html(self) -> str:
        """Return the full tearsheet as a self-contained HTML string."""
        first: list[bool] = [True]  # mutable flag — first chart includes plotlyjs

        def _chart(fig: go.Figure) -> str:
            h = fig.to_html(full_html=False, include_plotlyjs="cdn" if first[0] else False)
            first[0] = False
            return h

        paths = self._trade_paths()
        ts_net = self._a.trade_summary(method="net")
        ts_con = self._a.trade_summary(method="conservative")

        sections = [
            self._header_html(),
            "<h2>Portfolio Summary</h2>",
            self._metrics_html(),
            _chart(self._equity_fig()),
            "<h2>Per-Trade Quality</h2>",
            self._quality_table_html(ts_net, ts_con),
            "<h2>Aligned Trade Paths — Net Fill</h2>",
            _chart(self._paths_fig(paths, "cum_net", "Aligned Trade Paths — Net Fill",
                                   _P["accent_yellow"], "rgba(255,215,0,0.10)")),
            "<h2>Aligned Trade Paths — Conservative Fill</h2>",
            _chart(self._paths_fig(paths, "cum_con", "Aligned Trade Paths — Conservative Fill",
                                   _P["accent_orange"], "rgba(255,140,0,0.10)")),
            "<h2>Return Distribution</h2>",
            _chart(self._distribution_fig(ts_net, ts_con)),
        ]

        body = "\n".join(sections)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{self._title}</title>\n"
            f"<style>{_CSS}</style>\n"
            f"</head>\n<body>\n{body}\n</body>\n</html>"
        )

    # ----------------------------------------------------------------- sections

    def _header_html(self) -> str:
        data = self._a._data
        symbols = sorted(data.index.get_level_values("symbol").unique())
        dates = data.index.get_level_values("timestamp")
        return (
            f"<h1>{self._title}</h1>\n"
            f'<div class="meta">'
            f"{', '.join(symbols)}"
            f" &nbsp;|&nbsp; {dates.min().date()} &rarr; {dates.max().date()}"
            f" &nbsp;|&nbsp; fill: {self._method}"
            f"</div>"
        )

    def _metrics_html(self) -> str:
        pm = self._a.portfolio_metrics(method=self._method)
        ts = self._a.trade_summary(method=self._method)

        def _card(label: str, value: str, cls: str = "neu") -> str:
            return (
                f'<div class="card">'
                f'<div class="card-label">{label}</div>'
                f'<div class="card-value {cls}">{value}</div>'
                f"</div>"
            )

        def _cls(v: float) -> str:
            return "pos" if v > 0 else ("neg" if v < 0 else "neu")

        total_ret = float(self._a.portfolio_equity(method=self._method).iloc[-1]) - 1
        avg_exp = float(ts["expectancy"].mean())
        avg_wr = float(ts["win_rate"].mean())

        cards = [
            _card("Total Return", f"{total_ret:+.1%}", _cls(total_ret)),
            _card("CAGR", f"{pm.cagr:+.1%}", _cls(pm.cagr)),
            _card("Ann. Vol", f"{pm.annualised_vol:.1%}", "neu"),
            _card("Sharpe", f"{pm.sharpe:.2f}", _cls(pm.sharpe)),
            _card("Sortino", f"{pm.sortino:.2f}", _cls(pm.sortino)),
            _card("Calmar", f"{pm.calmar:.2f}", _cls(pm.calmar)),
            _card("Max Drawdown", f"{pm.max_drawdown:.1%}", "neg" if pm.max_drawdown < 0 else "neu"),
            _card("Avg Drawdown", f"{pm.avg_drawdown:.1%}", "neg" if pm.avg_drawdown < 0 else "neu"),
            _card("Win Rate", f"{avg_wr:.1%}", _cls(avg_wr - 0.5)),
            _card("Expectancy", f"{avg_exp:+.1%}", _cls(avg_exp)),
        ]
        return f'<div class="cards">{"".join(cards)}</div>'

    def _quality_table_html(
        self, ts_net: pd.DataFrame, ts_con: pd.DataFrame
    ) -> str:
        _METRICS = [
            "win_rate", "avg_win", "avg_loss", "expectancy", "expectancy_ex_top",
            "median_return", "profit_factor", "skewness", "avg_intra_drawdown",
            "max_intra_drawdown",
        ]
        df = pd.concat([
            ts_net[["n_trades"]],
            ts_net[_METRICS],
            ts_con[_METRICS].rename(columns=lambda c: f"{c}_con"),
            ts_net[["avg_duration"]],
        ], axis=1)

        _cols = ["Win Rate", "Avg Win", "Avg Loss", "Expectancy", "Exp ex-Top",
                 "Median", "Profit Factor", "Skewness", "Avg DD", "Worst DD"]
        df.columns = pd.MultiIndex.from_tuples(
            [("Bets", "Count")]
            + [("Net", c) for c in _cols]
            + [("Conservative", c) for c in _cols]
            + [("Duration", "Avg (bars)")]
        )
        df[("", "Flags")] = [self._outlier_flag(df.loc[sym]) for sym in df.index]

        _pct, _pct0, _f2 = "{:+.1%}", "{:.1%}", "{:.2f}"
        fmt = {
            ("Bets", "Count"): "{:.0f}",
            ("Duration", "Avg (bars)"): "{:.0f}",
            **{("Net", c): _pct for c in ["Avg Win", "Avg Loss", "Expectancy", "Exp ex-Top", "Median", "Avg DD", "Worst DD"]},
            **{("Conservative", c): _pct for c in ["Avg Win", "Avg Loss", "Expectancy", "Exp ex-Top", "Median", "Avg DD", "Worst DD"]},
            ("Net", "Win Rate"): _pct0, ("Conservative", "Win Rate"): _pct0,
            ("Net", "Profit Factor"): _f2, ("Conservative", "Profit Factor"): _f2,
            ("Net", "Skewness"): "{:+.2f}", ("Conservative", "Skewness"): "{:+.2f}",
        }

        styler = df.style.format(fmt)

        exp_bound = max(
            abs(float(df[[("Net", "Expectancy"), ("Conservative", "Expectancy")]].min().min())),
            abs(float(df[[("Net", "Expectancy"), ("Conservative", "Expectancy")]].max().max())),
            0.001,
        )
        for col in [("Net", "Expectancy"), ("Net", "Exp ex-Top"), ("Net", "Median"),
                    ("Conservative", "Expectancy"), ("Conservative", "Exp ex-Top"),
                    ("Conservative", "Median")]:
            styler = styler.map(lambda v, b=exp_bound: _bg(v, b), subset=[col])

        for col in [("Net", "Win Rate"), ("Conservative", "Win Rate")]:
            styler = styler.map(
                lambda v: _bg(float(v) - 0.5, 0.5) if not pd.isna(v) else "",
                subset=[col],
            )

        for col in [("Net", "Profit Factor"), ("Conservative", "Profit Factor")]:
            styler = styler.map(
                lambda v: _bg(min(float(v), 3.0) - 1.0, 2.0)
                if not pd.isna(v) and v != float("inf") else "",
                subset=[col],
            )

        dd_bound = max(
            abs(float(df[[("Net", "Worst DD"), ("Conservative", "Worst DD")]].min().min())),
            0.001,
        )
        for col in [("Net", "Avg DD"), ("Net", "Worst DD"),
                    ("Conservative", "Avg DD"), ("Conservative", "Worst DD")]:
            styler = styler.map(lambda v, b=dd_bound: _bg_red(v, b), subset=[col])

        _border = "border-left: 2px solid #555 !important;"
        styler = styler.set_table_styles(
            [{"selector": sel, "props": _border} for sel in [
                "th.col_heading.level0.col1",  "th.col_heading.level1.col1",  "td.col1",
                "th.col_heading.level0.col11", "th.col_heading.level1.col11", "td.col11",
                "th.col_heading.level0.col21", "th.col_heading.level1.col21", "td.col21",
                "th.col_heading.level0.col22", "th.col_heading.level1.col22", "td.col22",
            ]],
            overwrite=False,
        )

        note_html = '<div class="note">' + "<br>".join(_TABLE_NOTES) + "</div>"
        return f'<div class="table-wrap">{styler.to_html()}{note_html}</div>'

    # ----------------------------------------------------------------- charts

    def _equity_fig(self) -> go.Figure:
        eq = self._a.portfolio_equity(method=self._method)
        pm = self._a.portfolio_metrics(method=self._method)
        dd = pm.drawdown_series

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.04,
            shared_xaxes=True,
        )

        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            mode="lines", line=dict(color=_P["accent_blue"], width=2),
            name="Portfolio NAV",
            hovertemplate="%{x|%Y-%m-%d}<br>NAV: <b>%{y:.3f}</b><extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", line=dict(color=_P["accent_red"], width=1.5),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.15)",
            name="Drawdown",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: <b>%{y:.1%}</b><extra></extra>",
        ), row=2, col=1)

        fig.add_hline(y=1, row=1, col=1,
                      line=dict(color=_P["text_secondary"], width=0.8, dash="dash"))
        fig.add_hline(y=0, row=2, col=1,
                      line=dict(color=_P["text_secondary"], width=0.8, dash="dash"))

        apply_theme(fig, title="", height=400)
        fig.update_layout(
            showlegend=False,
            margin=dict(t=10),
            yaxis=dict(title_text="NAV"),
            yaxis2=dict(title_text="Drawdown", tickformat=".0%"),
        )
        return fig

    def _paths_fig(
        self,
        paths: list[dict],
        cum_key: str,
        title: str,
        avg_colour: str,
        band_rgba: str,
    ) -> go.Figure:
        fig = go.Figure()
        shown: set[str] = set()

        for t in paths:
            cum = t[cum_key]
            win = float(cum.iloc[-1]) > 0
            colour = _P["accent_green"] if win else _P["accent_red"]
            label = "Win" if win else "Loss"
            show = label not in shown
            if show:
                shown.add(label)
            n = len(cum)
            custom = [[
                t["label"],
                str(t["entry_dt"].date()),
                str(t["exit_dt"].date()),
                t["duration"],
                f"{t['final_net']:+.1%}",
                f"{t['final_con']:+.1%}",
                f"{t['max_dd_net']:.1%}",
                f"{t['max_dd_con']:.1%}",
            ]] * n
            fig.add_trace(go.Scatter(
                x=list(range(n)),
                y=(cum * 100).tolist(),
                mode="lines",
                line=dict(color=colour, width=1.5),
                opacity=0.55,
                name=label, legendgroup=label, showlegend=show,
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Entry: %{customdata[1]}  &rarr;  Exit: %{customdata[2]}<br>"
                    "Duration: %{customdata[3]} bars<br>"
                    "Day %{x}: <b>%{y:.1f}%</b><br>"
                    "Final — net: %{customdata[4]}  |  conservative: %{customdata[5]}<br>"
                    "Max DD — net: %{customdata[6]}  |  conservative: %{customdata[7]}"
                    "<extra></extra>"
                ),
            ))

        max_dur = max(len(t[cum_key]) for t in paths)
        avgs, meds, stds = [], [], []
        for d in range(max_dur):
            vals = [float(t[cum_key].iloc[d]) * 100 for t in paths if d < len(t[cum_key])]
            avgs.append(np.mean(vals))
            meds.append(np.median(vals))
            stds.append(np.std(vals))

        days = list(range(max_dur))
        upper = [m + s for m, s in zip(avgs, stds)]
        lower = [m - s for m, s in zip(avgs, stds)]

        fig.add_trace(go.Scatter(
            x=days + days[::-1], y=upper + lower[::-1],
            fill="toself", fillcolor=band_rgba,
            line=dict(color="rgba(0,0,0,0)"),
            name="±1σ", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=days, y=avgs, mode="lines",
            line=dict(color=avg_colour, width=2.5), name="Mean",
            hovertemplate="Day %{x}<br>Mean: <b>%{y:.1f}%</b><extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=days, y=meds, mode="lines",
            line=dict(color=_P["accent_purple"], width=2, dash="dash"), name="Median",
            hovertemplate="Day %{x}<br>Median: <b>%{y:.1f}%</b><extra></extra>",
        ))

        fig.add_hline(y=0, line=dict(color=_P["text_secondary"], width=0.8, dash="dash"))
        apply_theme(fig, title=title, height=460)
        fig.update_layout(
            xaxis_title="Bars since entry",
            yaxis_title="Cumulative return from entry (%)",
        )
        return fig

    def _distribution_fig(
        self, ts_net: pd.DataFrame, ts_con: pd.DataFrame
    ) -> go.Figure:
        trades = self._a.trade_stats()
        symbols = sorted(trades["symbol"].unique())
        n_sym = len(symbols)

        fig = make_subplots(
            rows=1, cols=n_sym,
            subplot_titles=symbols,
            shared_yaxes=True,
            horizontal_spacing=0.06,
        )

        for colour, dash, label in [
            (_P["accent_yellow"], "solid", "Expectancy — Net"),
            (_P["accent_orange"], "dot",   "Expectancy — Conservative"),
            (_P["accent_purple"], "dash",  "Median — Net"),
        ]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=colour, width=2, dash=dash), name=label,
            ))

        shown: set[str] = set()
        for col_i, sym in enumerate(symbols, start=1):
            ret_net = trades[trades["symbol"] == sym]["return_net"]

            for data, colour, label in [
                (ret_net[ret_net <= 0], _P["accent_red"],   "Loss"),
                (ret_net[ret_net >  0], _P["accent_green"], "Win"),
            ]:
                show = label not in shown
                if show:
                    shown.add(label)
                fig.add_trace(go.Histogram(
                    x=data, name=label, legendgroup=label, showlegend=show,
                    marker_color=colour, opacity=0.8, nbinsx=8,
                ), row=1, col=col_i)

            fig.add_vline(x=0,
                          line_dash="dash", line_color=_P["text_secondary"], line_width=1,
                          row=1, col=col_i)
            fig.add_vline(x=float(ts_net.loc[sym, "expectancy"]),
                          line_dash="solid", line_color=_P["accent_yellow"], line_width=2,
                          row=1, col=col_i)
            fig.add_vline(x=float(ts_con.loc[sym, "expectancy"]),
                          line_dash="dot", line_color=_P["accent_orange"], line_width=2,
                          row=1, col=col_i)
            fig.add_vline(x=float(ts_net.loc[sym, "median_return"]),
                          line_dash="dash", line_color=_P["accent_purple"], line_width=2,
                          row=1, col=col_i)

        fig.update_layout(barmode="overlay")
        for i in range(1, n_sym + 1):
            axis = f"xaxis{'' if i == 1 else i}"
            fig.update_layout(**{axis: dict(tickformat=".0%", title_text="Return per trade")})
        fig.update_layout(yaxis_title="# Trades")
        apply_theme(fig, title="Trade Return Distribution — Net Fill", height=400)
        return fig

    # ----------------------------------------------------------------- helpers

    def _trade_paths(self) -> list[dict]:
        df = self._a._data
        trade_stats = self._a.trade_stats()
        paths: list[dict] = []
        sym_counters: dict[str, int] = {}
        for (sym, trade_id), group in df[df["cycle"] != "None"].groupby(
            ["symbol", "trade_cycle_id"]
        ):
            sym_counters[sym] = sym_counters.get(sym, 0) + 1
            label = f"{sym} T{sym_counters[sym]}"
            rets_net = group["return_net"].reset_index(drop=True)
            rets_con = group["return_conservative"].reset_index(drop=True)
            cum_net = pd.concat([pd.Series([0.0]), (1 + rets_net).cumprod() - 1]).reset_index(drop=True)
            cum_con = pd.concat([pd.Series([0.0]), (1 + rets_con).cumprod() - 1]).reset_index(drop=True)
            entry_dt = group.index.get_level_values("timestamp")[0]
            exit_dt = group.index.get_level_values("timestamp")[-1]
            ts_row = trade_stats[
                (trade_stats["symbol"] == sym) & (trade_stats["entry_date"] == entry_dt)
            ]
            paths.append({
                "label":      label,
                "sym":        sym,
                "cum_net":    cum_net,
                "cum_con":    cum_con,
                "entry_dt":   entry_dt,
                "exit_dt":    exit_dt,
                "duration":   len(rets_net),
                "final_net":  float(cum_net.iloc[-1]),
                "final_con":  float(cum_con.iloc[-1]),
                "max_dd_net": float(ts_row["max_intra_drawdown_net"].iloc[0]) if len(ts_row) else float("nan"),
                "max_dd_con": float(ts_row["max_intra_drawdown_conservative"].iloc[0]) if len(ts_row) else float("nan"),
            })
        return paths

    @staticmethod
    def _outlier_flag(row: pd.Series) -> str:
        exp    = row[("Net", "Expectancy")]
        ex_top = row[("Net", "Exp ex-Top")]
        med    = row[("Net", "Median")]
        skew   = row[("Net", "Skewness")]
        flags: list[str] = []
        if exp > 0 and med < 0:
            flags.append("⚠ median < 0")
        if exp != 0 and abs(exp - ex_top) / abs(exp) > 0.3:
            flags.append("⚠ top trade >30% of edge")
        if skew > 1.5:
            flags.append("↑ skewed right")
        elif skew < -1.5:
            flags.append("↓ skewed left")
        return ", ".join(flags) if flags else "—"
