"""Standalone HTML tearsheet for a single signal backtest."""

from __future__ import annotations

import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hailmary.analytics.signal_analytics import SignalAnalytics, TradeQuality
from hailmary.viz.theme import PALETTE, apply_theme

_P = PALETTE

_METHOD_COLOUR: dict[str, str] = {
    "net":          _P["accent_blue"],
    "mtc":          _P["accent_purple"],
    "conservative": _P["accent_orange"],
}
_METHOD_FILL: dict[str, str] = {
    "net":          "rgba(88,166,255,0.12)",
    "mtc":          "rgba(188,140,255,0.12)",
    "conservative": "rgba(255,166,0,0.12)",
}
_METHOD_LABEL: dict[str, str] = {
    "net":          "Net",
    "mtc":          "MTC",
    "conservative": "Conservative",
}
_MODE_LABEL: dict[str, str] = {
    "rebalanced":   "equal-weight, rebalanced daily",
    "buy_and_hold": "equal-weight, buy & hold",
    "fixed_stake":  "fixed stake per entry",
}

_CSS = """
*, *::before, *::after { box-sizing: border-box; }
body {
    background: #0d1117; color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 13px; line-height: 1.5;
    margin: 0; padding: 0 0 56px; max-width: 1600px;
}
.sticky-header {
    position: sticky; top: 0; z-index: 100;
    background: rgba(13,17,23,0.96); backdrop-filter: blur(8px);
    border-bottom: 1px solid #30363d; padding: 12px 40px 10px;
}
.main-content { padding: 4px 40px 0; }
h1 { font-size: 1.35rem; font-weight: 600; margin: 0 0 2px; }
h2 {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #8b949e;
    margin: 36px 0 14px; padding-bottom: 6px; border-bottom: 1px solid #30363d;
}
.meta { font-size: 0.77rem; color: #8b949e; }
.header-bar {
    display: flex; align-items: flex-start;
    justify-content: space-between; flex-wrap: wrap; gap: 10px;
}
.method-toggle { display: flex; gap: 4px; align-items: center; padding-top: 2px; }
.toggle-lbl { font-size: 0.68rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.07em; margin-right: 2px; }
.method-btn {
    padding: 4px 14px; font-size: 0.72rem; font-weight: 600; cursor: pointer;
    background: #161b22; border: 1px solid #30363d; border-radius: 4px;
    color: #8b949e; transition: all 0.15s;
}
.method-btn:hover  { background: #21262d; color: #e6edf3; }
.method-btn.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
nav { display: flex; gap: 22px; margin-top: 8px; padding-top: 8px; border-top: 1px solid #21262d; }
nav a { color: #58a6ff; text-decoration: none; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em; }
nav a:hover { color: #e6edf3; text-decoration: underline; }
.section-tag {
    display: inline-block; font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; padding: 1px 7px; border-radius: 3px;
}
.tag-net          { background: rgba(88,166,255,0.15); color: #58a6ff; }
.tag-conservative { background: rgba(255,166,0,0.15);  color: #ffa600; }
.tag-mtc          { background: rgba(188,140,255,0.15); color: #bc8cff; }
.section-tag-block { display: inline-block; margin-bottom: 8px; }
.f-bar { display: flex; align-items: center; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }
.qf-btn {
    padding: 3px 11px; font-size: 0.70rem; font-weight: 600; cursor: pointer;
    background: #161b22; border: 1px solid #30363d; border-radius: 4px;
    color: #8b949e; transition: all 0.15s;
}
.qf-btn:hover  { background: #21262d; color: #e6edf3; }
.qf-btn.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
.table-wrap { overflow-x: auto; margin-bottom: 8px; }
.table-wrap table, .t-tbl { border-collapse: collapse; font-size: 0.77rem; }
.table-wrap th, .table-wrap td, .t-tbl th, .t-tbl td {
    padding: 4px 9px; border: 1px solid #30363d;
    text-align: center; white-space: nowrap;
}
.table-wrap th, .t-tbl th { background: #161b22; color: #e6edf3; font-weight: 600; }
.table-wrap td, .t-tbl td { background: #0d1117; color: #e6edf3; }
.t-tbl .win-row  td { background: rgba(63,185,80,0.07); }
.t-tbl .loss-row td { background: rgba(248,81,73,0.07); }
.note { font-size: 0.71rem; color: #8b949e; line-height: 1.8; padding-top: 6px; border-top: 1px solid #30363d; }
.trade-group-hdr {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; padding: 10px 0 5px;
}
.muted { font-size: 0.65rem; color: #8b949e; }
"""

_JS = """
function setMethod(m, btn) {
    document.querySelectorAll('.method-btn').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
    document.querySelectorAll('[data-method]').forEach(function(el) {
        el.style.display = (m === 'both' || el.dataset.method === m) ? '' : 'none';
    });
}
var _qMethod = 'all', _qSymbol = 'all';
function filterQuality(dim, val, btn) {
    var cls = dim === 'method' ? '.qf-method' : '.qf-symbol';
    document.querySelectorAll(cls).forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
    if (dim === 'method') { _qMethod = val; } else { _qSymbol = val; }
    document.querySelectorAll('#quality-tbl tr[data-method]').forEach(function(row) {
        var okM = _qMethod === 'all' || row.dataset.method === _qMethod;
        var okS = _qSymbol === 'all' || row.dataset.symbol === _qSymbol;
        row.style.display = (okM && okS) ? '' : 'none';
    });
}
function filterTL(sym, btn) {
    document.querySelectorAll('.tl-sym').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
    document.querySelectorAll('.trade-row').forEach(function(row) {
        row.style.display = (sym === 'all' || row.dataset.symbol === sym) ? '' : 'none';
    });
}
function filterTiming(sym, btn) {
    document.querySelectorAll('.timing-sym').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
    document.querySelectorAll('[data-timing-sym]').forEach(function(el) {
        el.style.display = (sym === 'all' || el.dataset.timingSym === sym) ? '' : 'none';
    });
}
function filterDist(sym, btn) {
    document.querySelectorAll('.dist-sym').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
    document.querySelectorAll('[data-sym-filter]').forEach(function(el) {
        el.style.display = (sym === 'all' || el.dataset.symFilter === sym) ? '' : 'none';
    });
}
"""

_TABLE_NOTES = [
    "<b>Win Rate</b> — fraction of trades that closed with a positive return.",
    "<b>Expectancy</b> — win_rate &times; avg_win + (1 &minus; win_rate) &times; avg_loss; expected return per bet.",
    "<b>Exp ex-Top</b> — expectancy after removing the single best trade. Large gap = outlier-driven edge.",
    "<b>Median</b> — 50th-percentile return; Median &laquo; Expectancy = right-skewed distribution.",
    "<b>Profit Factor</b> — &Sigma; wins / |&Sigma; losses|; &gt; 1 earns more than it loses.",
    "<b>Skewness</b> — &gt; +1: rare large wins; &lt; &minus;1: rare large losses.",
    "<b>Avg DD / Worst DD</b> — intra-trade peak-to-trough from entry price. Always &le; 0.",
    "<b>WR@5d</b> — win rate at bar 5 (trades &ge;5 bars only, i.e. reached bar 5); <b>5d&rarr;Exit WR</b> — win rate from bar 5 to close (trades &gt;5 bars only).",
    "<b>Conservative fill flags</b>: <i>edge reversed vs net</i> — conservative expectancy &le; 0 while net &gt; 0; <i>fill halves edge</i> — conservative &lt; 50% of net expectancy; <i>median flips negative</i> — net median &ge; 0 but conservative &lt; 0.",
    "<b>Net</b> = MTC fill minus round-trip cost.  <b>Conservative</b> = high entry / low exit fill.",
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
    return f"background-color:rgb({r},{g},{b});color:#111;"


def _bg_red(val: float, bound: float) -> str:
    if pd.isna(val) or bound == 0:
        return ""
    t = min(1.0, abs(float(val) / bound))
    r = round(255 * (1 - 0.15 * t))
    g = b = round(255 * (1 - 0.75 * t))
    return f"background-color:rgb({r},{g},{b});color:#111;"


class SignalTearsheet:
    """Signal-evaluation HTML tearsheet focused on per-trade quality.

    Sections: Per-Trade Quality → Entry Timing → Trade Log →
    Aligned Trade Paths → Return Distribution.

    Args:
        analytics: :class:`~hailmary.analytics.SignalAnalytics` instance.
        title: Heading shown at the top.
        method: Fill method(s) — string or list, default ``["net", "conservative"]``.
        mode: Capital model (kept for API compatibility; not used in this sheet).
        amount_per_entry: Only relevant for ``mode="fixed_stake"``.

    Example::

        SignalTearsheet(analytics, title="MA-200 — BTC/ETH/SOL").save("ma200.html", open=True)
    """

    def __init__(
        self,
        analytics: SignalAnalytics,
        title: str = "Signal Tearsheet",
        method: str | list[str] = ("net", "conservative"),
        mode: str = "rebalanced",
        amount_per_entry: float = 1_000.0,
    ) -> None:
        self._a = analytics
        self._title = title
        self._methods: list[str] = [method] if isinstance(method, str) else list(method)
        self._mode = mode
        self._amount_per_entry = amount_per_entry

    # ----------------------------------------------------------------- public

    def save(self, path: str | Path, open: bool = False) -> Path:
        """Write the tearsheet to *path* and return the resolved path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.html(), encoding="utf-8")
        resolved = p.resolve()
        if open:
            webbrowser.open(resolved.as_uri())
        return resolved

    def html(self) -> str:
        """Return the full tearsheet as a self-contained HTML string."""
        def _chart(fig: go.Figure) -> str:
            return fig.to_html(full_html=False, include_plotlyjs=False)

        paths  = TradeQuality(self._a).trade_paths()
        ts_net = self._a.trade_summary(method="net")
        ts_con = self._a.trade_summary(method="conservative")

        data    = self._a._data
        symbols = sorted(data.index.get_level_values("symbol").unique())
        dates   = data.index.get_level_values("timestamp")

        meta = (
            f"{', '.join(symbols)}"
            f" &nbsp;|&nbsp; {dates.min().date()} &rarr; {dates.max().date()}"
        )

        toggle_html = ""
        if len(self._methods) > 1:
            btns = [
                '<span class="toggle-lbl">Fill:</span>',
                '<button class="method-btn active" onclick="setMethod(\'both\', this)">Both</button>',
            ]
            for m in self._methods:
                lbl = _METHOD_LABEL.get(m, m)
                btns.append(f'<button class="method-btn" onclick="setMethod(\'{m}\', this)">{lbl}</button>')
            toggle_html = f'<div class="method-toggle">{"".join(btns)}</div>'

        sticky = (
            '<div class="sticky-header">'
            '<div class="header-bar">'
            f'<div><h1>{self._title}</h1><div class="meta">{meta}</div></div>'
            f'{toggle_html}'
            '</div>'
            '<nav>'
            '<a href="#quality">Trade Quality</a>'
            '<a href="#timing">Entry Timing</a>'
            '<a href="#tradelog">Trade Log</a>'
            '<a href="#paths">Trade Paths</a>'
            '<a href="#distribution">Distribution</a>'
            '</nav>'
            '</div>'
        )

        # ---- helpers for symbol-filter bars ----
        def _sym_filter_bar(btn_cls: str, onclick_fn: str) -> str:
            btns = [
                f'<span class="toggle-lbl">Symbol:</span>',
                f'<button class="qf-btn {btn_cls} active" onclick="{onclick_fn}(\'all\', this)">All</button>',
            ]
            for sym in symbols:
                btns.append(
                    f'<button class="qf-btn {btn_cls}" onclick="{onclick_fn}(\'{sym}\', this)">{sym}</button>'
                )
            return f'<div class="f-bar">{"".join(btns)}</div>'

        # ---- trade paths: method divs, symbol filter inside figure ----
        path_sections: list[str] = []
        for m in self._methods:
            cum_key    = "cum_net" if m == "net" else "cum_con"
            avg_colour = _P["accent_yellow"] if m == "net" else _P["accent_orange"]
            band       = "rgba(255,215,0,0.10)" if m == "net" else "rgba(255,140,0,0.10)"
            lbl = _METHOD_LABEL.get(m, m)
            tag = f'<div class="section-tag-block"><span class="section-tag tag-{m}">{lbl}</span></div>'
            path_sections.append(
                f'<div data-method="{m}">{tag}'
                f'{_chart(self._paths_fig(paths, cum_key, f"Aligned Trade Paths — {lbl} Fill", avg_colour, band))}'
                f'</div>'
            )

        # ---- timing: method + sym filter divs ----
        timing_sym_divs: list[str] = []
        for sym_filter, sym_list in [("all", None)] + [(s, [s]) for s in symbols]:
            method_charts: list[str] = []
            for m in self._methods:
                lbl = _METHOD_LABEL.get(m, m)
                tag = f'<div class="section-tag-block"><span class="section-tag tag-{m}">{lbl}</span></div>'
                method_charts.append(
                    f'<div data-method="{m}">{tag}'
                    f'{_chart(self._timing_fig(paths, m, None if sym_list is None else sym_list))}'
                    f'</div>'
                )
            hidden = '' if sym_filter == 'all' else ' style="display:none"'
            timing_sym_divs.append(
                f'<div data-timing-sym="{sym_filter}"{hidden}>{"".join(method_charts)}</div>'
            )

        # ---- distribution: method + sym filter divs ----
        dist_sym_divs: list[str] = []
        for sym_filter, sym_list in [("all", None)] + [(s, [s]) for s in symbols]:
            method_charts = []
            for m in self._methods:
                lbl = _METHOD_LABEL.get(m, m)
                tag = f'<div class="section-tag-block"><span class="section-tag tag-{m}">{lbl}</span></div>'
                method_charts.append(
                    f'<div data-method="{m}">{tag}'
                    f'{_chart(self._distribution_fig(m, ts_net, ts_con, sym_list))}'
                    f'</div>'
                )
            hidden = '' if sym_filter == 'all' else ' style="display:none"'
            dist_sym_divs.append(
                f'<div data-sym-filter="{sym_filter}"{hidden}>{"".join(method_charts)}</div>'
            )

        main = "\n".join([
            '<div class="main-content">',
            '<section id="quality"><h2>Per-Trade Quality</h2>',
            self._quality_table_html(ts_net, ts_con, paths),
            '</section>',
            '<section id="timing"><h2>Entry Timing</h2>',
            _sym_filter_bar("timing-sym", "filterTiming"),
            "\n".join(timing_sym_divs),
            '</section>',
            '<section id="tradelog"><h2>Trade Log</h2>',
            self._trade_log_html(paths, symbols),
            '</section>',
            '<section id="paths"><h2>Aligned Trade Paths</h2>',
            "\n".join(path_sections),
            '</section>',
            '<section id="distribution"><h2>Return Distribution</h2>',
            _sym_filter_bar("dist-sym", "filterDist"),
            "\n".join(dist_sym_divs),
            '</section>',
            '</div>',
        ])

        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{self._title}</title>\n'
            f'<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
            f'<style>{_CSS}</style>\n'
            f'</head>\n<body>\n{sticky}\n{main}\n'
            f'<script>{_JS}</script>\n'
            f'</body>\n</html>'
        )

    # ----------------------------------------------------------------- sections

    def _quality_table_html(
        self, ts_net: pd.DataFrame, ts_con: pd.DataFrame, paths: list[dict]
    ) -> str:
        _METRICS = [
            "win_rate", "avg_win", "avg_loss", "expectancy", "expectancy_ex_top",
            "median_return", "profit_factor", "skewness", "avg_intra_drawdown",
            "max_intra_drawdown",
        ]
        symbols = sorted(ts_net.index.tolist())

        rows: list[dict] = []
        for sym in symbols:
            for method_key, ts in [("net", ts_net), ("conservative", ts_con)]:
                r: dict = {
                    "symbol":     sym,
                    "method_key": method_key,
                    "method_lbl": _METHOD_LABEL.get(method_key, method_key),
                    "n_trades":   int(ts.loc[sym, "n_trades"]),
                    "avg_duration": float(ts.loc[sym, "avg_duration"]),
                }
                for m in _METRICS:
                    r[m] = float(ts.loc[sym, m])
                flags = TradeQuality.quality_flags(
                    r["expectancy"], r["expectancy_ex_top"],
                    r["median_return"], r["skewness"],
                    net_expectancy=float(ts_net.loc[sym, "expectancy"]) if method_key == "conservative" else None,
                    net_median=float(ts_net.loc[sym, "median_return"]) if method_key == "conservative" else None,
                )
                r["flags"] = self._outlier_flag_html(flags)
                r.update(TradeQuality.d5_stats(sym, method_key, paths))
                rows.append(r)

        exp_bound = max(
            max(abs(r["expectancy"])        for r in rows),
            max(abs(r["expectancy_ex_top"]) for r in rows),
            max(abs(r["median_return"])     for r in rows),
            0.001,
        )
        dd_bound = max(max(abs(r["max_intra_drawdown"]) for r in rows), 0.001)

        method_btns = [
            '<span class="toggle-lbl">Fill:</span>',
            '<button class="qf-btn qf-method active" onclick="filterQuality(\'method\',\'all\',this)">All</button>',
            '<button class="qf-btn qf-method" onclick="filterQuality(\'method\',\'net\',this)">Net</button>',
            '<button class="qf-btn qf-method" onclick="filterQuality(\'method\',\'conservative\',this)">Conservative</button>',
        ]
        sym_btns = [
            '<span class="toggle-lbl" style="margin-left:14px">Symbol:</span>',
            '<button class="qf-btn qf-symbol active" onclick="filterQuality(\'symbol\',\'all\',this)">All</button>',
        ]
        for sym in symbols:
            sym_btns.append(
                f'<button class="qf-btn qf-symbol" onclick="filterQuality(\'symbol\',\'{sym}\',this)">{sym}</button>'
            )
        filter_html = f'<div class="f-bar">{"".join(method_btns)}{"".join(sym_btns)}</div>'

        header_html = (
            "<tr>"
            "<th>Symbol</th><th>Fill</th><th>N</th>"
            "<th>Win Rate</th><th>Avg Win</th><th>Avg Loss</th>"
            "<th>Expectancy</th><th>Exp ex-Top</th><th>Median</th>"
            "<th>PF</th><th>Skew</th>"
            "<th>Avg DD</th><th>Worst DD</th>"
            "<th>Dur</th><th>Flags</th>"
            "<th>WR@5d</th><th>5d&rarr;Exit WR</th>"
            "</tr>"
        )

        def _td(content: str, style: str = "") -> str:
            s = f' style="{style}"' if style else ""
            return f"<td{s}>{content}</td>"

        body_rows: list[str] = []
        prev_sym = None
        for r in rows:
            sym        = r["symbol"]
            method_key = r["method_key"]
            bt         = "border-top:2px solid #444;" if sym != prev_sym else ""
            prev_sym   = sym
            fill_tag   = f'<span class="section-tag tag-{method_key}">{r["method_lbl"]}</span>'
            pf_val     = r["profit_factor"]
            pf_fmt     = "&#x221e;" if pf_val == float("inf") else f"{pf_val:.2f}"

            wr_d5    = r["wr_d5"]
            wr_tail  = r["wr_tail"]
            n_d5     = r["wr_d5_n"]
            n_win_d5 = r["wr_d5_nwin"]
            n_tail   = r["wr_tail_n"]
            n_win_t  = r["wr_tail_nwin"]

            wr_d5_fmt   = f'{n_win_d5}/{n_d5} ({wr_d5:.0%})'   if not pd.isna(wr_d5)   else "—"
            wr_tail_fmt = (
                f'{n_win_t}/{n_tail} ({wr_tail:.0%})'           if not pd.isna(wr_tail) else
                f'<span class="muted">no trades &gt;5d</span>'
            )

            sep = "border-left:2px solid #555;"
            cells = "".join([
                _td(sym,                               bt),
                _td(fill_tag,                          bt),
                _td(str(r["n_trades"]),                bt),
                _td(f'{r["win_rate"]:.1%}',            _bg(r["win_rate"] - 0.5, 0.5)                            + bt),
                _td(f'{r["avg_win"]:+.1%}',            bt),
                _td(f'{r["avg_loss"]:+.1%}',           bt),
                _td(f'{r["expectancy"]:+.1%}',         _bg(r["expectancy"],        exp_bound)                   + bt),
                _td(f'{r["expectancy_ex_top"]:+.1%}',  _bg(r["expectancy_ex_top"], exp_bound)                   + bt),
                _td(f'{r["median_return"]:+.1%}',      _bg(r["median_return"],     exp_bound)                   + bt),
                _td(pf_fmt,                            (_bg(min(pf_val, 3.0) - 1.0, 2.0) if pf_val != float("inf") else "") + bt),
                _td(f'{r["skewness"]:+.2f}',           bt),
                _td(f'{r["avg_intra_drawdown"]:.1%}',  _bg_red(r["avg_intra_drawdown"], dd_bound)               + bt),
                _td(f'{r["max_intra_drawdown"]:.1%}',  _bg_red(r["max_intra_drawdown"], dd_bound)               + bt),
                _td(f'{r["avg_duration"]:.0f}',        bt),
                _td(r["flags"],                        bt),
                _td(wr_d5_fmt,                         sep + bt),
                _td(wr_tail_fmt,                       bt),
            ])
            body_rows.append(
                f'<tr data-method="{method_key}" data-symbol="{sym}">{cells}</tr>'
            )

        table_html = (
            f'<table id="quality-tbl" style="width:100%;border-collapse:collapse;font-size:0.77rem">'
            f'<thead>{header_html}</thead>'
            f'<tbody>{"".join(body_rows)}</tbody>'
            f'</table>'
        )
        note_html = '<div class="note">' + "<br>".join(_TABLE_NOTES) + "</div>"
        return f'{filter_html}<div class="table-wrap">{table_html}{note_html}</div>'

    def _trade_log_html(self, paths: list[dict], symbols: list[str]) -> str:
        wins   = [t for t in paths if t["final_net"] > 0]
        losses = [t for t in paths if t["final_net"] <= 0]

        sym_btns = [
            '<span class="toggle-lbl">Symbol:</span>',
            '<button class="qf-btn tl-sym active" onclick="filterTL(\'all\', this)">All</button>',
        ]
        for sym in symbols:
            sym_btns.append(
                f'<button class="qf-btn tl-sym" onclick="filterTL(\'{sym}\', this)">{sym}</button>'
            )
        filter_html = f'<div class="f-bar">{"".join(sym_btns)}</div>'

        def _ret_style(v: float) -> str:
            if v > 0:   return ' style="color:#3fb950;font-weight:600"'
            if v < 0:   return ' style="color:#f85149;font-weight:600"'
            return ""

        def _dd_style(v: float) -> str:
            if pd.isna(v): return ""
            t = min(1.0, abs(v) / 0.20)
            r = round(248 * t + 80 * (1 - t))
            g = b = round(20 * (1 - t) + 5)
            return f' style="background:rgb({r},{g},{b});color:#111"'

        hdrs = (
            "<tr><th>Trade</th><th>Entry</th><th>Exit</th><th>Bars</th>"
            "<th>5-Day Net</th><th>5-Day Con</th>"
            "<th>5d&rarr;Exit Net</th><th>5d&rarr;Exit Con</th>"
            "<th>Net</th><th>Conservative</th>"
            "<th>Max DD (Net)</th><th>Max DD (Con)</th></tr>"
        )

        def _row(t: dict, row_cls: str) -> str:
            dur = t["duration"]
            short = f' <span class="muted">({dur}d)</span>' if dur < 5 else ""
            return (
                f'<tr class="trade-row {row_cls}" data-symbol="{t["sym"]}">'
                f'<td>{t["label"]}</td>'
                f'<td>{t["entry_dt"].date()}</td>'
                f'<td>{t["exit_dt"].date()}</td>'
                f'<td>{dur}</td>'
                f'<td{_ret_style(t["d5_net"])}>{t["d5_net"]:+.1%}{short}</td>'
                f'<td{_ret_style(t["d5_con"])}>{t["d5_con"]:+.1%}{short}</td>'
                f'<td{_ret_style(t["d5_to_exit_net"])}>{t["d5_to_exit_net"]:+.1%}{short}</td>'
                f'<td{_ret_style(t["d5_to_exit_con"])}>{t["d5_to_exit_con"]:+.1%}{short}</td>'
                f'<td{_ret_style(t["final_net"])}>{t["final_net"]:+.1%}</td>'
                f'<td{_ret_style(t["final_con"])}>{t["final_con"]:+.1%}</td>'
                f'<td{_dd_style(t["max_dd_net"])}>{t["max_dd_net"]:.1%}</td>'
                f'<td{_dd_style(t["max_dd_con"])}>{t["max_dd_con"]:.1%}</td>'
                f'</tr>'
            )

        parts: list[str] = [filter_html, '<div class="table-wrap">']
        for group, row_cls, icon, clr, label in [
            (wins,   "win-row",  "&#x2714;", "#3fb950", "Wins"),
            (losses, "loss-row", "&#x2718;", "#f85149", "Losses"),
        ]:
            if not group:
                continue
            rows_html = "".join(_row(t, row_cls) for t in group)
            parts.append(
                f'<div class="trade-group-hdr" style="color:{clr}">'
                f'{icon} {label} ({len(group)})</div>'
                f'<table class="t-tbl"><thead>{hdrs}</thead><tbody>{rows_html}</tbody></table>'
            )
        parts.append("</div>")
        return "\n".join(parts)

    # ----------------------------------------------------------------- charts

    def _paths_fig(
        self,
        paths: list[dict],
        cum_key: str,
        title: str,
        avg_colour: str,
        band_rgba: str,
    ) -> go.Figure:
        fig     = go.Figure()
        symbols = sorted(set(t["sym"] for t in paths))
        trace_meta: list[tuple[str, str]] = []
        shown_labels: set[str] = set()

        for t in paths:
            cum    = t[cum_key]
            win    = float(cum.iloc[-1]) > 0
            colour = _P["accent_green"] if win else _P["accent_red"]
            label  = "Win" if win else "Loss"
            show   = label not in shown_labels
            if show:
                shown_labels.add(label)
            n      = len(cum)
            custom = [[
                t["label"], str(t["entry_dt"].date()), str(t["exit_dt"].date()),
                t["duration"],
                f"{t['final_net']:+.1%}", f"{t['final_con']:+.1%}",
                f"{t['max_dd_net']:.1%}", f"{t['max_dd_con']:.1%}",
            ]] * n
            fig.add_trace(go.Scatter(
                x=list(range(n)), y=(cum * 100).tolist(),
                mode="lines", line=dict(color=colour, width=1.5), opacity=0.55,
                name=label, legendgroup=label, showlegend=show,
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Entry: %{customdata[1]} &rarr; Exit: %{customdata[2]}<br>"
                    "Duration: %{customdata[3]} bars | Day %{x}: <b>%{y:.1f}%</b><br>"
                    "Final — net: %{customdata[4]}  |  conservative: %{customdata[5]}<br>"
                    "Max DD — net: %{customdata[6]}  |  conservative: %{customdata[7]}"
                    "<extra></extra>"
                ),
            ))
            trace_meta.append((t["sym"], "trade"))

        def _add_agg(subset: list[dict], group_key: str, visible: bool, show_legend: bool) -> None:
            max_dur = max(len(t[cum_key]) for t in subset)
            avgs, meds, stds = [], [], []
            for d in range(max_dur):
                vals = [float(t[cum_key].iloc[d]) * 100 for t in subset if d < len(t[cum_key])]
                avgs.append(float(np.mean(vals)))
                meds.append(float(np.median(vals)))
                stds.append(float(np.std(vals)))
            days  = list(range(max_dur))
            upper = [a + s for a, s in zip(avgs, stds)]
            lower = [a - s for a, s in zip(avgs, stds)]
            fig.add_trace(go.Scatter(
                x=days + days[::-1], y=upper + lower[::-1],
                fill="toself", fillcolor=band_rgba,
                line=dict(color="rgba(0,0,0,0)"),
                name="±1σ", showlegend=False, hoverinfo="skip", visible=visible,
            ))
            fig.add_trace(go.Scatter(
                x=days, y=avgs, mode="lines",
                line=dict(color=avg_colour, width=2.5),
                name="Mean", showlegend=show_legend, visible=visible,
                hovertemplate="Day %{x} | Mean: <b>%{y:.1f}%</b><extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=days, y=meds, mode="lines",
                line=dict(color=_P["accent_purple"], width=2, dash="dash"),
                name="Median", showlegend=show_legend, visible=visible,
                hovertemplate="Day %{x} | Median: <b>%{y:.1f}%</b><extra></extra>",
            ))
            trace_meta.extend([(group_key, "agg")] * 3)

        _add_agg(paths, "__all__", visible=True, show_legend=True)
        for sym in symbols:
            sym_paths = [p for p in paths if p["sym"] == sym]
            if sym_paths:
                _add_agg(sym_paths, sym, visible=False, show_legend=False)

        all_options: list[tuple[str, str | None]] = [("All", None)] + [(s, s) for s in symbols]
        buttons = []
        for btn_label, filter_sym in all_options:
            vis: list[bool] = []
            for sym, ttype in trace_meta:
                if filter_sym is None:
                    vis.append(ttype == "trade" or sym == "__all__")
                else:
                    vis.append(sym == filter_sym)
            buttons.append(dict(label=btn_label, method="update", args=[{"visible": vis}]))

        fig.update_layout(
            updatemenus=[dict(
                type="buttons", direction="right", buttons=buttons,
                showactive=True,
                x=0.0, xanchor="left", y=1.13, yanchor="top",
                font=dict(size=11), bgcolor="#161b22", bordercolor="#30363d", active=0,
            )]
        )

        # Day-5 marker
        fig.add_vline(
            x=5,
            line=dict(color=_P["text_secondary"], width=1.2, dash="dot"),
            annotation_text="Day 5",
            annotation_position="top right",
            annotation_font=dict(size=10, color=_P["text_secondary"]),
        )
        fig.add_hline(y=0, line=dict(color=_P["text_secondary"], width=0.8, dash="dash"))
        apply_theme(fig, title=title, height=480)
        fig.update_layout(
            margin=dict(t=70),
            xaxis_title="Bars since entry",
            yaxis_title="Cumulative return from entry (%)",
        )
        return fig

    def _timing_fig(
        self,
        paths: list[dict],
        method: str,
        sym_subset: list[str] | None = None,
    ) -> go.Figure:
        """Horizontal stacked bar: first-5-bars return (blue) + 5d→exit return (orange) per trade."""
        d5_col   = "d5_net"         if method == "net" else "d5_con"
        tail_col = "d5_to_exit_net" if method == "net" else "d5_to_exit_con"
        label    = _METHOD_LABEL.get(method, method)

        subset = sorted(
            [p for p in paths if sym_subset is None or p["sym"] in sym_subset],
            key=lambda p: (p["sym"], p["label"]),
        )
        labels    = [p["label"]    for p in subset]
        d5_vals   = [p[d5_col]     for p in subset]
        tail_vals = [p[tail_col]   for p in subset]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            orientation="h", y=labels, x=d5_vals,
            name="First 5 bars",
            marker_color=_P["accent_blue"], opacity=0.85,
            hovertemplate="%{y}<br>First 5 bars: <b>%{x:.1%}</b><extra></extra>",
        ))
        fig.add_trace(go.Bar(
            orientation="h", y=labels, x=tail_vals,
            name="5d → Exit",
            marker_color=_P["accent_orange"], opacity=0.85,
            hovertemplate="%{y}<br>5d → Exit: <b>%{x:.1%}</b><extra></extra>",
        ))

        fig.add_vline(x=0, line=dict(color=_P["text_secondary"], width=0.8, dash="dash"))
        height = max(300, 32 * len(subset) + 80)
        fig.update_layout(barmode="relative")
        apply_theme(fig, title=f"Entry Timing — {label} Fill", height=height)
        fig.update_layout(
            margin=dict(t=30),
            xaxis=dict(tickformat=".0%", title_text="Return"),
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=1.08, x=0),
        )
        return fig

    def _distribution_fig(
        self,
        method: str,
        ts_net: pd.DataFrame,
        ts_con: pd.DataFrame,
        symbols: list[str] | None = None,
    ) -> go.Figure:
        trades   = self._a.trade_stats()
        all_syms = sorted(trades["symbol"].unique())
        syms     = symbols if symbols is not None else all_syms
        n_sym    = len(syms)

        ret_col = "return_conservative" if method == "conservative" else "return_net"
        ts      = ts_con if method == "conservative" else ts_net
        colour  = _METHOD_COLOUR.get(method, _P["accent_yellow"])
        label   = _METHOD_LABEL.get(method, method)

        fig = make_subplots(
            rows=1, cols=n_sym, subplot_titles=syms,
            shared_yaxes=True, horizontal_spacing=0.06,
        )

        for leg_colour, dash, leg_label in [
            (colour,              "solid", f"Expectancy ({label})"),
            (_P["accent_purple"], "dash",  f"Median ({label})"),
        ]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=leg_colour, width=2, dash=dash), name=leg_label,
            ))

        shown: set[str] = set()
        for col_i, sym in enumerate(syms, start=1):
            ret = trades[trades["symbol"] == sym][ret_col]
            for data, bar_colour, bar_label in [
                (ret[ret <= 0], _P["accent_red"],   "Loss"),
                (ret[ret >  0], _P["accent_green"], "Win"),
            ]:
                show = bar_label not in shown
                if show:
                    shown.add(bar_label)
                fig.add_trace(go.Histogram(
                    x=data, name=bar_label, legendgroup=bar_label, showlegend=show,
                    marker_color=bar_colour, opacity=0.8, nbinsx=8,
                ), row=1, col=col_i)

            fig.add_vline(x=0,
                          line_dash="dash", line_color=_P["text_secondary"], line_width=1,
                          row=1, col=col_i)
            fig.add_vline(x=float(ts.loc[sym, "expectancy"]),
                          line_dash="solid", line_color=colour, line_width=2,
                          row=1, col=col_i)
            fig.add_vline(x=float(ts.loc[sym, "median_return"]),
                          line_dash="dash", line_color=_P["accent_purple"], line_width=2,
                          row=1, col=col_i)

        fig.update_layout(barmode="overlay")
        for i in range(1, n_sym + 1):
            axis = f"xaxis{'' if i == 1 else i}"
            fig.update_layout(**{axis: dict(tickformat=".0%", title_text="Return per trade")})
        fig.update_layout(yaxis_title="# Trades")
        apply_theme(fig, title=f"Return Distribution — {label} Fill", height=400)
        return fig

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _outlier_flag_html(flags: dict[str, bool]) -> str:
        """Render a :meth:`TradeQuality.quality_flags` result as an HTML string."""
        parts: list[str] = []
        if flags["median_negative"]:
            parts.append("&#x26a0; median &lt; 0")
        if flags["top_trade_outlier"]:
            parts.append("&#x26a0; top trade &gt;30%")
        if flags["skewed_right"]:
            parts.append("&#x2191; skewed right")
        elif flags["skewed_left"]:
            parts.append("&#x2193; skewed left")
        if flags["edge_reversed"]:
            parts.append("&#x26a0; edge reversed vs net")
        elif flags["fill_halves_edge"]:
            parts.append("&#x26a0; fill halves edge")
        if flags["median_flips"]:
            parts.append("&#x26a0; median flips negative")
        return ", ".join(parts) if parts else "&#x2014;"
