"""ETF Explorer — discovery view over Stashaway's full ETF Explorer offering.

Parses Stashaway's ETF universe xlsx, fetches Yahoo total-return data for each
ticker (Bloomberg → Yahoo notation), computes multi-window metrics (YTD / 1Y /
3Y / 5Y) and correlation with the user's combined book. Renders to a sortable
HTML report.

v1 scope: flat-period metrics. v2 (when the regime classifier lands): add
drawdown-bucket Sharpe and rolling correlation with the user's book.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from hailmary.allocation.diagnostic import (
    _fmt_compact,
    _style_dd,
    _style_pos_neg,
    _style_sharpe,
    book_performance,
)
from hailmary.allocation.portfolios import Portfolio
from hailmary.analytics.metrics import PerformanceMetrics

_BBG_SUFFIX_MAP: dict[str, str] = {
    "LN": ".L",   # London Stock Exchange (mostly UCITS)
    "US": "",     # NYSE / Nasdaq
    "SP": ".SI",  # Singapore Exchange
    "HK": ".HK",  # Hong Kong
    "GY": ".DE",  # Frankfurt
    "IM": ".MI",  # Milan
    "FP": ".PA",  # Paris
}

_WRAPPER_BY_SUFFIX: dict[str, str] = {
    "LN": "UCITS (LSE)",
    "US": "US",
    "SP": "SG",
    "HK": "HK",
    "GY": "EU",
    "IM": "EU",
    "FP": "EU",
}

_WINDOWS_DAYS: dict[str, int] = {"1Y": 252, "3Y": 756, "5Y": 1260}


def bloomberg_to_yahoo(bbg: str, name: str | None = None) -> str | None:
    """Convert Bloomberg ticker notation (e.g. 'ISAC LN', '9801:HK') to Yahoo
    (e.g. 'ISAC.L', '9801.HK'). Returns None for unparseable input.

    Bare tickers (no exchange suffix) are ambiguous in Stashaway's list:
    some are UCITS LSE-listed (XAID, ISDE, CSPX...), others are US-listed
    (FETH, INDY, EIDO, FLTW...). Heuristic: if the fund name contains
    "UCITS", default to ``.L``; otherwise leave bare (US listing).
    """
    if bbg is None or (isinstance(bbg, float) and pd.isna(bbg)):
        return None
    s = str(bbg).strip()
    if not s:
        return None
    if ":HK" in s:
        return s.replace(":HK", ".HK")
    parts = s.split()
    if len(parts) == 1:
        is_ucits = name is not None and "UCITS" in str(name).upper()
        return parts[0] + (".L" if is_ucits else "")
    sym, exch = parts[0], parts[1].upper()
    return sym + _BBG_SUFFIX_MAP.get(exch, "")


def _wrapper_label(bbg: str) -> str:
    if bbg is None or (isinstance(bbg, float) and pd.isna(bbg)):
        return ""
    s = str(bbg).strip()
    if ":HK" in s:
        return "HK"
    parts = s.split()
    if len(parts) < 2:
        return "?"
    return _WRAPPER_BY_SUFFIX.get(parts[1].upper(), "?")


def parse_etf_universe(xlsx_path: Path | str) -> pd.DataFrame:
    """Read Stashaway's ETF universe xlsx into a normalised DataFrame.

    Columns: ``asset_class``, ``name``, ``bbg_ticker``, ``yahoo_ticker``,
    ``wrapper``, ``fund_manager``.
    """
    df = pd.read_excel(xlsx_path).dropna(subset=["Ticker"])
    df = df.rename(
        columns={
            "Asset Class": "asset_class",
            "Underlying ETF": "name",
            "Ticker": "bbg_ticker",
            "Fund Manager": "fund_manager",
        }
    )
    df["yahoo_ticker"] = df.apply(
        lambda r: bloomberg_to_yahoo(r["bbg_ticker"], name=r.get("name")), axis=1
    )
    df["wrapper"] = df["bbg_ticker"].apply(_wrapper_label)
    # Stashaway's xlsx ticker column has known mistakes — override to the actual
    # Yahoo symbol where they differ from the underlying fund's real listing:
    _XLSX_OVERRIDES = {
        # Listed as XMOV US but the actual ticker is the LSE UCITS XMOV.L
        "XMOV US": "XMOV.L",
        # Row labelled "AHYG SP" (Asia HY USD) — real SGX symbol is QL3 (SGD
        # share class) per Yahoo longName lookup. AHYG.SI has no Yahoo data.
        "AHYG SP": "QL3.SI",
        # Row labelled "IBOXIG" (no exchange suffix) — Stashaway-internal code
        # for iShares iBoxx $ IG Corporate Bond ETF, real ticker LQD on NYSE.
        "IBOXIG": "LQD",
    }
    df["yahoo_ticker"] = df.apply(
        lambda r: _XLSX_OVERRIDES.get(str(r["bbg_ticker"]).strip(), r["yahoo_ticker"]), axis=1
    )
    return df[["asset_class", "name", "bbg_ticker", "yahoo_ticker", "wrapper", "fund_manager"]]


def _window_metrics(series: pd.Series, *, risk_free_rate: float = 0.0) -> dict[str, float]:
    """Sharpe / ann_return / vol / max_dd over `series`."""
    s = series.dropna()
    if len(s) < 5:
        return {
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
            "n_days": int(len(s)),
        }
    m = PerformanceMetrics(s, risk_free_rate=risk_free_rate)
    return {
        "ann_return": float(m.annualised_return),
        "ann_vol": float(m.annualised_vol),
        "sharpe": float(m.sharpe),
        "max_dd": float(m.max_drawdown),
        "n_days": int(len(s)),
    }


def _ytd_return(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    last_year = series.index.max().year
    sub = series[series.index.year == last_year]
    return float((1.0 + sub).prod() - 1.0) if not sub.empty else float("nan")


def build_etf_explorer(
    xlsx_path: Path | str,
    *,
    portfolios: Sequence[Portfolio],
    price_source: Any,
    fx_series_usd_sgd: pd.Series | None = None,
    start: date | datetime,
    end: date | datetime,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Build the ETF Explorer DataFrame.

    For each ETF in the Stashaway universe, fetches Yahoo total-return data
    over ``[start, end]``, computes multi-window metrics + correlation with
    the user's combined book.
    """
    universe = parse_etf_universe(xlsx_path)
    tickers = sorted({t for t in universe["yahoo_ticker"] if t})
    logger.info(f"ETF Explorer: fetching {len(tickers)} symbols from Yahoo…")
    try:
        all_returns = price_source.get_returns(tickers, start, end)
    except Exception as exc:
        logger.warning(f"Bulk fetch failed: {exc}; falling back to per-ticker fetches")
        all_returns = pd.DataFrame()
        for t in tickers:
            try:
                r = price_source.get_returns([t], start, end)
                if t in r.columns:
                    all_returns[t] = r[t]
            except Exception as e:
                logger.debug(f"  {t}: {e}")

    book_perf = book_performance(
        portfolios,
        start=start,
        end=end,
        price_source=price_source,
        fx_series_usd_sgd=fx_series_usd_sgd,
        align_window=True,
    )
    book_nav = book_perf.get("nav", pd.Series(dtype=float))
    book_returns = book_nav.pct_change().dropna() if not book_nav.empty else pd.Series(dtype=float)

    rows: list[dict[str, Any]] = []
    for _, row in universe.iterrows():
        t = row["yahoo_ticker"]
        series = all_returns[t].dropna() if (t and t in all_returns.columns) else pd.Series(dtype=float)
        rec: dict[str, Any] = {
            "asset_class": row["asset_class"],
            "name": row["name"],
            "ticker": t or "—",
            "wrapper": row["wrapper"],
            "fund_manager": row["fund_manager"],
            "has_data": not series.empty,
            "n_days": int(len(series)),
            # Short windows: cumulative (not annualised) — annualising a 1M
            # number extrapolates too aggressively
            "cum_return_1M": float((1.0 + series.tail(21)).prod() - 1.0) if len(series) >= 5 else float("nan"),
            "cum_return_3M": float((1.0 + series.tail(63)).prod() - 1.0) if len(series) >= 20 else float("nan"),
            "ytd_return": _ytd_return(series),
        }
        for label, n in _WINDOWS_DAYS.items():
            sub = series.tail(n) if not series.empty else pd.Series(dtype=float)
            m = _window_metrics(sub, risk_free_rate=risk_free_rate)
            rec[f"sharpe_{label}"] = m["sharpe"]
            rec[f"ann_return_{label}"] = m["ann_return"]
            rec[f"max_dd_{label}"] = m["max_dd"]
            rec[f"vol_{label}"] = m["ann_vol"]
        # Correlation with user's combined book — multi-window so we see
        # whether the relationship is stable or just a long-period average.
        if not book_returns.empty and not series.empty:
            aligned_all = pd.concat([book_returns, series], axis=1, join="inner").dropna()
            rec["corr_n"] = int(len(aligned_all))
            this_year = aligned_all.index.max().year if not aligned_all.empty else None
            for label, n in [
                ("1M", 21),
                ("3M", 63),
                ("YTD", None),
                ("1Y", 252),
                ("3Y", 756),
                ("5Y", 1260),
            ]:
                if label == "YTD":
                    sub = aligned_all[aligned_all.index.year == this_year] if this_year else aligned_all
                else:
                    sub = aligned_all.tail(n) if n else aligned_all
                # 1M has only ~21 trading days — lower threshold so it's defined
                min_obs = 10 if label == "1M" else 20
                if len(sub) >= min_obs:
                    rec[f"corr_book_{label}"] = float(sub.iloc[:, 0].corr(sub.iloc[:, 1]))
                else:
                    rec[f"corr_book_{label}"] = float("nan")
        else:
            rec["corr_n"] = 0
            for label in ("1M", "3M", "YTD", "1Y", "3Y", "5Y"):
                rec[f"corr_book_{label}"] = float("nan")
        rows.append(rec)

    out = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _style_corr(val: float) -> str:
    """Lower correlation = better diversification (greenish), higher = redder."""
    if pd.isna(val):
        return ""
    if val < 0.3:
        return "background-color: rgba(63, 185, 80, 0.22); color: #3fb950;"
    if val < 0.6:
        return "background-color: rgba(210, 153, 34, 0.15); color: #d29922;"
    return "background-color: rgba(248, 81, 73, 0.18); color: #f85149;"


def _style_target(target: float):
    def styler(val: float) -> str:
        if pd.isna(val):
            return ""
        if val >= target:
            return "background-color: rgba(63, 185, 80, 0.20); color: #3fb950;"
        if val >= 0:
            return "background-color: rgba(210, 153, 34, 0.15); color: #d29922;"
        return "background-color: rgba(248, 81, 73, 0.20); color: #f85149;"
    return styler


_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: Inter, system-ui, sans-serif; background: #0d1117; color: #e6edf3;
         margin: 0; padding: 24px; }}
  h1, h2 {{ color: #e6edf3; }}
  h2 {{ border-bottom: 1px solid #30363d; padding-bottom: 6px; margin-top: 32px; }}
  .meta {{ color: #8b949e; font-size: 13px; }}
  .ds-table {{ border-collapse: collapse; margin-top: 8px; font-size: 12px; }}
  .ds-table th, .ds-table td {{ border: 1px solid #30363d; padding: 3px 7px; text-align: right; }}
  .ds-table th {{ background: #161b22; text-align: left; cursor: pointer; user-select: none; position: relative; }}
  .ds-table th:hover {{ background: #1f2730; }}
  .ds-table th[data-sort-dir="asc"]::after {{ content: " ▲"; color: #58a6ff; }}
  .ds-table th[data-sort-dir="desc"]::after {{ content: " ▼"; color: #58a6ff; }}
  .ds-table td:first-child, .ds-table th:first-child, .ds-table td:nth-child(2), .ds-table th:nth-child(2) {{ text-align: left; }}
  .ds-table td {{ font-variant-numeric: tabular-nums; }}
  .footnote {{ color: #8b949e; font-size: 12px; line-height: 1.45; max-width: 1100px; }}
  .footnote code {{ background: #161b22; padding: 1px 4px; border-radius: 3px; font-size: 11.5px; }}
  ul.footnote {{ padding-left: 20px; }}
</style>
</head><body>
<h1>{title}</h1>
<p class="meta">Generated {generated_at} · {n_etfs} ETFs · {n_with_data} with Yahoo data
   · target ann return = {target:.1%}</p>
<p class="footnote">
  Discovery view over Stashaway's ETF Explorer universe. <strong>Click any column header to sort.</strong>
</p>
<ul class="footnote">
  <li><strong>1M / 3M / YTD</strong> are <em>cumulative</em> returns over the window
      (not annualised — annualising a 1M return extrapolates too aggressively).
      Colour: green = positive, red = negative.</li>
  <li><strong>1Y / 3Y / 5Y ret</strong> are <em>annualised</em> returns. Traffic-light:
      green ≥ target ({target:.1%}), yellow positive-but-under, red negative.</li>
  <li><strong>Sharpe (5Y)</strong> gradient: deeper green = higher risk-adjusted return.</li>
  <li><strong>Max DD (5Y)</strong> red intensity = magnitude of worst drawdown.</li>
  <li><strong>ρ 1M / 3M / YTD / 1Y / 3Y / 5Y</strong> traffic-light:
      <span style="color:#3fb950">green &lt; 0.3</span> (good diversifier in that window),
      <span style="color:#d29922">yellow 0.3–0.6</span>, <span style="color:#f85149">red ≥ 0.6</span>.
      Multi-window so you can see if the relationship is <em>stable</em> (consistent across
      columns) or just a long-period average — an ETF that's green 5Y but red 1M
      has become correlated recently.</li>
</ul>
<p class="footnote">
  <em>Caveat</em>: still flat-period metrics within each window. A low-ρ ETF can become high-ρ
  specifically in equity sell-offs. v2 of this report will add drawdown-bucket Sharpe and
  rolling-correlation timelines.
</p>
{table}
<script>
(function () {{
  function parseCell(text) {{
    var t = (text || "").trim();
    if (t === "" || t === "—" || t === "no data" || t.toLowerCase() === "nan") return null;
    var compact = t.match(/^([-+]?[0-9]+(?:\\.[0-9]+)?)\\s*([KMB])$/i);
    if (compact) {{
      var mult = {{ K: 1e3, M: 1e6, B: 1e9 }}[compact[2].toUpperCase()];
      return parseFloat(compact[1]) * mult;
    }}
    if (/%$/.test(t)) {{
      var pct = parseFloat(t.replace(/[^0-9.+\\-]/g, ""));
      return isNaN(pct) ? t : pct / 100;
    }}
    var stripped = t.replace(/[,\\s]/g, "");
    if (/^[-+]?[0-9]+(?:\\.[0-9]+)?$/.test(stripped)) return parseFloat(stripped);
    return t.toLowerCase();
  }}
  function compare(a, b) {{
    if (a === null && b === null) return 0;
    if (a === null) return 1;
    if (b === null) return -1;
    if (typeof a === "number" && typeof b === "number") return a - b;
    return String(a).localeCompare(String(b));
  }}
  function makeSortable(table) {{
    var headerRows = table.querySelectorAll("thead tr");
    if (!headerRows.length) return;
    var headers = headerRows[headerRows.length - 1].querySelectorAll("th");
    var tbody = table.querySelector("tbody");
    if (!tbody) return;
    headers.forEach(function (th, idx) {{
      th.addEventListener("click", function () {{
        var current = th.getAttribute("data-sort-dir");
        var nextDir = current === "asc" ? "desc" : "asc";
        headers.forEach(function (h) {{ h.removeAttribute("data-sort-dir"); }});
        th.setAttribute("data-sort-dir", nextDir);
        var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));
        rows.sort(function (rA, rB) {{
          var vA = parseCell(rA.cells[idx] ? rA.cells[idx].textContent : "");
          var vB = parseCell(rB.cells[idx] ? rB.cells[idx].textContent : "");
          var c = compare(vA, vB);
          return nextDir === "asc" ? c : -c;
        }});
        rows.forEach(function (r) {{ tbody.appendChild(r); }});
      }});
    }});
  }}
  document.addEventListener("DOMContentLoaded", function () {{
    document.querySelectorAll("table.ds-table").forEach(makeSortable);
  }});
}})();
</script>
</body></html>
"""


def render_etf_explorer_report(
    df: pd.DataFrame,
    output_path: Path | str,
    *,
    target_ann_return: float = 0.05,
    title: str = "Stashaway ETF Explorer",
) -> Path:
    """Render the ETF Explorer DataFrame to a self-contained HTML report."""
    display = df.rename(
        columns={
            "asset_class": "Asset Class",
            "name": "Name",
            "ticker": "Ticker",
            "wrapper": "Wrap",
            "fund_manager": "Manager",
            "has_data": "Data?",
            "n_days": "Days",
            "cum_return_1M": "1M",
            "cum_return_3M": "3M",
            "ytd_return": "YTD",
            "ann_return_1Y": "1Y ret",
            "ann_return_3Y": "3Y ret",
            "ann_return_5Y": "5Y ret",
            "sharpe_1Y": "1Y Sharpe",
            "sharpe_3Y": "3Y Sharpe",
            "sharpe_5Y": "5Y Sharpe",
            "max_dd_5Y": "5Y MaxDD",
            "vol_5Y": "5Y Vol",
            "corr_book_1M": "ρ 1M",
            "corr_book_3M": "ρ 3M",
            "corr_book_YTD": "ρ YTD",
            "corr_book_1Y": "ρ 1Y",
            "corr_book_3Y": "ρ 3Y",
            "corr_book_5Y": "ρ 5Y",
            "corr_n": "ρ days",
        }
    )
    cols = [
        "Asset Class", "Name", "Ticker", "Wrap", "Manager", "Data?", "Days",
        "1M", "3M", "YTD", "1Y ret", "3Y ret", "5Y ret",
        "1Y Sharpe", "3Y Sharpe", "5Y Sharpe",
        "5Y MaxDD", "5Y Vol",
        "ρ 1M", "ρ 3M", "ρ YTD", "ρ 1Y", "ρ 3Y", "ρ 5Y", "ρ days",
    ]
    display = display[[c for c in cols if c in display.columns]]
    display = display.copy()
    display["Data?"] = display["Data?"].map({True: "✓", False: "—"})

    target_styler = _style_target(target_ann_return)
    return_pct_cols = ["1M", "3M", "YTD", "1Y ret", "3Y ret", "5Y ret"]
    formatters: dict[str, Any] = {c: "{:+.2%}".format for c in return_pct_cols}
    formatters["5Y Vol"] = "{:.2%}".format
    formatters["5Y MaxDD"] = "{:.2%}".format
    formatters["1Y Sharpe"] = "{:.2f}".format
    formatters["3Y Sharpe"] = "{:.2f}".format
    formatters["5Y Sharpe"] = "{:.2f}".format
    for c in ("ρ 1M", "ρ 3M", "ρ YTD", "ρ 1Y", "ρ 3Y", "ρ 5Y"):
        formatters[c] = "{:+.2f}".format
    formatters["Days"] = "{:,}".format
    formatters["ρ days"] = "{:,}".format

    styler = display.style.format(formatters, na_rep="—")
    # Short-window returns (1M/3M/YTD) are cumulative, not annualised — use
    # sign-only colouring (positive=green, negative=red) rather than the
    # annual-target traffic-light which would mislead at these scales.
    for c in ["1M", "3M", "YTD"]:
        if c in display.columns:
            styler = styler.map(_style_pos_neg, subset=[c])
    for c in ["1Y ret", "3Y ret", "5Y ret"]:
        if c in display.columns:
            styler = styler.map(target_styler, subset=[c])
    if "5Y Sharpe" in display.columns:
        styler = styler.map(_style_sharpe, subset=["5Y Sharpe"])
    if "5Y MaxDD" in display.columns:
        styler = styler.map(_style_dd, subset=["5Y MaxDD"])
    corr_cols = [c for c in ("ρ 1M", "ρ 3M", "ρ YTD", "ρ 1Y", "ρ 3Y", "ρ 5Y") if c in display.columns]
    if corr_cols:
        styler = styler.map(_style_corr, subset=corr_cols)
    styler = styler.hide(axis="index").set_table_attributes('class="ds-table"')

    table_html = styler.to_html()
    n_with_data = int(df["has_data"].sum()) if "has_data" in df.columns else 0
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _TEMPLATE.format(
            title=title,
            generated_at=datetime.now().isoformat(timespec="seconds"),
            n_etfs=len(df),
            n_with_data=n_with_data,
            target=target_ann_return,
            table=table_html,
        ),
        encoding="utf-8",
    )
    return out
