# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04 — Signal Quality
#
# Walks `SignalTradePerformance` method-by-method, then composes them all into
# the standalone HTML tearsheet at the end.
#
# **Pipeline**
# ```
# YahooFinanceProvider  →  TrendSignal  →  BarBacktest  →  SignalTradePerformance       (iid bets)
#       (bars)            (signal cols)   (bt_result)    │                              ▲
#                                                        │                         this notebook
#                                                        └─ SignalAllocationPerformance (capital deployed)
# ```
#
# **Sections**
#
# 1. **Pipeline & setup** — `bars → signal_df → bt_result → trade_perf`. Each
#    section below defines its own intermediate (`trades_df`, `ts_net`/`ts_con`,
#    `paths`) and threads it forward via the `trade_stats=` kwarg, so nothing
#    recomputes the per-trade groupby.
# 2. **`trade_stats()`** — defines `trades_df`. Canonical per-trade scalar table
#    (one row per entry-to-exit cycle). Every other method derives from this.
# 3. **`trade_summary(method)`** — defines `ts_net` / `ts_con`. Per-symbol
#    aggregation: win rate, expectancy, profit factor, skew, intra-trade DD.
# 4. **`quality_flags(row)`** — uses `ts_net` / `ts_con`. Warning flags from one
#    row of a `trade_summary`. Cross-fill flags fire when non-net is compared to net.
# 5. **`d5_stats(symbol, method)`** — uses `trades_df`. Entry-timing breakdown:
#    how often does a trade work by bar 5, and how often does the tail add return?
# 6. **`quality_table(methods)`** — uses `trades_df`. The joined view:
#    trade_summary + d5_stats + quality_flags, indexed by `(symbol, method)`.
# 7. **`trade_paths()`** — defines `paths`. Per-bar cumulative-return Series.
#    Feeds the aligned trade-paths chart and the entry-timing horizontal-bar chart.
# 8. **Distribution** — per-symbol return histograms with expectancy / median vlines.
# 9. **`SignalComparison`** — multi-variant harness. Includes the
#    `BarBacktest(..., entry_offset=5)` *delayed-entry* hypothetical alongside
#    the native run, plus any other MA-window variants you want to drop in.
# 10. **HTML Tearsheet** — `SignalTearsheet(...).save()` bundles every section above
#     into one self-contained `.html`.
#
# Column / flag definitions live in `hailmary.analytics` as `TRADE_STATS_DOCS`,
# `TRADE_SUMMARY_DOCS`, `QUALITY_FLAG_DOCS`, `D5_STATS_DOCS`, `FILL_METHOD_DOCS`.
# The HTML tearsheet's footnotes and the docs rendered below in each section are
# both built from those same dicts — change a description in one place and both
# surfaces update.

# %%
import pandas as pd
from pathlib import Path
from IPython.display import Markdown, display

from hailmary.data.providers import YahooFinanceProvider
from hailmary.models import TrendSignal
from hailmary.backtest.signal_backtest import BarBacktest
from hailmary.analytics import (
    SignalTradePerformance,
    TRADE_STATS_DOCS,
    TRADE_SUMMARY_DOCS,
    QUALITY_FLAG_DOCS,
    D5_STATS_DOCS,
    FILL_METHOD_DOCS,
    docs_markdown,
)
from hailmary.analytics.signal_comparison import SignalComparison
from hailmary.viz.signal_tearsheet import (
    SignalTearsheet,
    paths_fig,
    timing_fig,
    distribution_fig,
    quality_table_styler,
    trade_log_styler,
)

# %% [markdown]
# ## 1. Pipeline & Setup
#
# Fetch bars → run signal → run backtest → wrap in `SignalTradePerformance`.
# This cell only builds `trade_perf`. Each section below defines its own
# intermediate (`trades_df`, `ts_net`/`ts_con`, `paths`) at the point of first
# use and threads it forward via the `trade_stats=` keyword.

# %%
signal = TrendSignal(ma_window=200)
yahoo  = YahooFinanceProvider()
symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
start, end = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")

fetch_start = start - pd.offsets.BDay(signal.warmup)
bars = yahoo.get_bars(symbols, start=fetch_start, end=end, adjust=False)

signal_df  = signal.run(bars, trim_start=start)
bt_result  = BarBacktest().run(signal_df)
trade_perf = SignalTradePerformance(bt_result)

# Each section below defines its own intermediate at point of first use:
#   §2  trades_df = trade_perf.trade_stats()
#   §3  ts_net    = trade_perf.trade_summary("net",          trade_stats=trades_df)
#       ts_con    = trade_perf.trade_summary("conservative", trade_stats=trades_df)
#   §7  paths     = trade_perf.trade_paths(trade_stats=trades_df)

print(
    f"{bt_result.data.shape[0]:,} bars  "
    f"|  {bt_result.data.index.get_level_values('symbol').nunique()} symbols"
)

# %% [markdown]
# ## 2. `trade_stats()`
#
# The canonical per-trade scalar DataFrame: one row per entry-to-exit cycle.
# Every other method on `SignalTradePerformance` derives from this — passing
# the result back via `trade_stats=trades_df` avoids redoing the per-trade
# groupby.
#
# `trade_log_styler` shows the same data with green/red colouring on the
# return columns and a red gradient on intra-trade drawdowns; trades shorter
# than 5 bars carry a `(Nd)` muted suffix on the 5-day columns.

# %%
trades_df = trade_perf.trade_stats()

display(trade_log_styler(trades_df))
display(Markdown(docs_markdown(TRADE_STATS_DOCS, title="Columns")))

# %% [markdown]
# ## 3. `trade_summary(method)`
#
# Per-symbol aggregation of `trades_df`: collapses every trade for one
# symbol into edge metrics — win rate, expectancy, profit factor, skew,
# intra-trade DD. Returns one DataFrame per fill method.

# %%
ts_net = trade_perf.trade_summary("net",          trade_stats=trades_df)
ts_con = trade_perf.trade_summary("conservative", trade_stats=trades_df)

display(Markdown("**`trade_summary('net')`**"))
display(ts_net)
display(Markdown("**`trade_summary('conservative')`**"))
display(ts_con)
display(Markdown(docs_markdown(TRADE_SUMMARY_DOCS, title="Columns")))

# %% [markdown]
# ## 4. `quality_flags(row)`
#
# Boolean warning flags computed from one row of a `trade_summary`. The
# cross-fill flags (`edge_reversed`, `fill_halves_edge`, `median_flips`)
# only fire when a non-net method is compared to net via the optional
# `net_row` keyword.

# %%
sym = "BTC-USD"

flags_net = SignalTradePerformance.quality_flags(ts_net.loc[sym])
flags_con = SignalTradePerformance.quality_flags(
    ts_con.loc[sym],
    net_row=ts_net.loc[sym],
)

display(Markdown(f"**Net flags ({sym})**"))
display(pd.Series(flags_net))
display(Markdown(f"**Conservative flags ({sym}) — vs net**"))
display(pd.Series(flags_con))
display(Markdown(docs_markdown(QUALITY_FLAG_DOCS, title="Flags")))

# %% [markdown]
# ## 5. `d5_stats(symbol, method)`
#
# Entry-timing breakdown for one symbol: how often does a trade work by
# bar 5 (`wr_d5`), and how often does the post-bar-5 tail add return
# (`wr_tail`)?  `wr_d5` filters out trades that didn't reach bar 5;
# `wr_tail` filters out trades ≤ 5 bars.

# %%
d5_rows = [
    {"symbol": s, **SignalTradePerformance.d5_stats(s, "net", trade_stats=trades_df)}
    for s in trades_df["symbol"].unique()
]
display(Markdown("**`d5_stats(symbol, 'net')`** — across all symbols"))
display(pd.DataFrame(d5_rows).set_index("symbol"))
display(Markdown(docs_markdown(D5_STATS_DOCS, title="Fields")))

# %% [markdown]
# ## 6. `quality_table(methods)`
#
# The joined view: `trade_summary` + `d5_stats` + `quality_flags` per
# `(symbol, method)` pair. `quality_table_styler` formats it with
# diverging green/red on the edge metrics, descending-red on drawdowns,
# and the raw `flags` dict rendered to compact text.

# %%
quality_table_styler(trade_perf.quality_table(trade_stats=trades_df))

# %% [markdown]
# ## 7. `trade_paths()`
#
# Per-bar cumulative-return Series for every trade (one Series per fill
# method, starting at 0.0 on the entry bar). Carries the scalar fields
# (`final_*`, `d5_*`, `max_dd_*`) sourced from `trades_df` so chart hovers
# don't have to look them up separately.
#
# - `paths_fig(paths, method)` — every trade pivoted to *day 0 = entry*,
#   win/loss-coloured, with mean / median / ±1σ overlays.
# - `timing_fig(paths, method)` — horizontal stacked bar: first-5-bars
#   return vs 5d→exit return per individual trade.

# %%
paths = trade_perf.trade_paths(trade_stats=trades_df)

paths_fig(paths,  method="net").show()
paths_fig(paths,  method="conservative").show()
timing_fig(paths, method="net").show()
timing_fig(paths, method="conservative").show()

# %% [markdown]
# ## 8. Return Distribution
#
# Per-symbol histogram of trade returns — green bars are winners, red are
# losers. Vertical lines mark the **expectancy** (solid) and the **median**
# (purple dashed) for the chosen fill method. Expectancy ≫ median means a
# few large winners are inflating the mean — the same outlier-driven edge
# that the *Exp ex-Top* and *top trade outlier* flags surface in §6.

# %%
distribution_fig(trades_df, ts_net, ts_con, method="net").show()
distribution_fig(trades_df, ts_net, ts_con, method="conservative").show()

# %% [markdown]
# ## 9. `SignalComparison`
#
# Drop multiple `BarBacktestResult` variants into the `variants` dict and
# re-run — `SignalComparison(variants).tearsheet()` produces a three-panel
# side-by-side view: pooled-quality table per variant, per-symbol expectancy
# heatmap, and per-symbol return-distribution boxplots. Use this to rank
# signal variants without bouncing between standalone tearsheets.
#
# The cell below shows the **delayed-entry hypothetical** as a variant:
# `BarBacktest().run(signal_df, entry_offset=5)` only takes a trade if the
# signal still holds 5 bars after the original flip — the delay is an
# *execution* parameter, not a separate signal class. Compare its expectancy
# and trade count against `MA-200 Base` to see whether waiting for
# confirmation improves the edge or just costs you trades.

# %%
bt_delayed_5 = BarBacktest().run(signal_df, entry_offset=5)

variants = {
    "MA-200 Base":       bt_result,
    "MA-200 Delayed-5":  bt_delayed_5,
    # Add more here as you iterate. Each value is a BarBacktestResult.
    # e.g. BarBacktest().run(TrendSignal(ma_window=100).run(bars))
}

SignalComparison(variants).tearsheet().show()

# %% [markdown]
# ## 10. Standalone HTML Tearsheets
#
# `SignalTearsheet` bundles every section above (Per-Trade Quality → Entry
# Timing → Trade Log → Aligned Trade Paths → Return Distribution) into one
# self-contained `.html` file with sticky navigation and symbol/fill
# filters — viewable in any browser without Jupyter.
#
# Footnotes in the HTML's quality table are built from the same `*_DOCS`
# dicts shown above — single source of truth across both surfaces.
#
# The cell below saves **two** tearsheets, one per `BarBacktestResult` variant:
#
# - `ma200_native.html` — the native run (`bt_result`).
# - `ma200_delayed_5.html` — the `entry_offset=5` run (`bt_delayed_5` from §9).
#
# Each tearsheet's `d5` / `5d→Exit` / timing surfaces describe the realized
# trades of *its own* variant: native → first-5 bars from the original signal
# flip; delayed → first-5 bars after the 5-bar confirmation hold. Open both
# side-by-side in the browser to compare. Only the native one auto-opens to
# keep the tab count reasonable.

# %%
reports = Path("../../reports")

native_path = SignalTearsheet(
    trade_perf,
    title="MA-200 Trend Signal — BTC/ETH/SOL 2022–2024 (Native)",
).save(reports / "ma200_native.html", open=True)

delayed_path = SignalTearsheet(
    SignalTradePerformance(bt_delayed_5),
    title="MA-200 Trend Signal — BTC/ETH/SOL 2022–2024 (Delayed-5)",
).save(reports / "ma200_delayed_5.html", open=True)

print(f"Native  → {native_path}")
print(f"Delayed → {delayed_path}")
