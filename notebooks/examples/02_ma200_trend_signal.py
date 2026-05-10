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
# # 02 — Trend Signal Generation
#
# Generates the MA-200 crossover signal that drives every other notebook in this
# series. The output is a bar-level frame with signal columns appended — *no* fills
# or returns yet (those come from `BarBacktest` in notebook 03).
#
# **Pipeline**
# ```
# YahooFinanceProvider  →  TrendSignal  →  BarBacktest  →  SignalTradePerformance       (iid bets)
#       (bars)            (signal cols)   (bt_result)    └─  SignalAllocationPerformance (capital deployed)
#                               ▲
#                          this notebook
# ```
#
# **What this notebook covers**
#
# 1. **Warmup-aware fetch** — pulls `signal.warmup` extra business days before
#    `start` so the rolling MA-200 is valid from the first in-window bar
#    (`trim_start=start` then drops the warmup region).
# 2. **`TrendSignal.run()` columns** — documents every appended column: `ma`,
#    `signal_close` (observable, post-close), `signal_open` (tradeable, shifted
#    +1 bar), `trade_direction`, `turnover`, `enter` / `exit`, `cycle`,
#    `signal_age`.
# 3. **Signal visualisation** — for one symbol, plots close + MA-200 with shaded
#    regions where `signal_open = 1` and triangle markers at every entry/exit.
# 4. **Clipboard export** — single-symbol signal frame copied for inspection in
#    Excel.

# %%
import pandas as pd
import plotly.graph_objects as go
from hailmary.data.providers import YahooFinanceProvider
from hailmary.models import TrendSignal
from hailmary.viz.theme import PALETTE, apply_theme


# %% [markdown]
# ## 1. Data

# %%
signal = TrendSignal(ma_window=200)
yahoo = YahooFinanceProvider()
symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
start, end = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")

# Fetch warmup bars before start so MA-200 is valid from day one
fetch_start = start - pd.offsets.BDay(signal.warmup)
bars = yahoo.get_bars(symbols, start=fetch_start, end=end, adjust=False)

signal_df = signal.run(bars, trim_start=start)
print(f"{signal_df.shape[0]:,} bars  |  {signal_df.index.get_level_values('symbol').nunique()} symbols")
print(f"MA NaN count: {signal_df['ma'].isna().sum()}  (should be 0)")
signal_df.head(6)

# %% [markdown]
# ## 2. Signal Columns
#
# `TrendSignal.run()` returns the input bars with these columns appended:
#
# | Column | Description |
# |---|---|
# | `ma` | Rolling MA of close |
# | `signal_close` | 1 when `close > MA` — observable after the close, not yet tradeable |
# | `signal_open` | `signal_close` shifted +1 bar — this drives actual trades |
# | `trade_direction` | +1 on entry, −1 on exit, 0 otherwise |
# | `turnover` | Absolute change in `signal_open` |
# | `enter` / `exit` | 1 on the first / last bar of each position |
# | `cycle` | `init` / `held` / `exit` / `None` |
# | `signal_age` | Bars since the position opened (including exit bar) |
#
# Execution columns (`return_*`, `position_*`, `equity_*`) are added by `BarBacktest` in notebook 03.

# %%
signal_df = TrendSignal(ma_window=200).run(bars, trim_start=start)

cols = ["close", "ma", "signal_close", "signal_open", "cycle", "enter", "exit", "signal_age"]
signal_df[cols].head(12)

# %% [markdown]
# ## 3. Signal Visualisation
#
# Price and MA with shaded regions where `signal_open = 1` and entry/exit markers.

# %%
sym =  "SOL-USD"# , "SOL-USD""BTC-USD"
d = signal_df.xs(sym, level="symbol")

entries = d[d["enter"] == 1].index
exits   = d[d["exit"]  == 1].index

fig = go.Figure()

# Shaded regions where position is on (signal_open = 1)
in_pos = False
for ts, row in d.iterrows():
    if row["signal_open"] == 1 and not in_pos:
        region_start, in_pos = ts, True
    elif row["signal_open"] == 0 and in_pos:
        fig.add_vrect(x0=region_start, x1=ts,
                      fillcolor=PALETTE["accent_blue"], opacity=0.08, line_width=0)
        in_pos = False
if in_pos:
    fig.add_vrect(x0=region_start, x1=d.index[-1],
                  fillcolor=PALETTE["accent_blue"], opacity=0.08, line_width=0)

# Close and MA
fig.add_trace(go.Scatter(x=d.index, y=d["close"],
    name="Close", line=dict(color=PALETTE["text_primary"], width=1)))
fig.add_trace(go.Scatter(x=d.index, y=d["ma"],
    name="MA-200", line=dict(color=PALETTE["accent_blue"], width=1.5, dash="dot")))

# Entry / exit markers
fig.add_trace(go.Scatter(x=entries, y=d.loc[entries, "close"],
    name="Entry", mode="markers",
    marker=dict(color=PALETTE["accent_green"], size=8, symbol="triangle-up")))
fig.add_trace(go.Scatter(x=exits, y=d.loc[exits, "close"],
    name="Exit", mode="markers",
    marker=dict(color=PALETTE["accent_red"], size=8, symbol="triangle-down")))

apply_theme(fig, title=f"{sym} — Price, MA-200, and Signal", height=480)
fig.update_layout(yaxis_title="Price (USD)")
fig.show()

# %%
sym = "SOL-USD"
signal_df.query("symbol == @sym").unstack('symbol').to_clipboard()
print(f"Copied {sym} to clipboard")

# TODO — think about how to handle the case for enter and exit on the same bar (e.g. if signal_open flips from 0 to 1 to 0 again in 3 bars or less, which can happen with short MAs). Currently the code treats this as a full trade cycle, but it might make more sense to ignore the exit and just keep the position open until the signal stabilises. This would be a bit more complex to implement but would avoid over-trading in choppy conditions.
