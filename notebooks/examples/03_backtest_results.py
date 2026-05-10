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
# # 03 — Backtest Results
#
# Reference card for the `BarBacktest` output. Documents the execution and equity
# columns it appends on top of the signal frame from notebook 02 — the per-trade
# log is in notebook 04, the equity curves in notebook 05.
#
# **Pipeline**
# ```
# YahooFinanceProvider  →  TrendSignal  →  BarBacktest  →  SignalTradePerformance       (iid bets)
#       (bars)            (signal cols)   (bt_result)    └─  SignalAllocationPerformance (capital deployed)
#                                             ▲
#                                        this notebook
# ```
#
# **What this notebook covers**
#
# 1. **Data → Signal → Backtest** — minimal end-to-end wiring to produce a
#    `BarBacktestResult`.
# 2. **`bt_result.data` schema** — every execution column appended by the
#    backtest: `cycle`, three return series (`return_mark_to_close`,
#    `return_conservative`, `return_net`), `position_start` / `position_end`,
#    three matching equity curves, and `trade_cycle_id`. Single-symbol slice
#    copied to clipboard for Excel inspection.
# 3. **Delayed entry — `entry_offset`** — `BarBacktest.run(signal_df,
#    entry_offset=k)` only takes a trade if the signal still holds `k` bars
#    after the original flip. Pure execution-layer change: the analytics layer
#    treats the resulting `BarBacktestResult` identically.

# %%
import pandas as pd
import plotly.graph_objects as go

from hailmary.data.providers import YahooFinanceProvider
from hailmary.models import TrendSignal
from hailmary.backtest.signal_backtest import BarBacktest
from hailmary.viz.theme import PALETTE, apply_theme

# %% [markdown]
# ## 1. Data → Signal → Backtest

# %%
signal = TrendSignal(ma_window=200)
yahoo  = YahooFinanceProvider()
symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
start, end = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")

fetch_start = start - pd.offsets.BDay(signal.warmup)
bars = yahoo.get_bars(symbols, start=fetch_start, end=end, adjust=False)

signal_df = signal.run(bars, trim_start=start)
bt_result = BarBacktest().run(signal_df)

df = bt_result.data
print(f"{df.shape[0]:,} bars  |  {df.index.get_level_values('symbol').nunique()} symbols")

# %% [markdown]
# ## 2. bt_result Structure
#
# `BarBacktest.run()` returns a `BarBacktestResult` whose `.data` is a `(symbol, timestamp)`
# MultiIndex DataFrame. The original signal columns are preserved and the backtest engine
# appends execution and equity columns.
#
# | Column | Description |
# |---|---|
# | `cycle` | `None` (flat), `init` (entry bar), `held` (mid-trade), `exit` (exit bar) |
# | `return_mark_to_close` | Fill-aware daily return — entry on open after `init`, exit on open after `exit` |
# | `return_conservative` | Worst-case fill — entry on `init` high, exit on `exit` low |
# | `return_net` | MTC return minus round-trip cost (`cost_bps / 10 000` on entry and exit bars) |
# | `position_start` | 1 when capital is deployed at bar open |
# | `position_end` | 1 when capital is deployed at bar close |
# | `strategy_equity_mark_to_close` | Cumulative product of MTC returns (starts at 1.0) |
# | `strategy_equity_conservative` | Cumulative product of conservative returns |
# | `strategy_equity_net` | Cumulative product of net returns |
# | `trade_cycle_id` | Integer ID incrementing on each new trade or flat period |

# %%
bt_result.data.columns

# %%
exec_cols = [
    "cycle", "return_mark_to_close", "return_conservative", "return_net",
    "position_start", "position_end",
    "strategy_equity_mark_to_close", "strategy_equity_conservative", "strategy_equity_net",
    "trade_cycle_id",
]
df[exec_cols].head(12)

# %%
sym = "BTC-USD"
bt_result.data.query("symbol == @sym").unstack('symbol').to_clipboard()

# %% [markdown]
# ## 3. Delayed Entry — `entry_offset`
#
# `BarBacktest.run(signal_df, entry_offset=k)` delays each trade's entry by `k`
# bars and **only takes the trade if the signal still holds at bar k**. Trades
# that don't survive (run length ≤ `k + 1` bars) are dropped entirely; surviving
# trades' entry timestamp shifts forward by `k` bars while the original exit bar
# stays put.
#
# This is a pure execution-layer parameter — same `TrendSignal` output, same
# `signal_df`, same `BarBacktestResult` shape. Every downstream call
# (`SignalTradePerformance`, `SignalAllocationPerformance`, the HTML tearsheet,
# `SignalComparison`) consumes the delayed result with **zero special-casing**.
#
# The cell below compares native vs `entry_offset=5`:

# %%
bt_delayed_5 = BarBacktest().run(signal_df, entry_offset=5)

eq_col = "strategy_equity_mark_to_close"
base_trades    = bt_result.data.groupby(level="symbol")["enter"].sum().astype(int)
delayed_trades = bt_delayed_5.data.groupby(level="symbol")["enter"].sum().astype(int)
final_base     = bt_result.data.groupby(level="symbol")[eq_col].last() - 1
final_delayed  = bt_delayed_5.data.groupby(level="symbol")[eq_col].last() - 1

pd.DataFrame({
    "trades_native":     base_trades,
    "trades_delayed_5":  delayed_trades,
    "trades_dropped":    base_trades - delayed_trades,
    "return_native":     final_base.map(lambda x: f"{x:+.1%}"),
    "return_delayed_5":  final_delayed.map(lambda x: f"{x:+.1%}"),
})
