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
# # 05 — Portfolio Analytics
#
# Aggregates the per-symbol backtest from notebook 03 into portfolio-level NAV and
# risk-adjusted metrics. Strictly the *capital-deployed* lens via
# `SignalAllocationPerformance` — per-trade analysis lives in notebook 04.
#
# **Pipeline**
# ```
# YahooFinanceProvider  →  TrendSignal  →  BarBacktest  →  SignalTradePerformance       (iid bets)
#       (bars)            (signal cols)   (bt_result)    └─  SignalAllocationPerformance (capital deployed)
#                                                                      ▲
#                                                                 this notebook
# ```
#
# **What this notebook covers**
#
# 1. **Per-symbol equity curves** — each symbol's NAV with MTC (solid) vs
#    conservative (dotted) overlaid; the gap = fill-risk range, flat stretches =
#    signal off.
# 2. **Per-symbol drawdown** — filled drawdown plot from the conservative
#    equity curve; flat regions confirm no capital was deployed.
# 3. **Equal-weight portfolio NAV** — `portfolio_equity()` weighting each
#    in-signal symbol at `1 / N_universe` (idle capital sits as cash),
#    overlaid on the individual symbol curves.
# 4. **Three capital models compared** — `rebalanced` (1/N daily reset),
#    `buy_and_hold` (let winners compound), and `fixed_stake` (fixed dollar
#    amount per entry, P&L → cash). All normalised to start at 1.0.
# 5. **Risk-adjusted metrics table** — total return, CAGR, vol, Sharpe,
#    Sortino, max drawdown, Calmar across the three capital models.
# 6. **Full `PerformanceMetrics` summary** — complete stat block (incl. VaR,
#    CVaR, skewness, kurtosis, Omega) for the rebalanced portfolio.

# %%
import pandas as pd
import plotly.graph_objects as go

from hailmary.data.providers import YahooFinanceProvider
from hailmary.models import TrendSignal
from hailmary.backtest.signal_backtest import BarBacktest
from hailmary.analytics.signal_analytics import SignalAllocationPerformance
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
alloc = SignalAllocationPerformance(bt_result)

colours      = [PALETTE["accent_blue"], PALETTE["accent_green"], PALETTE["accent_purple"]]
fill_colours = ["rgba(88,166,255,0.12)", "rgba(63,185,80,0.12)", "rgba(163,113,247,0.12)"]

eq_con = alloc.equity(method="conservative")

print(f"{bt_result.data.shape[0]:,} bars  |  {bt_result.data.index.get_level_values('symbol').nunique()} symbols")

# %% [markdown]
# ## 2. Per-Symbol Equity Curves
#
# Each symbol's NAV plotted with both fill methods side by side: **solid** = MTC
# (open after signal bar, best realistic fill); **dotted** = conservative (worst
# realistic fill — high entry / low exit). The gap between the two lines is the
# fill-risk range for that symbol; flat stretches confirm the signal was off.

# %%
eq_mtc = alloc.equity(method="mtc")

fig = go.Figure()
for i, sym in enumerate(eq_mtc.columns):
    c = colours[i % len(colours)]
    fig.add_trace(go.Scatter(x=eq_mtc.index, y=eq_mtc[sym], name=f"{sym} MTC",
                             line=dict(color=c, width=1.8)))
    fig.add_trace(go.Scatter(x=eq_con.index, y=eq_con[sym], name=f"{sym} Conservative",
                             line=dict(color=c, width=1, dash="dot")))

apply_theme(fig, title="Per-Symbol Equity Curves — MA-200 Trend Signal (MTC vs Conservative)", height=460)
fig.update_layout(yaxis_title="Equity (normalised to 1.0)")
fig.show()

# %% [markdown]
# ## 3. Drawdown
#
# Per-symbol drawdown from the conservative equity curve.  Flat periods reflect the signal
# being off — no capital is deployed so the NAV holds still and drawdown does not accumulate.

# %%
dd = eq_con / eq_con.cummax() - 1

fig = go.Figure()
for i, sym in enumerate(dd.columns):
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd[sym],
        name=sym,
        fill="tozeroy",
        line=dict(color=colours[i % len(colours)], width=1),
        fillcolor=fill_colours[i % len(fill_colours)],
    ))

apply_theme(fig, title="Drawdown — MA-200 Trend Signal (Conservative)", height=380)
fig.update_layout(yaxis_title="Drawdown", yaxis_tickformat=".0%")
fig.show()

# %% [markdown]
# ## 4. Combined Portfolio — Equal-Weight
#
# `portfolio_equity()` weights each in-signal symbol at `1 / N_universe` at each bar —
# uninvested capital sits as cash when fewer symbols are in signal.  The bold yellow line
# is the combined portfolio; muted dotted lines are individual symbols.

# %%
port_eq = alloc.portfolio_equity(method="conservative")

fig = go.Figure()

for i, sym in enumerate(eq_con.columns):
    fig.add_trace(go.Scatter(
        x=eq_con.index, y=eq_con[sym],
        name=sym,
        line=dict(color=colours[i % len(colours)], width=1, dash="dot"),
        opacity=0.5,
    ))

fig.add_trace(go.Scatter(
    x=port_eq.index, y=port_eq,
    name="Equal-weight Portfolio",
    line=dict(color=PALETTE["accent_yellow"], width=2.5),
))

apply_theme(fig, title="Equal-Weight Portfolio vs Individual Symbols (Conservative)", height=460)
fig.update_layout(yaxis_title="Equity (normalised to 1.0)")
fig.show()

# %% [markdown]
# ## 5. Capital Models — Side-by-Side
#
# Three ways to aggregate individual symbol returns into a portfolio NAV:
#
# | Model | Within a trade | Between trades | Use when |
# |---|---|---|---|
# | **`rebalanced`** | Daily reset to `1/N_universe` weight | Geometric | Comparing signal quality independent of capital size |
# | **`buy_and_hold`** | Compounds from `1/N_universe` entry | Geometric (winners grow) | Long-only momentum with no rebalancing |
# | **`fixed_stake`** | Compounds from `$amount_per_entry` | Arithmetic (P&L to cash) | Fixed-size position sizing |
#
# All three curves start at 1.0 (NAV divided by `amount × N_universe`) so they are directly comparable.

# %%
port_rebalanced = alloc.portfolio_equity(method="conservative", mode="rebalanced")
port_buy_hold   = alloc.portfolio_equity(method="conservative", mode="buy_and_hold")
port_fixed      = alloc.portfolio_equity(method="conservative", mode="fixed_stake", amount_per_entry=1_000)

fig = go.Figure()

for i, sym in enumerate(eq_con.columns):
    fig.add_trace(go.Scatter(
        x=eq_con.index, y=eq_con[sym],
        name=sym, showlegend=True,
        line=dict(color=colours[i % len(colours)], width=1, dash="dot"),
        opacity=0.35,
    ))

fig.add_trace(go.Scatter(
    x=port_rebalanced.index, y=port_rebalanced,
    name="Rebalanced (1/N daily)",
    line=dict(color=PALETTE["accent_yellow"], width=2, dash="dash"),
))
fig.add_trace(go.Scatter(
    x=port_buy_hold.index, y=port_buy_hold,
    name="Buy & Hold (let it run)",
    line=dict(color=PALETTE["accent_green"], width=2),
))
fig.add_trace(go.Scatter(
    x=port_fixed.index, y=port_fixed,
    name="Fixed $1k/entry",
    line=dict(color=PALETTE["accent_blue"], width=2.5),
))

apply_theme(fig, title="Portfolio Capital Models — Conservative Fill", height=500)
fig.update_layout(yaxis_title="NAV (normalised to 1.0)")
fig.show()


# %% [markdown]
# ## 6. Performance Metrics
#
# Risk-adjusted metrics for each capital model using conservative fill.  Sharpe and
# Calmar are the primary ranking metrics; max drawdown sets the risk budget ceiling.

# %%
def _metrics_row(pm) -> dict:
    return {
        "Total Return": f"{pm.total_return:+.1%}",
        "CAGR":         f"{pm.annualised_return:+.1%}",
        "Ann. Vol":     f"{pm.annualised_vol:.1%}",
        "Sharpe":       f"{pm.sharpe:.2f}",
        "Sortino":      f"{pm.sortino:.2f}",
        "Max Drawdown": f"{pm.max_drawdown:.1%}",
        "Calmar":       f"{pm.calmar:.2f}",
    }


rows = {
    "Rebalanced":       _metrics_row(alloc.portfolio_metrics(method="conservative", mode="rebalanced")),
    "Buy & Hold":       _metrics_row(alloc.portfolio_metrics(method="conservative", mode="buy_and_hold")),
    "Fixed $1k/entry":  _metrics_row(alloc.portfolio_metrics(method="conservative", mode="fixed_stake", amount_per_entry=1_000)),
}

pd.DataFrame(rows).T

# %% [markdown]
# ## 7. Full Metrics — Rebalanced Model
#
# Complete `PerformanceMetrics` summary for the rebalanced portfolio (conservative fill).
# Includes VaR, CVaR, skewness, kurtosis, and Omega ratio.

# %%
alloc.portfolio_metrics(method="conservative", mode="rebalanced").summary()
