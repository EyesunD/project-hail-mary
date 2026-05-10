# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 — Market Data Providers
#
# Entry point to the abstract data layer. Fetches daily crypto OHLCV from Yahoo via
# `YahooFinanceProvider`, then reshapes the MultiIndex `(symbol, timestamp)` bar frame
# into the two layouts the rest of the platform consumes.
#
# **Pipeline**
# ```
# YahooFinanceProvider  →  TrendSignal  →  BarBacktest  →  SignalTradePerformance       (iid bets)
#       (bars)            (signal cols)   (bt_result)    └─  SignalAllocationPerformance (capital deployed)
#         ▲
#    this notebook
# ```
#
# **What this notebook covers**
#
# 1. **Provider instantiation** — `YahooFinanceProvider` with optional `DataCache`
#    (Parquet on disk) and `ProviderRegistry` for priority-based fallback.
# 2. **`get_bars()`** — daily OHLCV for a basket of crypto symbols, returned as a
#    long-format `(symbol, timestamp)` MultiIndex DataFrame.
# 3. **Wide reshape** — `bars["close"].unstack("symbol")` to get the
#    `(date × symbol)` close-price matrix used downstream.
# 4. **`get_returns()`** — convenience helper that returns a wide returns matrix
#    directly, skipping the manual reshape.

# %% jupyter={"is_executing": true}
from hailmary.data import DataCache, ProviderRegistry
from hailmary.data.providers import YahooFinanceProvider
from hailmary.data.base import Timeframe
import pandas as pd
pd.set_option('display.max_columns', 330)


cache = DataCache('../../data/cache', ttl_hours=48)
# yahoo = YahooFinanceProvider(cache=cache)
yahoo = YahooFinanceProvider(cache=None)

# registry = ProviderRegistry()
# registry.register(yahoo, priority=10)
# print(registry)

# %%
# Fetch daily OHLCV for a basket of equities
symbols = [
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "SOL-USD",  # Solana
    "BNB-USD",  # BNB
    # "XRP-USD",  # XRP
    # "ADA-USD",  # Cardano
    # "DOGE-USD", # Dogecoin
    # "AVAX-USD", # Avalanche
    # "LINK-USD", # Chainlink
    # "LTC-USD",  # Litecoin
]
start=pd.to_datetime('2022-01-01')
end=pd.to_datetime('2024-01-01')
bars = yahoo.get_bars(symbols, start=start, end=end)
assert len(bars.index.get_level_values('symbol').unique()) == len(symbols)
assert set(bars.index.get_level_values('symbol').unique()) == set(symbols)
assert bars.index.get_level_values("timestamp").min().strftime("%Y-%m-%d") == start.strftime("%Y-%m-%d")
assert bars.index.get_level_values("timestamp").max().strftime("%Y-%m-%d") == end.strftime("%Y-%m-%d")

print(bars.shape)
bars.head(4)

# %%
# Wide close-price DataFrame
close = bars['close'].unstack(level=0)
close.tail()

# %%
# Convenience: get returns directly
returns = yahoo.get_returns(symbols,
                            start=pd.Timestamp("2022-01-01"),
                            end=pd.Timestamp("2024-01-01"))
returns.describe()
