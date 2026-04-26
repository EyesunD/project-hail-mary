# %%
import numpy as np
from hailmary.data.providers import YahooFinanceProvider
# import yfinance as yf
yahoo = YahooFinanceProvider(cache=None)
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)
symbols = [
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "SOL-USD",  # Solana
    # "BNB-USD",  # BNB
    # "XRP-USD",  # XRP
    # "ADA-USD",  # Cardano
    # "DOGE-USD", # Dogecoin
    # "AVAX-USD", # Avalanche
    # "LINK-USD", # Chainlink
    # "LTC-USD",  # Litecoin
]
start=pd.to_datetime('2022-01-01')
end=pd.to_datetime('2024-01-01')

yahoo = YahooFinanceProvider(cache=None)
bars= yahoo.get_bars(symbols, start=start, end=end,adjust = False)
assert len(bars.index.get_level_values('symbol').unique()) == len(symbols)
assert set(bars.index.get_level_values('symbol').unique()) == set(symbols)
assert bars.index.get_level_values("timestamp").min().strftime("%Y-%m-%d") == start.strftime("%Y-%m-%d")
assert bars.index.get_level_values("timestamp").max().strftime("%Y-%m-%d") == end.strftime("%Y-%m-%d")
print(bars.shape)
print(bars.shape)
bars.head(4)


# %%
returns = yahoo.get_returns(symbols, start=start, end=end,adjust = False)
returns.head(4)


# %%
import numpy as np
def new_trade_id(x):
    return np.logical_and(
        np.sign(x) != np.sign(x.shift(1)),
        ~x.shift(1).ffill().isin([0])
    ) | (x.isin([0]) & ~x.shift(-1).ffill().isin([0]))

# worst_price = True
sym = 'BTC-USD'
out = (
    bars
    # .merge(returns.stack("symbol").rename("return"), left_index=True, right_index=True, how="outer")
    # .query("symbol == @sym")
    .assign(
        ma200=lambda df: df.groupby(level="symbol")["close"].transform(
            lambda s: s.rolling(200).mean()
        )
    )
    # Signal known after today's close
    .assign(
        signal_close=lambda x: (x["close"] > x["ma200"]).astype(int)
    )
    # Signal known at next day's open
    .assign(
        signal_open=lambda df: df.groupby(level="symbol")["signal_close"].transform(
            lambda s: s.shift(1).fillna(0).astype(int)
        )
    )
    .assign(
        trade_direction =lambda df: df.groupby(level="symbol")["signal_open"].transform(
            lambda s: s.diff()
        ),
        turnover  =lambda df: df.groupby(level="symbol")["signal_open"].transform(
            lambda s: s.diff().abs()
        ),
        enter=lambda df: df.groupby(level="symbol")["signal_open"].transform(
            lambda s: np.where(~s.isin([0]) & s.shift(1).fillna(0).isin([0]), 1, 0)
        ),
        exit=lambda df: df.groupby(level="symbol")["signal_open"].transform(
            lambda s: np.where(s.isin([0]) & ~s.shift(1).fillna(0).isin([0]), 1, 0)
        )
    )
    .assign(
        cycle =  lambda df: np.select(
            [
                df["enter"].eq(1),
                df["exit"].eq(1),
                df["signal_open"].eq(1),
            ],
            [
                'init',
                'exit',
                'held',
            ],
            default='None',
        )
    )
    .assign(
        return_conservative=lambda df: np.select(
            [
                df["enter"].eq(1),
                df["exit"].eq(1),
                df["signal_open"].eq(1),
            ],
            [
                (df["close"] / df["high"]) - 1,
                (df["low"] / df.groupby(level="symbol")["close"].shift(1)) - 1,
                (df["close"] / df.groupby(level="symbol")["close"].shift(1)) - 1,
            ],
            default=0.0,
        )
    )
    # entry: open → close
    # held: previous close → close
    # exit: previous close → open
    .assign(
        return_mark_to_close=lambda df: np.select(
            [
                df["enter"].eq(1),
                df["exit"].eq(1),
                df["signal_open"].eq(1),
            ],
            [
                (df["close"] / df["open"]) - 1,
                (df["open"] / df.groupby(level="symbol")["close"].shift(1)) - 1,
                (df["close"] / df.groupby(level="symbol")["close"].shift(1)) - 1,
            ],
            default=0.0,
        )
    )
    .assign(
        signal_age=lambda df: df.groupby(level="symbol")["signal_open"].transform(
            lambda s: (
                (
                        s.fillna(0).eq(1) |
                        ((s.fillna(0).shift() == 1) & (s.fillna(0) == 0))
                )
                .astype(int)
                .groupby(
                    (~(s.fillna(0).eq(1) | ((s.fillna(0).shift() == 1) & (s.fillna(0) == 0)))).cumsum()
                )
                .cumsum()
                .astype("Int64")
            )
        )
    )
    # on entry, signal_open = 1, so we need to buy intraday; so position_open = 0
    # holding, means signal_open = 1, so position_open = 1;
    # on exit, signal_open = 0, but exist = 1 so position_open = 1
    .assign(
        position_start=lambda s: np.where(s['cycle'].isin(['exit','held']),1,0),
        position_end=lambda s: np.where(s['cycle'].isin(['init','held']),1,0),
        strategy_equity_mark_to_close=lambda df: (1 + df["return_mark_to_close"]).groupby(level="symbol").cumprod(),
        strategy_equity_conservative=lambda df: (1 + df["return_conservative"]).groupby(level="symbol").cumprod(),
    )
    .assign(
        trade_cycle_id=lambda df: df.groupby(level="symbol")["position_start"].transform(
            lambda s: (
               np.where(new_trade_id(s), 1, 0)
            ).cumsum()
        )
    )
    # .unstack('symbol')
    # .to_clipboard()
    # .tail()
)
out.groupby(level="symbol").agg(
    total_return=("strategy_equity_mark_to_close", lambda x: x.iloc[-1] - 1),
    num_entries=("enter", "sum"),
    num_exits=("exit", "sum"),
    invested_days=("position_end", "sum"),
)
# out.unstack('symbol').to_clipboard()
