"""Tests for TrendSignal."""

from __future__ import annotations

import pandas as pd

from hailmary.models.signals import TrendSignal

SIGNAL_COLS = {
    "ma", "signal_close", "signal_open", "trade_direction", "turnover",
    "enter", "exit", "cycle", "signal_age",
}


def test_trend_signal_returns_dataframe(bars_df: pd.DataFrame) -> None:
    result = TrendSignal(ma_window=200).run(bars_df)
    assert isinstance(result, pd.DataFrame)


def test_trend_signal_columns_present(bars_df: pd.DataFrame) -> None:
    result = TrendSignal(ma_window=200).run(bars_df)
    assert SIGNAL_COLS.issubset(set(result.columns))


def test_trend_signal_no_lookahead(bars_df: pd.DataFrame) -> None:
    result = TrendSignal(ma_window=200).run(bars_df)
    for sym in bars_df.index.get_level_values("symbol").unique():
        d = result.xs(sym, level="symbol")
        expected = d["signal_close"].shift(1).fillna(0).astype(int)
        pd.testing.assert_series_equal(d["signal_open"], expected, check_names=False)


def test_trend_signal_enter_exit_balanced(bars_df: pd.DataFrame) -> None:
    result = TrendSignal(ma_window=200).run(bars_df)
    for sym in bars_df.index.get_level_values("symbol").unique():
        d = result.xs(sym, level="symbol")
        assert d["enter"].sum() - d["exit"].sum() in (0, 1)


def test_trend_signal_no_execution_columns(bars_df: pd.DataFrame) -> None:
    result = TrendSignal(ma_window=200).run(bars_df)
    execution_cols = {"return_mark_to_close", "return_conservative", "return_net",
                      "position_start", "position_end", "trade_cycle_id"}
    assert execution_cols.isdisjoint(set(result.columns))


def test_trend_signal_warmup_property() -> None:
    assert TrendSignal(ma_window=200).warmup == 200
    assert TrendSignal(ma_window=50).warmup == 50


def test_trend_signal_trim_start(bars_df: pd.DataFrame) -> None:
    signal = TrendSignal(ma_window=50)
    trim_ts = bars_df.index.get_level_values("timestamp").unique()[100]
    result = signal.run(bars_df, trim_start=trim_ts)
    assert result.index.get_level_values("timestamp").min() >= trim_ts


def test_trend_signal_trim_start_ma_not_nan(bars_df: pd.DataFrame) -> None:
    signal = TrendSignal(ma_window=50)
    trim_ts = bars_df.index.get_level_values("timestamp").unique()[60]
    result = signal.run(bars_df, trim_start=trim_ts)
    assert result["ma"].notna().all()


def _make_trending_bars(n: int = 300, ma_window: int = 50) -> pd.DataFrame:
    """Build a single-symbol bars DataFrame with a sustained uptrend so signal_open=1
    throughout the post-warmup period."""
    idx = pd.bdate_range("2020-01-01", periods=n, name="timestamp")
    close = 100.0 * (1.002 ** pd.RangeIndex(n))  # steady 0.2 % daily rise
    close = close.values
    df = pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": 1_000_000.0,
    }, index=idx)
    df["symbol"] = "TREND"
    return df.reset_index().set_index(["symbol", "timestamp"])


def test_trend_signal_trim_start_already_in_signal() -> None:
    """A symbol in signal before trim_start must show enter=1 on the first trimmed bar."""
    ma_window = 50
    bars = _make_trending_bars(n=300, ma_window=ma_window)
    signal = TrendSignal(ma_window=ma_window)
    timestamps = bars.index.get_level_values("timestamp").unique()

    # Trim at bar 200 — well past warmup, symbol has been in signal since ~bar 51
    trim_ts = timestamps[200]

    first = signal.run(bars, trim_start=trim_ts).xs("TREND", level="symbol").iloc[0]

    assert first["signal_open"] == 1,     "expected to be in signal at trim boundary"
    assert first["enter"] == 1,           "expected enter=1 on first trimmed bar"
    assert first["trade_direction"] == 1, "expected trade_direction=+1"
    assert first["turnover"] == 1,        "expected turnover=1"
    assert first["cycle"] == "init",      "expected cycle='init'"
    assert first["signal_age"] == 1,      "expected signal_age=1"


def test_trend_signal_trim_no_nan_returns_in_backtest() -> None:
    """BarBacktest must produce no NaN returns when signal is trimmed mid-position."""
    from hailmary.backtest.signal_backtest import BarBacktest

    bars = _make_trending_bars(n=300, ma_window=50)
    signal_df = TrendSignal(ma_window=50).run(
        bars, trim_start=bars.index.get_level_values("timestamp").unique()[200]
    )
    result = BarBacktest().run(signal_df)

    for col in ("return_mark_to_close", "return_conservative", "return_net"):
        in_signal = result.data[result.data["signal_open"] == 1]
        assert not in_signal[col].isna().any(), f"NaN in {col} for in-signal bars"
