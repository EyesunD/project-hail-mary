"""Tests for Portfolio and Trade mechanics."""

import pandas as pd
import pytest
from datetime import datetime

from hailmary.backtest.portfolio import Portfolio, Trade


def test_portfolio_nav_equals_initial_with_no_positions() -> None:
    p = Portfolio(1_000_000)
    prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0})
    assert p.nav(prices) == 1_000_000


def test_execute_trade_updates_cash() -> None:
    p = Portfolio(100_000)
    t = Trade(datetime(2020, 1, 1), "AAPL", quantity=10, price=150.0)
    p.execute_trade(t)
    assert p.cash == pytest.approx(100_000 - 10 * 150.0)
    assert p.positions["AAPL"].quantity == 10


def test_nav_reflects_market_value() -> None:
    p = Portfolio(100_000)
    t = Trade(datetime(2020, 1, 1), "AAPL", quantity=10, price=100.0)
    p.execute_trade(t)
    prices = pd.Series({"AAPL": 150.0})
    # cash = 90_000, position = 10 * 150 = 1500 → nav = 91_500
    assert p.nav(prices) == pytest.approx(91_500.0)


def test_set_weights_produces_trades(price_df: pd.DataFrame) -> None:
    p = Portfolio(1_000_000)
    prices = price_df.iloc[-1]
    weights = pd.Series(0.1, index=price_df.columns[:5])
    trades = p.set_weights(weights, prices, datetime(2023, 1, 1))
    assert len(trades) > 0


def test_trade_log_structure(price_df: pd.DataFrame) -> None:
    p = Portfolio(1_000_000)
    prices = price_df.iloc[-1]
    weights = pd.Series(0.2, index=price_df.columns[:3])
    p.set_weights(weights, prices, datetime(2023, 1, 1))
    log = p.trade_log
    assert "symbol" in log.columns
    assert "quantity" in log.columns
