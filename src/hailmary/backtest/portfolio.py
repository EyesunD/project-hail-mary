"""Portfolio state tracker — positions, cash, NAV history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class Trade:
    timestamp: datetime
    symbol: str
    quantity: float   # positive = buy, negative = sell
    price: float
    commission: float = 0.0

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.price)

    @property
    def cost(self) -> float:
        return self.quantity * self.price + self.commission


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-9

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealised_pnl(self, price: float) -> float:
        return (price - self.avg_cost) * self.quantity

    def update(self, quantity: float, price: float) -> None:
        if self.quantity + quantity == 0:
            self.avg_cost = 0.0
        elif self.quantity == 0 or (self.quantity > 0) == (quantity > 0):
            self.avg_cost = (self.quantity * self.avg_cost + quantity * price) / (self.quantity + quantity)
        # reducing position: avg_cost unchanged
        self.quantity += quantity


class Portfolio:
    """Track cash, positions, and NAV through a backtest.

    All monetary values are in the *account currency*.
    """

    def __init__(self, initial_capital: float = 1_000_000.0) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self._nav_history: list[tuple[datetime, float]] = []

    # ------------------------------------------------------------------ API

    def execute_trade(self, trade: Trade) -> None:
        sym = trade.symbol
        if sym not in self.positions:
            self.positions[sym] = Position(sym)
        self.positions[sym].update(trade.quantity, trade.price)
        self.cash -= trade.cost
        self.trades.append(trade)

    def set_weights(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        timestamp: datetime,
        execution_model: object | None = None,
    ) -> list[Trade]:
        """Rebalance portfolio to target weights.  Returns list of Trades executed."""
        nav = self.nav(prices)
        trades = []
        for sym, w in target_weights.items():
            target_value = w * nav
            current_value = self.positions.get(sym, Position(sym)).market_value(prices.get(sym, 0))
            delta_value = target_value - current_value
            price = prices.get(sym)
            if price is None or price <= 0:
                continue
            quantity = delta_value / price
            if abs(quantity) < 1e-6:
                continue
            if execution_model is not None:
                price = execution_model.fill_price(sym, price, quantity)
                commission = execution_model.commission(quantity, price)
            else:
                commission = 0.0
            trade = Trade(timestamp=timestamp, symbol=sym, quantity=quantity,
                          price=price, commission=commission)
            self.execute_trade(trade)
            trades.append(trade)
        # Close positions not in target
        for sym in list(self.positions):
            if sym not in target_weights.index and not self.positions[sym].is_flat:
                price = prices.get(sym, 0)
                if price <= 0:
                    continue
                qty = -self.positions[sym].quantity
                trade = Trade(timestamp=timestamp, symbol=sym, quantity=qty,
                              price=price, commission=0.0)
                self.execute_trade(trade)
                trades.append(trade)
        return trades

    def nav(self, prices: pd.Series) -> float:
        """Net asset value: cash + market value of all positions."""
        mv = sum(
            pos.market_value(prices.get(sym, pos.avg_cost))
            for sym, pos in self.positions.items()
            if not pos.is_flat
        )
        return self.cash + mv

    def snapshot(self, timestamp: datetime, prices: pd.Series) -> float:
        """Record NAV at *timestamp* and return it."""
        n = self.nav(prices)
        self._nav_history.append((timestamp, n))
        return n

    @property
    def nav_series(self) -> pd.Series:
        if not self._nav_history:
            return pd.Series(dtype=float)
        ts, vals = zip(*self._nav_history)
        return pd.Series(vals, index=pd.DatetimeIndex(ts), name="nav")

    @property
    def trade_log(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(
            [{"timestamp": t.timestamp, "symbol": t.symbol,
              "quantity": t.quantity, "price": t.price,
              "notional": t.notional, "commission": t.commission}
             for t in self.trades]
        ).set_index("timestamp")
