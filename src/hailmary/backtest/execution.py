"""Execution models: slippage, commissions, market impact."""

from __future__ import annotations

import abc

import numpy as np


class SlippageModel(abc.ABC):
    """Abstract slippage model — override :meth:`slippage_bps` to customise."""

    @abc.abstractmethod
    def slippage_bps(self, symbol: str, price: float, quantity: float) -> float:
        """Return one-way slippage in basis points."""

    def fill_price(self, symbol: str, price: float, quantity: float) -> float:
        bps = self.slippage_bps(symbol, price, quantity)
        direction = 1 if quantity > 0 else -1  # buys pay up, sells pay down
        return price * (1 + direction * bps / 10_000)


class FixedSlippage(SlippageModel):
    """Constant slippage in basis points regardless of size."""

    def __init__(self, bps: float = 5.0) -> None:
        self.bps = bps

    def slippage_bps(self, symbol: str, price: float, quantity: float) -> float:
        return self.bps


class VolumeSlippage(SlippageModel):
    """Slippage proportional to participation rate (simplified square-root model)."""

    def __init__(
        self,
        base_bps: float = 2.0,
        impact_coefficient: float = 0.1,
        adv_map: dict[str, float] | None = None,
    ) -> None:
        self.base_bps = base_bps
        self.impact_coefficient = impact_coefficient
        self.adv_map = adv_map or {}

    def slippage_bps(self, symbol: str, price: float, quantity: float) -> float:
        adv = self.adv_map.get(symbol, 1_000_000.0)  # default $1M ADV
        participation = abs(quantity * price) / adv
        impact = self.impact_coefficient * np.sqrt(participation) * 100  # to bps
        return self.base_bps + impact


class CommissionModel(abc.ABC):
    @abc.abstractmethod
    def commission(self, quantity: float, price: float) -> float:
        """Return total commission for a trade in dollars."""


class ZeroCommission(CommissionModel):
    def commission(self, quantity: float, price: float) -> float:
        return 0.0


class PerShareCommission(CommissionModel):
    def __init__(self, per_share: float = 0.005, min_commission: float = 1.0) -> None:
        self.per_share = per_share
        self.min_commission = min_commission

    def commission(self, quantity: float, price: float) -> float:
        return max(abs(quantity) * self.per_share, self.min_commission)


class PercentCommission(CommissionModel):
    def __init__(self, pct: float = 0.001) -> None:
        self.pct = pct

    def commission(self, quantity: float, price: float) -> float:
        return abs(quantity * price) * self.pct


class ExecutionModel:
    """Combine a slippage model and a commission model."""

    def __init__(
        self,
        slippage: SlippageModel | None = None,
        commissions: CommissionModel | None = None,
    ) -> None:
        self.slippage = slippage or FixedSlippage()
        self.commissions = commissions or ZeroCommission()

    def fill_price(self, symbol: str, price: float, quantity: float) -> float:
        return self.slippage.fill_price(symbol, price, quantity)

    def commission(self, quantity: float, price: float) -> float:
        return self.commissions.commission(quantity, price)
