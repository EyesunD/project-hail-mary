"""Core backtesting engine — event-driven, strategy-agnostic."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import pandas as pd
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from hailmary.backtest.execution import ExecutionModel
from hailmary.backtest.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class Strategy(abc.ABC):
    """Abstract trading strategy.

    Implement :meth:`generate_weights` to define your alpha logic.
    The method is called at each rebalance bar and should return target
    portfolio weights as a ``pd.Series(symbol → weight)``.
    """

    @abc.abstractmethod
    def generate_weights(
        self,
        prices: pd.DataFrame,
        timestamp: pd.Timestamp,
        portfolio: Portfolio,
        **context: object,
    ) -> pd.Series:
        """Return target portfolio weights.

        Args:
            prices:    Historical OHLCV with all data up to *timestamp*.
            timestamp: Current bar timestamp.
            portfolio: Current portfolio state.
            **context: Extra data passed from the engine (e.g. fundamentals).

        Returns:
            pd.Series: target weights (index=symbol). Must sum to ≤1.
        """


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Complete backtest result."""

    nav: pd.Series                                 # daily NAV time series
    returns: pd.Series                             # daily portfolio returns
    weights: pd.DataFrame                          # end-of-day portfolio weights
    trade_log: pd.DataFrame                        # all executed trades
    benchmark_nav: pd.Series | None = None        # optional benchmark NAV
    metadata: dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------ computed

    @property
    def total_return(self) -> float:
        return float(self.nav.iloc[-1] / self.nav.iloc[0] - 1)

    @property
    def annualised_return(self) -> float:
        n_years = len(self.nav) / 252
        return float((1 + self.total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    @property
    def annualised_vol(self) -> float:
        return float(self.returns.std() * (252 ** 0.5))

    @property
    def sharpe(self) -> float:
        rf = self.metadata.get("risk_free_rate", 0.0)
        excess = self.returns.mean() * 252 - float(rf)
        vol = self.annualised_vol
        return excess / vol if vol > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        roll_max = self.nav.cummax()
        dd = (self.nav - roll_max) / roll_max
        return float(dd.min())

    @property
    def calmar(self) -> float:
        mdd = abs(self.max_drawdown)
        return self.annualised_return / mdd if mdd > 0 else float("inf")

    def summary(self) -> pd.Series:
        return pd.Series({
            "Total Return": f"{self.total_return:.2%}",
            "Ann. Return": f"{self.annualised_return:.2%}",
            "Ann. Volatility": f"{self.annualised_vol:.2%}",
            "Sharpe Ratio": f"{self.sharpe:.2f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Calmar Ratio": f"{self.calmar:.2f}",
            "Num Trades": str(len(self.trade_log)),
        })


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Vectorised backtest engine.

    Usage::

        engine = BacktestEngine(
            bars=bars,
            strategy=my_strategy,
            initial_capital=1_000_000,
        )
        result = engine.run(start="2018-01-01", end="2023-12-31")
        result.summary()
    """

    def __init__(
        self,
        strategy: Strategy,
        *,
        bars: pd.DataFrame,
        initial_capital: float = 1_000_000.0,
        execution: ExecutionModel | None = None,
        benchmark: pd.Series | None = None,
        risk_free_rate: float = 0.0,
        fill_mode: Literal["mtc", "conservative"] = "mtc",
        **strategy_context: object,
    ) -> None:
        self.prices = bars["close"].unstack("symbol").sort_index()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.execution = execution or ExecutionModel()
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        self.fill_mode = fill_mode
        self.strategy_context = strategy_context
        self._open_df: pd.DataFrame | None = bars["open"].unstack("symbol") if fill_mode == "mtc" else None
        self._high_df: pd.DataFrame | None = bars["high"].unstack("symbol") if fill_mode == "conservative" else None
        self._low_df: pd.DataFrame | None = bars["low"].unstack("symbol") if fill_mode == "conservative" else None

    # ------------------------------------------------------------------ API

    def run(
        self,
        start: str | date | None = None,
        end: str | date | None = None,
        *,
        verbose: bool = True,
    ) -> BacktestResult:
        """Run the backtest over [start, end] and return a :class:`BacktestResult`."""
        prices = self._slice(start, end)
        portfolio = Portfolio(self.initial_capital)

        nav_records: list[tuple[pd.Timestamp, float]] = []
        weight_records: list[tuple[pd.Timestamp, pd.Series]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            disable=not verbose,
        ) as progress:
            task = progress.add_task("Running backtest...", total=len(prices))
            for ts, row in prices.iterrows():
                current_prices = row.dropna()
                hist = prices.loc[:ts]
                open_prices = self._open_df.loc[ts] if self._open_df is not None and ts in self._open_df.index else None
                high_prices = self._high_df.loc[ts] if self._high_df is not None and ts in self._high_df.index else None
                low_prices = self._low_df.loc[ts] if self._low_df is not None and ts in self._low_df.index else None
                try:
                    weights = self.strategy.generate_weights(
                        hist, ts, portfolio, **self.strategy_context
                    )
                    portfolio.set_weights(weights, current_prices, ts, self.execution,
                                          high_prices=high_prices, low_prices=low_prices,
                                          open_prices=open_prices)
                    weight_records.append((ts, weights))
                except Exception as exc:
                    logger.warning("Strategy failed at {}: {}", ts, exc)

                nav = portfolio.snapshot(ts, current_prices)
                nav_records.append((ts, nav))
                progress.advance(task)

        nav_ts, nav_vals = zip(*nav_records)
        nav_series = pd.Series(nav_vals, index=pd.DatetimeIndex(nav_ts), name="nav")
        returns = nav_series.pct_change().dropna()

        weights_df = pd.DataFrame(
            {ts: w for ts, w in weight_records}
        ).T if weight_records else pd.DataFrame()

        benchmark_nav = self._normalise_benchmark(nav_series) if self.benchmark is not None else None

        return BacktestResult(
            nav=nav_series,
            returns=returns,
            weights=weights_df,
            trade_log=portfolio.trade_log,
            benchmark_nav=benchmark_nav,
            metadata={"risk_free_rate": self.risk_free_rate, "fill_mode": self.fill_mode},
        )

    # ----------------------------------------------------------------- private

    def _slice(self, start: str | date | None, end: str | date | None) -> pd.DataFrame:
        df = self.prices
        if start:
            df = df.loc[str(start):]
        if end:
            df = df.loc[:str(end)]
        return df

    def _normalise_benchmark(self, nav: pd.Series) -> pd.Series:
        bm = self.benchmark.reindex(nav.index, method="ffill")  # type: ignore[union-attr]
        return bm / bm.iloc[0] * self.initial_capital
