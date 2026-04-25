"""Performance metrics — all standard quant risk-adjusted return statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class PerformanceMetrics:
    """Compute a full suite of performance metrics from a return series.

    All metrics default to 252 trading days per year.
    """

    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> None:
        self.returns = returns.dropna()
        self.rf = risk_free_rate
        self.ppy = periods_per_year

    # ----------------------------------------------------------------- returns

    @property
    def total_return(self) -> float:
        return float((1 + self.returns).prod() - 1)

    @property
    def annualised_return(self) -> float:
        n = len(self.returns)
        return float((1 + self.total_return) ** (self.ppy / n) - 1)

    @property
    def cagr(self) -> float:
        return self.annualised_return

    # ----------------------------------------------------------------- risk

    @property
    def annualised_vol(self) -> float:
        return float(self.returns.std() * np.sqrt(self.ppy))

    @property
    def downside_vol(self) -> float:
        neg = self.returns[self.returns < 0]
        return float(neg.std() * np.sqrt(self.ppy)) if len(neg) > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        cum = (1 + self.returns).cumprod()
        roll_max = cum.cummax()
        dd = (cum - roll_max) / roll_max
        return float(dd.min())

    @property
    def drawdown_series(self) -> pd.Series:
        cum = (1 + self.returns).cumprod()
        roll_max = cum.cummax()
        return (cum - roll_max) / roll_max

    @property
    def avg_drawdown(self) -> float:
        dd = self.drawdown_series
        return float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    @property
    def drawdown_duration(self) -> pd.Timedelta | None:
        """Longest period spent underwater (calendar time)."""
        dd = self.drawdown_series
        in_dd = dd < 0
        max_dur: pd.Timedelta | None = None
        start: pd.Timestamp | None = None
        for ts, is_dd in in_dd.items():
            if is_dd and start is None:
                start = ts  # type: ignore[assignment]
            elif not is_dd and start is not None:
                dur = ts - start  # type: ignore[operator]
                if max_dur is None or dur > max_dur:
                    max_dur = dur
                start = None
        return max_dur

    # ----------------------------------------------------------------- ratios

    @property
    def sharpe(self) -> float:
        excess = self.returns.mean() * self.ppy - self.rf
        vol = self.annualised_vol
        return excess / vol if vol > 0 else 0.0

    @property
    def sortino(self) -> float:
        excess = self.returns.mean() * self.ppy - self.rf
        dvol = self.downside_vol
        return excess / dvol if dvol > 0 else 0.0

    @property
    def calmar(self) -> float:
        mdd = abs(self.max_drawdown)
        return self.annualised_return / mdd if mdd > 0 else float("inf")

    @property
    def omega(self) -> float:
        threshold = self.rf / self.ppy
        gains = (self.returns - threshold).clip(lower=0).sum()
        losses = (threshold - self.returns).clip(lower=0).sum()
        return gains / losses if losses > 0 else float("inf")

    # ----------------------------------------------------------------- distribution

    @property
    def skewness(self) -> float:
        return float(self.returns.skew())

    @property
    def kurtosis(self) -> float:
        return float(self.returns.kurtosis())

    @property
    def var_95(self) -> float:
        return float(self.returns.quantile(0.05))

    @property
    def cvar_95(self) -> float:
        return float(self.returns[self.returns <= self.var_95].mean())

    @property
    def tail_ratio(self) -> float:
        p95 = abs(self.returns.quantile(0.95))
        p05 = abs(self.returns.quantile(0.05))
        return p95 / p05 if p05 > 0 else float("inf")

    # ----------------------------------------------------------------- vs benchmark

    def beta(self, benchmark_returns: pd.Series) -> float:
        aligned = pd.concat([self.returns, benchmark_returns], axis=1).dropna()
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0

    def alpha(self, benchmark_returns: pd.Series) -> float:
        b = self.beta(benchmark_returns)
        bm_ann = float((1 + benchmark_returns.mean()) ** self.ppy - 1)
        return self.annualised_return - (self.rf + b * (bm_ann - self.rf))

    def information_ratio(self, benchmark_returns: pd.Series) -> float:
        active = self.returns - benchmark_returns.reindex(self.returns.index).fillna(0)
        tracking_error = float(active.std() * np.sqrt(self.ppy))
        excess = float(active.mean() * self.ppy)
        return excess / tracking_error if tracking_error > 0 else 0.0

    def up_capture(self, benchmark_returns: pd.Series) -> float:
        up = benchmark_returns > 0
        if not up.any():
            return float("nan")
        return float(self.returns[up].mean() / benchmark_returns[up].mean())

    def down_capture(self, benchmark_returns: pd.Series) -> float:
        down = benchmark_returns < 0
        if not down.any():
            return float("nan")
        return float(self.returns[down].mean() / benchmark_returns[down].mean())

    # ----------------------------------------------------------------- summary

    def summary(self, benchmark_returns: pd.Series | None = None) -> pd.Series:
        d: dict[str, object] = {
            "Total Return": f"{self.total_return:.2%}",
            "CAGR": f"{self.cagr:.2%}",
            "Ann. Volatility": f"{self.annualised_vol:.2%}",
            "Sharpe Ratio": f"{self.sharpe:.2f}",
            "Sortino Ratio": f"{self.sortino:.2f}",
            "Calmar Ratio": f"{self.calmar:.2f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Avg Drawdown": f"{self.avg_drawdown:.2%}",
            "Skewness": f"{self.skewness:.2f}",
            "Kurtosis": f"{self.kurtosis:.2f}",
            "VaR 95%": f"{self.var_95:.2%}",
            "CVaR 95%": f"{self.cvar_95:.2%}",
            "Omega Ratio": f"{self.omega:.2f}",
            "Tail Ratio": f"{self.tail_ratio:.2f}",
        }
        if benchmark_returns is not None:
            d["Beta"] = f"{self.beta(benchmark_returns):.2f}"
            d["Alpha"] = f"{self.alpha(benchmark_returns):.2%}"
            d["Information Ratio"] = f"{self.information_ratio(benchmark_returns):.2f}"
            d["Up Capture"] = f"{self.up_capture(benchmark_returns):.2%}"
            d["Down Capture"] = f"{self.down_capture(benchmark_returns):.2%}"
        return pd.Series(d)

    # ----------------------------------------------------------------- rolling

    def rolling_sharpe(self, window: int = 252) -> pd.Series:
        roll_mean = self.returns.rolling(window).mean() * self.ppy
        roll_std = self.returns.rolling(window).std() * np.sqrt(self.ppy)
        return (roll_mean - self.rf) / roll_std

    def rolling_drawdown(self) -> pd.Series:
        return self.drawdown_series

    def monthly_returns(self) -> pd.DataFrame:
        monthly = self.returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)
        return monthly.unstack_with_resampler(monthly.index.year, monthly.index.month) if False else (
            pd.DataFrame({
                "year": monthly.index.year,
                "month": monthly.index.month,
                "return": monthly.values,
            }).pivot(index="year", columns="month", values="return")
        )
