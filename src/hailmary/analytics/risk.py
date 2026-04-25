"""Risk analytics: covariance estimation, factor risk decomposition, VaR."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


class RiskAnalytics:
    """Portfolio risk analytics and covariance estimation.

    Args:
        returns: Wide DataFrame of asset returns (index=date, columns=symbol).
    """

    def __init__(self, returns: pd.DataFrame) -> None:
        self.returns = returns.dropna(how="all")

    # ----------------------------------------------------------------- covariance

    def sample_cov(self, window: int | None = None) -> pd.DataFrame:
        """Standard sample covariance matrix (annualised)."""
        r = self.returns.tail(window) if window else self.returns
        return r.cov() * 252

    def ledoit_wolf_cov(self, window: int | None = None) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage covariance (better conditioned for small T/N)."""
        r = self.returns.tail(window) if window else self.returns
        lw = LedoitWolf()
        lw.fit(r.dropna())
        return pd.DataFrame(lw.covariance_ * 252, index=r.columns, columns=r.columns)

    def oas_cov(self, window: int | None = None) -> pd.DataFrame:
        """Oracle Approximating Shrinkage (OAS) covariance estimator."""
        r = self.returns.tail(window) if window else self.returns
        oas = OAS()
        oas.fit(r.dropna())
        return pd.DataFrame(oas.covariance_ * 252, index=r.columns, columns=r.columns)

    def ewma_cov(self, span: int = 60) -> pd.DataFrame:
        """Exponentially weighted moving average covariance (annualised)."""
        return self.returns.ewm(span=span).cov().iloc[-len(self.returns.columns):] * 252

    # ----------------------------------------------------------------- portfolio risk

    def portfolio_vol(self, weights: pd.Series, cov: pd.DataFrame | None = None) -> float:
        """Annualised portfolio volatility given weights and covariance matrix."""
        if cov is None:
            cov = self.ledoit_wolf_cov()
        w = weights.reindex(cov.index).fillna(0).values
        sigma = cov.values
        return float(np.sqrt(w @ sigma @ w))

    def marginal_risk_contribution(
        self, weights: pd.Series, cov: pd.DataFrame | None = None
    ) -> pd.Series:
        """Marginal risk contribution (∂σ_p / ∂w_i) for each asset."""
        if cov is None:
            cov = self.ledoit_wolf_cov()
        w = weights.reindex(cov.index).fillna(0)
        sigma = cov
        port_vol = self.portfolio_vol(w, sigma)
        mrc = sigma @ w / port_vol
        return mrc.rename("mrc")

    def risk_contribution(
        self, weights: pd.Series, cov: pd.DataFrame | None = None
    ) -> pd.Series:
        """Total risk contribution: w_i * MRC_i for each asset."""
        mrc = self.marginal_risk_contribution(weights, cov)
        return (weights * mrc).rename("rc")

    # ----------------------------------------------------------------- factor risk

    def factor_var_decomposition(
        self,
        weights: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> pd.Series:
        """Decompose portfolio variance into factor contributions.

        Args:
            weights: Portfolio weights (index=symbol).
            factor_returns: Factor return time series (index=date, columns=factor).

        Returns:
            Series of factor risk contributions (% of total variance).
        """
        # Regress portfolio returns onto factors
        port_ret = (self.returns * weights.reindex(self.returns.columns).fillna(0)).sum(axis=1)
        aligned = pd.concat([port_ret, factor_returns], axis=1).dropna()
        y = aligned.iloc[:, 0].values
        X = aligned.iloc[:, 1:].values
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        factor_cov = np.cov(X.T)
        factor_var = beta @ factor_cov @ beta
        total_var = np.var(y)
        specific_var = total_var - factor_var
        factor_names = factor_returns.columns.tolist()
        contrib = {f: float(beta[i] ** 2 * factor_cov[i, i] / total_var)
                   for i, f in enumerate(factor_names)}
        contrib["specific"] = float(specific_var / total_var)
        return pd.Series(contrib)

    # ----------------------------------------------------------------- tail risk

    def historical_var(self, confidence: float = 0.95, window: int | None = None) -> pd.Series:
        """Historical VaR for each asset."""
        r = self.returns.tail(window) if window else self.returns
        return r.quantile(1 - confidence)

    def historical_cvar(self, confidence: float = 0.95, window: int | None = None) -> pd.Series:
        """Historical CVaR (expected shortfall) for each asset."""
        var = self.historical_var(confidence, window)
        r = self.returns.tail(window) if window else self.returns
        return r[r.le(var)].mean()

    def correlation_matrix(self) -> pd.DataFrame:
        return self.returns.corr()
