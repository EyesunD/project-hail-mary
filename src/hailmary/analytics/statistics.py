"""Factor analytics: IC analysis, quintile spreads, signal decay."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class FactorAnalytics:
    """Factor signal evaluation toolkit.

    Args:
        factor_panel:   (date × symbol) factor score panel.
        returns_panel:  (date × symbol) forward return panel.
        n_quantiles:    Number of quantile buckets.
    """

    def __init__(
        self,
        factor_panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
        n_quantiles: int = 5,
    ) -> None:
        self.factor_panel = factor_panel
        self.returns_panel = returns_panel
        self.n_quantiles = n_quantiles

    # ------------------------------------------------------------------ IC

    def ic_series(self, horizon: int = 1) -> pd.Series:
        """Spearman IC between factor score and *horizon*-period forward return."""
        fwd = self.returns_panel.shift(-horizon)
        ics = {}
        for dt in self.factor_panel.index:
            if dt not in fwd.index:
                continue
            s = self.factor_panel.loc[dt].dropna()
            r = fwd.loc[dt].dropna()
            common = s.index.intersection(r.index)
            if len(common) < 5:
                continue
            ic, _ = stats.spearmanr(s[common], r[common])
            ics[dt] = ic
        return pd.Series(ics, name=f"IC_h{horizon}")

    def ic_stats(self, horizon: int = 1) -> pd.Series:
        ic = self.ic_series(horizon).dropna()
        t, p = stats.ttest_1samp(ic, 0)
        return pd.Series({
            "Mean IC": f"{ic.mean():.4f}",
            "IC Std": f"{ic.std():.4f}",
            "ICIR": f"{ic.mean() / ic.std():.2f}",
            "T-stat": f"{t:.2f}",
            "P-value": f"{p:.4f}",
            "% Positive": f"{(ic > 0).mean():.1%}",
            "Hit Rate (|IC|>0.05)": f"{(ic.abs() > 0.05).mean():.1%}",
        })

    def decay_curve(self, max_horizon: int = 63) -> pd.Series:
        """IC decay: compute mean IC at each horizon from 1 to *max_horizon*."""
        return pd.Series(
            {h: self.ic_series(h).mean() for h in range(1, max_horizon + 1)},
            name="mean_ic",
        )

    # ------------------------------------------------------------------ quantile

    def quantile_returns(self, horizon: int = 21) -> pd.DataFrame:
        """Mean forward return by quantile bucket at each date."""
        fwd = self.returns_panel.shift(-horizon)
        records = []
        for dt in self.factor_panel.index:
            if dt not in fwd.index:
                continue
            scores = self.factor_panel.loc[dt].dropna()
            rets = fwd.loc[dt].dropna()
            common = scores.index.intersection(rets.index)
            if len(common) < self.n_quantiles * 2:
                continue
            q = pd.qcut(scores[common], self.n_quantiles, labels=False) + 1
            row = rets[common].groupby(q).mean().to_dict()
            row["date"] = dt
            records.append(row)
        df = pd.DataFrame(records).set_index("date")
        df.columns = [f"Q{int(c)}" for c in df.columns]
        return df

    def quantile_spread(self, horizon: int = 21) -> pd.Series:
        """Q5 minus Q1 return spread (the signal's long-short return)."""
        qr = self.quantile_returns(horizon)
        if "Q5" not in qr or "Q1" not in qr:
            return pd.Series(dtype=float)
        return (qr["Q5"] - qr["Q1"]).rename("Q5-Q1 spread")

    def cumulative_quantile_returns(self, horizon: int = 21) -> pd.DataFrame:
        """Cumulative returns for each quantile bucket."""
        qr = self.quantile_returns(horizon)
        return (1 + qr).cumprod()

    # ------------------------------------------------------------------ turnover

    def factor_turnover(self) -> pd.Series:
        """Cross-sectional rank correlation of scores between periods (1 - turnover)."""
        correlations = []
        dates = self.factor_panel.index
        for i in range(1, len(dates)):
            prev = self.factor_panel.iloc[i - 1].dropna()
            curr = self.factor_panel.iloc[i].dropna()
            common = prev.index.intersection(curr.index)
            if len(common) < 5:
                continue
            rho, _ = stats.spearmanr(prev[common], curr[common])
            correlations.append(rho)
        return pd.Series(correlations, index=dates[1:len(correlations) + 1], name="rank_autocorr")

    # ------------------------------------------------------------------ returns attribution

    def factor_quintile_cumreturn(self) -> dict[str, float]:
        """Total return for each quintile portfolio over the full sample."""
        qr = self.quantile_returns()
        return {col: float((1 + qr[col].dropna()).prod() - 1) for col in qr.columns}
