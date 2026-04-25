"""Factor analysis visualisations: IC, quintile returns, decay, exposures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hailmary.viz.theme import PALETTE, QUANTILE_COLOURS, apply_theme

if TYPE_CHECKING:
    from hailmary.analytics.statistics import FactorAnalytics


class FactorCharts:
    """Interactive factor analysis charts built on :class:`~hailmary.analytics.statistics.FactorAnalytics`.

    Args:
        analytics: Pre-constructed FactorAnalytics instance.
        factor_name: Display name shown in chart titles.
    """

    def __init__(self, analytics: "FactorAnalytics", factor_name: str = "Factor") -> None:
        self.analytics = analytics
        self.factor_name = factor_name

    # ------------------------------------------------------------------ IC

    def ic_series_chart(self, horizon: int = 21) -> go.Figure:
        """Bar chart of IC over time with cumulative IC overlay."""
        ic = self.analytics.ic_series(horizon)
        cum_ic = ic.cumsum()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        colors = [PALETTE["accent_green"] if v >= 0 else PALETTE["accent_red"] for v in ic.values]
        fig.add_trace(go.Bar(x=ic.index, y=ic.values, name="IC", marker_color=colors, opacity=0.7))
        fig.add_trace(go.Scatter(
            x=cum_ic.index, y=cum_ic.values, name="Cumulative IC",
            line={"color": PALETTE["accent_blue"], "width": 2},
        ), secondary_y=True)
        fig.add_hline(y=0, line_color=PALETTE["border"])
        fig.add_hline(y=ic.mean(), line_color=PALETTE["accent_yellow"],
                      line_dash="dot", annotation_text=f"Mean IC={ic.mean():.3f}")

        apply_theme(fig, f"{self.factor_name} — IC Series (h={horizon}d)", height=400)
        return fig

    def ic_distribution(self, horizon: int = 21) -> go.Figure:
        """Histogram of IC values."""
        ic = self.analytics.ic_series(horizon).dropna()
        fig = go.Figure()
        fig.add_vline(x=0, line_color=PALETTE["border"])
        fig.add_vline(x=ic.mean(), line_color=PALETTE["accent_yellow"],
                      line_dash="dot", annotation_text=f"μ={ic.mean():.3f}")
        fig.add_trace(go.Histogram(
            x=ic.values, nbinsx=40, name="IC",
            marker_color=PALETTE["accent_blue"], opacity=0.8, histnorm="probability density",
        ))
        apply_theme(fig, f"{self.factor_name} — IC Distribution", height=350)
        fig.update_xaxes(title_text="Information Coefficient")
        return fig

    def ic_decay_chart(self, max_horizon: int = 63) -> go.Figure:
        """IC decay curve from horizon=1 to *max_horizon*."""
        decay = self.analytics.decay_curve(max_horizon)
        fig = go.Figure()
        fig.add_hline(y=0, line_color=PALETTE["border"])
        fig.add_trace(go.Scatter(
            x=decay.index, y=decay.values, name="Mean IC",
            mode="lines+markers",
            line={"color": PALETTE["accent_blue"], "width": 2},
            marker={"size": 4},
            fill="tozeroy", fillcolor="rgba(88,166,255,0.12)",
        ))
        apply_theme(fig, f"{self.factor_name} — IC Decay Curve", height=350)
        fig.update_xaxes(title_text="Horizon (trading days)")
        fig.update_yaxes(title_text="Mean Spearman IC")
        return fig

    # ------------------------------------------------------------------ quantile

    def quantile_returns_chart(self, horizon: int = 21) -> go.Figure:
        """Mean return by quantile bar chart."""
        qr = self.analytics.quantile_returns(horizon)
        means = qr.mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=means.index.tolist(),
            y=means.values * 100,
            name="Mean Return",
            marker_color=QUANTILE_COLOURS[:len(means)],
            text=[f"{v:.2f}%" for v in means.values * 100],
            textposition="outside",
        ))
        apply_theme(fig, f"{self.factor_name} — Mean Return by Quintile (h={horizon}d)", height=400)
        fig.update_yaxes(title_text="Mean Forward Return (%)")
        return fig

    def cumulative_quantile_chart(self, horizon: int = 21) -> go.Figure:
        """Cumulative return lines for each quintile bucket."""
        cum = self.analytics.cumulative_quantile_returns(horizon)
        fig = go.Figure()
        for i, col in enumerate(cum.columns):
            fig.add_trace(go.Scatter(
                x=cum.index, y=(cum[col] - 1) * 100,
                name=col, mode="lines",
                line={"color": QUANTILE_COLOURS[i % len(QUANTILE_COLOURS)], "width": 2},
            ))
        apply_theme(fig, f"{self.factor_name} — Cumulative Returns by Quintile", height=450)
        fig.update_yaxes(title_text="Cumulative Return (%)")
        return fig

    def spread_chart(self, horizon: int = 21) -> go.Figure:
        """Long-short (Q5 − Q1) spread over time."""
        spread = self.analytics.quantile_spread(horizon)
        cum_spread = (1 + spread).cumprod()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        colors = [PALETTE["accent_green"] if v >= 0 else PALETTE["accent_red"] for v in spread.values]
        fig.add_trace(go.Bar(x=spread.index, y=spread.values * 100,
                              name="Period Spread", marker_color=colors, opacity=0.6))
        fig.add_trace(go.Scatter(
            x=cum_spread.index, y=cum_spread.values, name="Cumulative Spread",
            line={"color": PALETTE["accent_blue"], "width": 2},
        ), secondary_y=True)
        apply_theme(fig, f"{self.factor_name} — Q5−Q1 Long-Short Spread", height=400)
        return fig

    # ------------------------------------------------------------------ turnover

    def turnover_chart(self) -> go.Figure:
        """Factor rank autocorrelation (proxy for turnover) over time."""
        autocorr = self.analytics.factor_turnover()
        fig = go.Figure()
        fig.add_hline(y=autocorr.mean(), line_color=PALETTE["accent_yellow"],
                      line_dash="dot", annotation_text=f"Mean={autocorr.mean():.2f}")
        fig.add_trace(go.Scatter(
            x=autocorr.index, y=autocorr.values, name="Rank Autocorrelation",
            mode="lines", line={"color": PALETTE["accent_purple"], "width": 1.5},
        ))
        apply_theme(fig, f"{self.factor_name} — Rank Autocorrelation (Signal Stability)", height=300)
        fig.update_yaxes(title_text="Spearman ρ (lag 1)")
        return fig

    # ------------------------------------------------------------------ full tearsheet

    def tearsheet(self, horizon: int = 21) -> go.Figure:
        """4-panel factor analysis tearsheet."""
        ic = self.analytics.ic_series(horizon)
        cum_ic = ic.cumsum()
        qr = self.analytics.quantile_returns(horizon)
        cum_qr = self.analytics.cumulative_quantile_returns(horizon)
        decay = self.analytics.decay_curve(min(horizon * 3, 63))

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "IC Series & Cumulative IC",
                "IC Decay Curve",
                "Mean Return by Quintile",
                "Cumulative Quintile Returns",
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # IC series
        colors = [PALETTE["accent_green"] if v >= 0 else PALETTE["accent_red"] for v in ic.values]
        fig.add_trace(go.Bar(x=ic.index, y=ic.values, marker_color=colors, opacity=0.6, name="IC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=cum_ic.index, y=cum_ic.values, name="Cum IC",
                                  line={"color": PALETTE["accent_blue"]}), row=1, col=1)

        # IC decay
        fig.add_trace(go.Scatter(x=decay.index, y=decay.values, name="IC Decay",
                                  fill="tozeroy", fillcolor="rgba(88,166,255,0.12)",
                                  line={"color": PALETTE["accent_blue"]}), row=1, col=2)

        # Quintile bars
        means = qr.mean()
        fig.add_trace(go.Bar(x=means.index.tolist(), y=means.values * 100,
                              marker_color=QUANTILE_COLOURS[:len(means)], name="Mean Return"), row=2, col=1)

        # Cumulative quintile lines
        for i, col in enumerate(cum_qr.columns):
            fig.add_trace(go.Scatter(
                x=cum_qr.index, y=(cum_qr[col] - 1) * 100,
                name=col, line={"color": QUANTILE_COLOURS[i % len(QUANTILE_COLOURS)]},
            ), row=2, col=2)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=PALETTE["background"],
            plot_bgcolor=PALETTE["surface"],
            font={"color": PALETTE["text_primary"]},
            height=750,
            title={"text": f"<b>{self.factor_name}</b> — Factor Tearsheet", "x": 0.02},
            margin={"l": 60, "r": 20, "t": 80, "b": 40},
            showlegend=False,
        )
        for axis in fig.layout:
            if axis.startswith("xaxis") or axis.startswith("yaxis"):
                fig.layout[axis].update(gridcolor=PALETTE["border"])

        return fig
