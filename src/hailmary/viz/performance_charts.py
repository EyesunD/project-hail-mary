"""Performance visualisations: equity curve, drawdowns, rolling metrics, tearsheet."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hailmary.viz.theme import PALETTE, apply_theme

if TYPE_CHECKING:
    from hailmary.backtest.engine import BacktestResult


class PerformanceCharts:
    """Generate interactive Plotly charts from a BacktestResult.

    All methods return a ``go.Figure`` that can be shown with ``.show()`` or
    exported with ``.write_html()`` / ``.write_image()``.
    """

    def __init__(self, result: "BacktestResult", name: str = "Strategy") -> None:
        self.result = result
        self.name = name

    # ----------------------------------------------------------------- curves

    def equity_curve(self, normalise: bool = True) -> go.Figure:
        """NAV growth curve vs optional benchmark."""
        fig = go.Figure()
        nav = self.result.nav
        if normalise:
            nav = nav / nav.iloc[0] * 100

        fig.add_trace(go.Scatter(
            x=nav.index, y=nav.values, name=self.name, mode="lines",
            line={"color": PALETTE["accent_blue"], "width": 2},
            fill="tozeroy", fillcolor=f"rgba(88,166,255,0.08)",
        ))

        if self.result.benchmark_nav is not None:
            bm = self.result.benchmark_nav
            if normalise:
                bm = bm / bm.iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=bm.index, y=bm.values, name="Benchmark", mode="lines",
                line={"color": PALETTE["text_secondary"], "width": 1.5, "dash": "dot"},
            ))

        apply_theme(fig, f"{self.name} — Equity Curve", height=450)
        fig.update_yaxes(title_text="Normalised Value (base=100)" if normalise else "NAV ($)")
        return fig

    def drawdown_chart(self) -> go.Figure:
        """Underwater equity chart."""
        cum = (1 + self.result.returns).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax() * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name="Drawdown", mode="lines",
            line={"color": PALETTE["accent_red"], "width": 1.5},
            fill="tozeroy", fillcolor=f"rgba(248,81,73,0.2)",
        ))

        apply_theme(fig, f"{self.name} — Drawdown", height=300)
        fig.update_yaxes(title_text="Drawdown (%)")
        return fig

    def rolling_sharpe(self, window: int = 252) -> go.Figure:
        """Rolling Sharpe ratio."""
        returns = self.result.returns
        rolling = (returns.rolling(window).mean() * 252 - 0.0) / (returns.rolling(window).std() * np.sqrt(252))

        fig = go.Figure()
        fig.add_hline(y=0, line_color=PALETTE["border"])
        fig.add_hline(y=1, line_color=PALETTE["accent_green"], line_dash="dot", annotation_text="Sharpe=1")
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values, name=f"{window}d Rolling Sharpe",
            line={"color": PALETTE["accent_purple"], "width": 2},
        ))

        apply_theme(fig, f"{self.name} — Rolling {window}d Sharpe", height=300)
        return fig

    def monthly_returns_heatmap(self) -> go.Figure:
        """Calendar heatmap of monthly returns."""
        monthly = self.result.returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)
        df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        }).pivot(index="year", columns="month", values="ret")
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        df.columns = [month_labels[int(m) - 1] for m in df.columns]

        fig = go.Figure(go.Heatmap(
            z=df.values * 100,
            x=df.columns.tolist(),
            y=df.index.tolist(),
            colorscale=[[0, PALETTE["accent_red"]], [0.5, PALETTE["surface"]], [1, PALETTE["accent_green"]]],
            zmid=0,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in df.values * 100],
            texttemplate="%{text}",
            showscale=True,
            colorbar={"title": "Return %"},
        ))

        apply_theme(fig, f"{self.name} — Monthly Returns (%)", height=max(300, 60 * len(df)))
        return fig

    def returns_distribution(self) -> go.Figure:
        """Histogram of daily returns with normal overlay."""
        r = self.result.returns.dropna()
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=r.values * 100, name="Daily Returns",
            nbinsx=80, histnorm="probability density",
            marker_color=PALETTE["accent_blue"], opacity=0.7,
        ))
        # Normal overlay
        x_range = np.linspace(r.min() * 100, r.max() * 100, 300)
        from scipy.stats import norm
        mu, sigma = r.mean() * 100, r.std() * 100
        normal_y = norm.pdf(x_range, mu, sigma)
        fig.add_trace(go.Scatter(
            x=x_range, y=normal_y, name="Normal fit",
            line={"color": PALETTE["accent_red"], "dash": "dot", "width": 2},
        ))

        apply_theme(fig, f"{self.name} — Return Distribution", height=400)
        fig.update_xaxes(title_text="Daily Return (%)")
        fig.update_yaxes(title_text="Density")
        return fig

    # ----------------------------------------------------------------- tearsheet

    def tearsheet(self) -> go.Figure:
        """Full 4-panel performance tearsheet."""
        fig = make_subplots(
            rows=4, cols=1,
            row_heights=[0.35, 0.2, 0.2, 0.25],
            subplot_titles=[
                "Equity Curve", "Drawdown", "Rolling 252d Sharpe", "Return Distribution"
            ],
            vertical_spacing=0.06,
            shared_xaxes=True,
        )

        nav = self.result.nav / self.result.nav.iloc[0] * 100
        cum = (1 + self.result.returns).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax() * 100
        returns = self.result.returns
        roll_sharpe = (returns.rolling(252).mean() * 252) / (returns.rolling(252).std() * np.sqrt(252))

        # Row 1: equity
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name=self.name,
                                  line={"color": PALETTE["accent_blue"], "width": 2}), row=1, col=1)
        if self.result.benchmark_nav is not None:
            bm = self.result.benchmark_nav / self.result.benchmark_nav.iloc[0] * 100
            fig.add_trace(go.Scatter(x=bm.index, y=bm, name="Benchmark",
                                      line={"color": PALETTE["text_secondary"], "dash": "dot"}), row=1, col=1)

        # Row 2: drawdown
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown", fill="tozeroy",
                                  line={"color": PALETTE["accent_red"]},
                                  fillcolor="rgba(248,81,73,0.2)"), row=2, col=1)

        # Row 3: rolling Sharpe
        fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe,
                                  name="Rolling Sharpe",
                                  line={"color": PALETTE["accent_purple"]}), row=3, col=1)
        fig.add_hline(y=0, line_color=PALETTE["border"], row=3, col=1)

        # Row 4: histogram
        fig.add_trace(go.Histogram(x=returns.values * 100, nbinsx=60, histnorm="probability density",
                                    name="Daily Returns", marker_color=PALETTE["accent_blue"],
                                    opacity=0.7), row=4, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=PALETTE["background"],
            plot_bgcolor=PALETTE["surface"],
            font={"color": PALETTE["text_primary"]},
            height=900,
            showlegend=True,
            title={"text": f"<b>{self.name}</b> — Performance Tearsheet", "x": 0.02},
            margin={"l": 60, "r": 20, "t": 80, "b": 40},
        )
        for axis in fig.layout:
            if axis.startswith("xaxis") or axis.startswith("yaxis"):
                fig.layout[axis].update(gridcolor=PALETTE["border"])

        return fig

    def weights_chart(self) -> go.Figure:
        """Stacked area chart of portfolio weights over time."""
        w = self.result.weights
        if w.empty:
            return go.Figure()
        top_n = w.mean().nlargest(15).index
        w = w[top_n].fillna(0)

        fig = go.Figure()
        colors = [PALETTE["accent_blue"], PALETTE["accent_green"], PALETTE["accent_purple"],
                  PALETTE["accent_orange"], PALETTE["accent_yellow"]] * 3
        for i, sym in enumerate(w.columns):
            fig.add_trace(go.Scatter(
                x=w.index, y=w[sym], name=sym, stackgroup="one",
                mode="lines", line={"width": 0.5},
                fillcolor=colors[i % len(colors)],
            ))

        apply_theme(fig, f"{self.name} — Portfolio Weights", height=400)
        fig.update_yaxes(title_text="Weight")
        return fig
