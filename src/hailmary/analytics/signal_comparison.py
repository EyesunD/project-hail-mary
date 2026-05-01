"""Side-by-side signal-quality comparison across variants."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hailmary.analytics.signal_analytics import SignalAnalytics
from hailmary.backtest.signal_backtest import BarBacktestResult
from hailmary.viz.theme import PALETTE, apply_theme

FillMethod = Literal["net", "mtc", "conservative"]

_RET_COL: dict[str, str] = {
    "net": "return_net",
    "mtc": "return_mtc",
    "conservative": "return_conservative",
}


def _pool_agg(trades: pd.DataFrame, method: FillMethod) -> pd.Series:
    ret = trades[_RET_COL[method]]
    dd = trades[f"max_intra_drawdown_{method}"]
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    n = len(ret)
    win_rate = len(wins) / n if n > 0 else float("nan")
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    loss_sum = float(losses.sum())
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    ret_ex_top = ret.drop(ret.idxmax()) if len(ret) > 1 else ret
    return pd.Series({
        "n_trades": n,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "expectancy_ex_top": float(ret_ex_top.mean()),
        "median_return": float(ret.median()),
        "profit_factor": float(wins.sum()) / abs(loss_sum) if loss_sum != 0 else float("inf"),
        "max_win": float(wins.max()) if len(wins) > 0 else 0.0,
        "max_loss": float(losses.min()) if len(losses) > 0 else 0.0,
        "skewness": float(ret.skew()),
        "avg_duration": float(trades["duration"].mean()),
        "avg_intra_drawdown": float(dd.mean()),
        "max_intra_drawdown": float(dd.min()),
    })


class SignalComparison:
    """Side-by-side quality comparison of multiple signal variants.

    Args:
        variants: Mapping of variant label → :class:`~hailmary.backtest.BarBacktestResult`.

    Example::

        from hailmary.analytics.signal_comparison import SignalComparison

        variants = {
            "MA-200 Base":        bt_base,
            "MA-200 + Regime":    bt_regime,
        }
        SignalComparison(variants).tearsheet().show()
    """

    _COLOURS = [
        PALETTE["accent_blue"],
        PALETTE["accent_orange"],
        PALETTE["accent_purple"],
        PALETTE["accent_green"],
        PALETTE["accent_yellow"],
    ]

    def __init__(self, variants: dict[str, BarBacktestResult]) -> None:
        self._analytics: dict[str, SignalAnalytics] = {
            name: SignalAnalytics(result) for name, result in variants.items()
        }

    # ----------------------------------------------------------------- tables

    def pooled_table(self, method: FillMethod = "net") -> pd.DataFrame:
        """One row per variant, trades pooled across all symbols.

        Args:
            method: ``"net"`` (default), ``"mtc"``, or ``"conservative"``.

        Returns:
            DataFrame indexed by variant name with trade-quality columns.
        """
        return pd.DataFrame(
            {name: _pool_agg(sa.trade_stats(), method) for name, sa in self._analytics.items()}
        ).T

    def symbol_table(self, method: FillMethod = "net") -> pd.DataFrame:
        """MultiIndex ``(variant, symbol)`` rows — per-asset breakdown.

        Args:
            method: ``"net"`` (default), ``"mtc"``, or ``"conservative"``.

        Returns:
            DataFrame with two-level index ``(variant, symbol)``.
        """
        return pd.concat(
            {name: sa.trade_summary(method=method) for name, sa in self._analytics.items()},
            names=["variant"],
        )

    # ----------------------------------------------------------------- tearsheet

    def tearsheet(self) -> go.Figure:
        """Three-panel comparison tearsheet.

        Panels (top to bottom):

        1. **Pooled quality table** — trades pooled across all symbols; net columns
           show WR, Avg Win/Loss, Expectancy, Exp ex-Top, Median, Profit Factor,
           Skewness, and Avg/Worst intra-trade DD; conservative columns show the
           same edge metrics for fill-risk comparison.
        2. **Per-symbol expectancy heatmap** — net expectancy per (variant × symbol);
           red = negative edge, green = positive edge.  Spot where a change helps
           one asset but hurts another.
        3. **Return distribution** — box + individual-trade scatter per symbol,
           one box per variant (net returns).

        Returns:
            :class:`plotly.graph_objects.Figure`
        """
        symbols = self._symbols()
        variant_names = list(self._analytics.keys())

        fig = make_subplots(
            rows=3, cols=1,
            specs=[[{"type": "table"}], [{"type": "xy"}], [{"type": "xy"}]],
            row_heights=[0.27, 0.27, 0.46],
            vertical_spacing=0.04,
        )

        self._add_pooled_table(fig, variant_names, row=1)
        self._add_heatmap(fig, symbols, variant_names, row=2)
        self._add_distribution(fig, row=3)

        # Table (row 1) has no xy axes; heatmap is xaxis/yaxis, distribution is xaxis2/yaxis2.
        fig.update_layout(
            boxmode="group",
            xaxis=dict(title_text="Symbol"),
            yaxis=dict(title_text="Variant"),
            xaxis2=dict(title_text="Symbol"),
            yaxis2=dict(tickformat=".0%", title_text="Return per bet (net)"),
        )
        fig.add_shape(
            type="line",
            x0=0, x1=1, xref="x2 domain",
            y0=0, y1=0, yref="y2",
            line=dict(color=PALETTE["text_secondary"], width=0.8, dash="dash"),
        )

        apply_theme(fig, title="Signal Quality Comparison", height=1100)
        return fig

    # ----------------------------------------------------------------- private

    def _symbols(self) -> list[str]:
        syms: set[str] = set()
        for sa in self._analytics.values():
            syms |= set(sa.trade_stats()["symbol"].unique())
        return sorted(syms)

    def _add_pooled_table(
        self, fig: go.Figure, variant_names: list[str], row: int
    ) -> None:
        pool_net = self.pooled_table("net")
        pool_con = self.pooled_table("conservative")

        def _pf(v: str, tbl: pd.DataFrame) -> str:
            val = tbl.loc[v, "profit_factor"]
            return "∞" if val == float("inf") else f"{val:.2f}"

        headers = [
            "Variant", "N",
            # Net
            "WR (N)", "Avg Win", "Avg Loss", "Exp (N)", "Exp ex-Top (N)", "Median (N)",
            "PF (N)", "Skew (N)", "Avg DD (N)", "Wst DD (N)",
            # Conservative
            "WR (C)", "Exp (C)", "Exp ex-Top (C)", "Median (C)", "PF (C)",
            # Shared
            "Dur",
        ]
        cells = [
            variant_names,
            [f"{int(pool_net.loc[v, 'n_trades'])}" for v in variant_names],
            # Net
            [f"{pool_net.loc[v, 'win_rate']:.1%}" for v in variant_names],
            [f"{pool_net.loc[v, 'avg_win']:+.1%}" for v in variant_names],
            [f"{pool_net.loc[v, 'avg_loss']:+.1%}" for v in variant_names],
            [f"{pool_net.loc[v, 'expectancy']:+.1%}" for v in variant_names],
            [f"{pool_net.loc[v, 'expectancy_ex_top']:+.1%}" for v in variant_names],
            [f"{pool_net.loc[v, 'median_return']:+.1%}" for v in variant_names],
            [_pf(v, pool_net) for v in variant_names],
            [f"{pool_net.loc[v, 'skewness']:+.2f}" for v in variant_names],
            [f"{pool_net.loc[v, 'avg_intra_drawdown']:.1%}" for v in variant_names],
            [f"{pool_net.loc[v, 'max_intra_drawdown']:.1%}" for v in variant_names],
            # Conservative
            [f"{pool_con.loc[v, 'win_rate']:.1%}" for v in variant_names],
            [f"{pool_con.loc[v, 'expectancy']:+.1%}" for v in variant_names],
            [f"{pool_con.loc[v, 'expectancy_ex_top']:+.1%}" for v in variant_names],
            [f"{pool_con.loc[v, 'median_return']:+.1%}" for v in variant_names],
            [_pf(v, pool_con) for v in variant_names],
            # Shared
            [f"{pool_net.loc[v, 'avg_duration']:.0f}" for v in variant_names],
        ]

        fig.add_trace(go.Table(
            header=dict(
                values=headers,
                fill_color=PALETTE["surface"],
                font=dict(color=PALETTE["text_primary"], size=11),
                align="center",
                line_color=PALETTE["border"],
            ),
            cells=dict(
                values=cells,
                fill_color=PALETTE["background"],
                font=dict(color=PALETTE["text_primary"], size=11),
                align=["left"] + ["center"] * (len(headers) - 1),
                line_color=PALETTE["border"],
            ),
        ), row=row, col=1)

    def _add_heatmap(
        self,
        fig: go.Figure,
        symbols: list[str],
        variant_names: list[str],
        row: int,
    ) -> None:
        sym_net = self.symbol_table("net")

        z = [
            [
                float(sym_net.loc[(v, s), "expectancy"])
                if (v, s) in sym_net.index
                else float("nan")
                for s in symbols
            ]
            for v in variant_names
        ]
        text = [
            [f"{val:+.1%}" if val == val else "n/a" for val in row_vals]
            for row_vals in z
        ]
        max_abs = max(
            (abs(val) for row_vals in z for val in row_vals if val == val),
            default=0.01,
        )

        fig.add_trace(go.Heatmap(
            x=symbols,
            y=variant_names,
            z=z,
            text=text,
            texttemplate="%{text}",
            colorscale=[
                [0.0, PALETTE["accent_red"]],
                [0.5, PALETTE["surface"]],
                [1.0, PALETTE["accent_green"]],
            ],
            zmid=0,
            zmin=-max_abs,
            zmax=max_abs,
            showscale=False,
            textfont=dict(color=PALETTE["text_primary"], size=12),
        ), row=row, col=1)

    def _add_distribution(self, fig: go.Figure, row: int) -> None:
        for v_idx, (name, sa) in enumerate(self._analytics.items()):
            colour = self._COLOURS[v_idx % len(self._COLOURS)]
            trades = sa.trade_stats()
            fig.add_trace(go.Box(
                x=trades["symbol"],
                y=trades["return_net"],
                name=name,
                legendgroup=name,
                marker_color=colour,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                line_width=1.5,
                marker_size=8,
                boxmean=True,
            ), row=row, col=1)
