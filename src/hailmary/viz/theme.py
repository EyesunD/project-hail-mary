"""Shared Plotly theme and colour palette for all Project Hail Mary charts."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Colour palette — dark, high-contrast, quant-appropriate
# ---------------------------------------------------------------------------

PALETTE = {
    "background": "#0d1117",
    "surface": "#161b22",
    "border": "#30363d",
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "accent_blue": "#58a6ff",
    "accent_green": "#3fb950",
    "accent_red": "#f85149",
    "accent_yellow": "#d29922",
    "accent_purple": "#a371f7",
    "accent_orange": "#ffa657",
}

QUANTILE_COLOURS = [
    PALETTE["accent_red"],
    PALETTE["accent_orange"],
    PALETTE["accent_yellow"],
    PALETTE["accent_blue"],
    PALETTE["accent_green"],
]

THEME: dict[str, object] = {
    "template": "plotly_dark",
    "paper_bgcolor": PALETTE["background"],
    "plot_bgcolor": PALETTE["surface"],
    "font_color": PALETTE["text_primary"],
    "colorway": list(PALETTE.values())[4:],
}


def apply_theme(fig: go.Figure, title: str = "", height: int = 500) -> go.Figure:
    """Apply the Project Hail Mary house theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["background"],
        plot_bgcolor=PALETTE["surface"],
        font={"color": PALETTE["text_primary"], "family": "Inter, system-ui, sans-serif"},
        title={
            "text": title,
            "font": {"size": 18, "color": PALETTE["text_primary"]},
            "x": 0.02,
            "xanchor": "left",
        },
        height=height,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        legend={
            "bgcolor": PALETTE["surface"],
            "bordercolor": PALETTE["border"],
            "borderwidth": 1,
        },
        xaxis={
            "gridcolor": PALETTE["border"],
            "zerolinecolor": PALETTE["border"],
            "showspikes": True,
            "spikecolor": PALETTE["text_secondary"],
            "spikethickness": 1,
        },
        yaxis={
            "gridcolor": PALETTE["border"],
            "zerolinecolor": PALETTE["border"],
            "showspikes": True,
            "spikecolor": PALETTE["text_secondary"],
            "spikethickness": 1,
        },
        hovermode="x unified",
    )
    return fig


# Register as named Plotly template
pio.templates["hailmary"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=PALETTE["background"],
        plot_bgcolor=PALETTE["surface"],
        font={"color": PALETTE["text_primary"]},
        colorway=QUANTILE_COLOURS,
    )
)
