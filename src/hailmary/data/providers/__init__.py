"""Concrete market-data providers."""

from hailmary.data.providers.csv_provider import CSVProvider
from hailmary.data.providers.yahoo import YahooFinanceProvider

__all__ = ["CSVProvider", "YahooFinanceProvider"]

# Optional providers — imported lazily so missing extras don't hard-fail
def get_alpaca_provider():  # type: ignore[return]
    """Return AlpacaProvider class (requires `pip install hailmary[alpaca]`)."""
    from hailmary.data.providers.alpaca import AlpacaProvider  # noqa: PLC0415
    return AlpacaProvider


def get_polygon_provider():  # type: ignore[return]
    """Return PolygonProvider class (requires `pip install hailmary[polygon]`)."""
    from hailmary.data.providers.polygon import PolygonProvider  # noqa: PLC0415
    return PolygonProvider
