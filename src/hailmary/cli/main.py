"""hailmary CLI — run backtests and inspect data from the terminal."""

from __future__ import annotations

import click
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """Project Hail Mary — quant backtesting & analytics platform."""


@main.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option("--start", "-s", default="2020-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--end", "-e", default=None, help="End date (YYYY-MM-DD, default=today)")
@click.option("--provider", "-p", default="yahoo", help="Data provider name")
@click.option("--cache/--no-cache", default=True, help="Use local cache")
def fetch(symbols: tuple[str, ...], start: str, end: str | None, provider: str, cache: bool) -> None:
    """Fetch and display OHLCV data for SYMBOLS."""
    from hailmary.data.providers import YahooFinanceProvider  # noqa: PLC0415
    from hailmary.data.cache import DataCache  # noqa: PLC0415

    dc = DataCache() if cache else None
    prov = YahooFinanceProvider(cache=dc)

    with console.status(f"Fetching {', '.join(symbols)} from {provider}..."):
        df = prov.get_bars(list(symbols), start=start, end=end or "2099-01-01")

    table = Table(title=f"OHLCV — {', '.join(symbols)}")
    for col in ["symbol", "timestamp", "open", "high", "low", "close", "volume"]:
        table.add_column(col, style="cyan" if col == "symbol" else "white")

    for (sym, ts), row in df.iterrows():
        table.add_row(sym, str(ts.date()), *[f"{row[c]:.2f}" if c != "volume" else f"{int(row[c]):,}"
                                              for c in ["open","high","low","close","volume"]])
    console.print(table)


@main.command()
@click.option("--cache-dir", default="data/cache", help="Cache directory to clear")
def clear_cache(cache_dir: str) -> None:
    """Clear the local data cache."""
    from hailmary.data.cache import DataCache  # noqa: PLC0415
    dc = DataCache(cache_dir)
    dc.clear()
    console.print(f"[green]Cache cleared:[/green] {cache_dir}")


@main.command()
def info() -> None:
    """Show platform version and registered providers."""
    from hailmary import __version__  # noqa: PLC0415
    console.print(f"[bold blue]Project Hail Mary[/bold blue] v{__version__}")
    console.print("Registered providers: Yahoo Finance, CSV, Alpaca (optional), Polygon (optional)")
