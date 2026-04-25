"""Provider registry — resolve and compose multiple data sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from hailmary.data.base import DataProvider, MarketDataError

if TYPE_CHECKING:
    pass


class ProviderRegistry:
    """Ordered collection of DataProviders with fallback semantics.

    Register providers in priority order; the registry will cascade through
    them when a request fails, giving you automatic failover.

        registry = ProviderRegistry()
        registry.register(YahooFinanceProvider(), priority=10)
        registry.register(AlpacaProvider(...), priority=20)   # preferred
    """

    def __init__(self) -> None:
        self._providers: list[tuple[int, DataProvider]] = []

    # ------------------------------------------------------------------ setup

    def register(self, provider: DataProvider, *, priority: int = 0) -> None:
        """Register *provider*. Higher priority = tried first."""
        self._providers.append((priority, provider))
        self._providers.sort(key=lambda t: t[0], reverse=True)
        logger.info("Registered provider '{}' with priority {}", provider.name, priority)

    def unregister(self, name: str) -> None:
        self._providers = [(p, prov) for p, prov in self._providers if prov.name != name]

    # ------------------------------------------------------------------ access

    @property
    def providers(self) -> list[DataProvider]:
        return [prov for _, prov in self._providers]

    def get(self, name: str) -> DataProvider:
        for prov in self.providers:
            if prov.name == name:
                return prov
        raise KeyError(f"No provider named '{name}' is registered.")

    def primary(self) -> DataProvider:
        if not self._providers:
            raise MarketDataError("No providers registered.")
        return self._providers[0][1]

    # -------------------------------------------------------------- delegation

    def get_bars(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Try providers in priority order, falling back on error."""
        last_exc: Exception | None = None
        for prov in self.providers:
            try:
                return prov.get_bars(*args, **kwargs)
            except Exception as exc:
                logger.warning("Provider '{}' failed: {}. Trying next.", prov.name, exc)
                last_exc = exc
        raise MarketDataError("All providers failed.") from last_exc

    def get_fundamentals(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        for prov in self.providers:
            try:
                return prov.get_fundamentals(*args, **kwargs)
            except NotImplementedError:
                continue
            except Exception as exc:
                logger.warning("Provider '{}' fundamentals failed: {}", prov.name, exc)
        raise MarketDataError("No provider could supply fundamentals.")

    def __repr__(self) -> str:
        names = [p.name for p in self.providers]
        return f"<ProviderRegistry providers={names}>"
