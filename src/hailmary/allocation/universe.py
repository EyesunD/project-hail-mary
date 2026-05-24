"""Stashaway asset universe → tradeable-ticker map.

Each entry maps the **identifier as it appears in a Stashaway statement** to a
ticker resolvable by the existing ``hailmary.data`` provider layer, plus
asset-class / region / sector metadata used by the diagnostic engine.

The map starts deliberately small — it is seeded incrementally as new tickers
are surfaced by the parser. Phase 1B (universe-gap analysis) will require
enumerating Stashaway's full offering; that work is out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AssetMetadata:
    """Static metadata for a Stashaway-tradeable asset."""

    ticker: str
    asset_class: str
    region: str
    sector: str | None = None


class UnknownAssetError(KeyError):
    """Raised when a Stashaway asset identifier is not present in the universe map."""

    def __init__(self, stashaway_id: str, source: str | None = None) -> None:
        self.stashaway_id = stashaway_id
        self.source = source
        msg = f"Unknown Stashaway asset {stashaway_id!r}"
        if source:
            msg += f" (source: {source})"
        super().__init__(msg)


def _m(ticker: str, asset_class: str, region: str, sector: str | None = None) -> AssetMetadata:
    return AssetMetadata(ticker=ticker, asset_class=asset_class, region=region, sector=sector)


STASHAWAY_UNIVERSE: dict[str, AssetMetadata] = {
    # ----------------------------------------------------------------- Synthetic cash
    "CASH_USD": _m("CASH_USD", "Cash", "US", "Cash"),
    "CASH_SGD": _m("CASH_SGD", "Cash", "Singapore", "Cash"),
    # ----------------------------------------------------------------- US-listed equity sector SPDRs
    "XLE": _m("XLE", "Equity", "US", "Energy"),
    "XLI": _m("XLI", "Equity", "US", "Industrials"),
    "XLK": _m("XLK", "Equity", "US", "Technology"),
    "XLP": _m("XLP", "Equity", "US", "Consumer Staples"),
    "XLU": _m("XLU", "Equity", "US", "Utilities"),
    "XLV": _m("XLV", "Equity", "US", "Health Care"),
    "XLY": _m("XLY", "Equity", "US", "Consumer Discretionary"),
    "RSP": _m("RSP", "Equity", "US", "Broad Market"),
    "IVV": _m("IVV", "Equity", "US", "Broad Market"),
    "VYM": _m("VYM", "Equity", "US", "Dividend"),
    "PPA": _m("PPA", "Equity", "US", "Aerospace & Defense"),
    # ----------------------------------------------------------------- US-listed regional / thematic
    "VEU": _m("VEU", "Equity", "Developed ex-US", "Broad Market"),
    "SPEM": _m("SPEM", "Equity", "Emerging Markets", "Broad Market"),
    "BBJP": _m("BBJP", "Equity", "Japan", "Broad Market"),
    "DXJ": _m("DXJ", "Equity", "Japan", "Hedged"),
    "FLIN": _m("FLIN", "Equity", "India", "Broad Market"),
    # ----------------------------------------------------------------- US-listed bonds & cash-equivalent
    "BB3M": _m("BB3M", "Bond", "US", "Treasury 0-3M"),
    # ----------------------------------------------------------------- US-listed thematic / income
    # Stashaway uses the UCITS variant for SG investors (15% withholding via
    # Ireland-US treaty vs 30% on the US-listed JEPQ). User confirmed 2026-05.
    "JEPQ": _m("JEPQ.L", "Equity", "US", "Nasdaq Covered Call (proxy: JEPQ.L UCITS)"),
    # ----------------------------------------------------------------- US-listed alts
    "GLDM": _m("GLDM", "Commodity", "Global", "Gold"),
    # Spot crypto prices (BTC-USD / ETH-USD) used as proxies — far longer history
    # than the 2024-launched FBTC / FETH spot-bitcoin/ether ETFs.
    "FBTC": _m("BTC-USD", "Crypto", "Global", "Bitcoin (proxy: BTC-USD spot)"),
    "FETH": _m("ETH-USD", "Crypto", "Global", "Ethereum (proxy: ETH-USD spot)"),
    # ----------------------------------------------------------------- LSE-listed UCITS ETFs (BlackRock sleeve)
    "ISAC": _m("ISAC.L", "Equity", "Global", "Broad Market"),
    "CSUS": _m("CSUS.L", "Equity", "US", "Broad Market"),
    "CSPX": _m("CSPX.L", "Equity", "US", "Broad Market"),
    "SASU": _m("SASU.L", "Equity", "US", "ESG Screened"),
    "IUIS": _m("IUIS.L", "Equity", "US", "Industrials"),
    "IJPA": _m("IJPA.L", "Equity", "Japan", "Broad Market"),
    "IJPD": _m("IJPD.L", "Equity", "Japan", "Hedged"),
    "ISFD": _m("ISFD.L", "Equity", "UK", "Broad Market"),
    "CCAU": _m("CCAU.L", "Equity", "Canada", "Broad Market"),
    # CEUU.L has no Yahoo data; CEU1.L is an alternative share class of the same fund.
    "CEUU": _m("CEU1.L", "Equity", "Eurozone", "Broad Market (proxy: CEU1.L)"),
    "CPXJ": _m("CPXJ.L", "Equity", "Pacific ex-Japan", "Broad Market"),
    # EXCH.L (EM ex-China UCITS) has no Yahoo coverage; SPEM is broad EM but the
    # closest tradeable proxy. Slight composition drift (includes China).
    "EXCH": _m("SPEM", "Equity", "Emerging Markets ex-China", "Broad Market (proxy: SPEM)"),
    # ICHN.L has no Yahoo coverage; MCHI is the US-listed China equivalent.
    "ICHN": _m("MCHI", "Equity", "China", "Broad Market (proxy: MCHI)"),
    "IDTM": _m("IDTM.L", "Bond", "US", "Treasury 7-10Y"),
    "IDTL": _m("IDTL.L", "Bond", "US", "Treasury 20+Y"),
    "IBTU": _m("IBTU.L", "Bond", "US", "Treasury 0-1Y"),
    "TIP5": _m("TIP5.L", "Bond", "US", "TIPS 0-5Y"),
    "IMBS": _m("IMBS.L", "Bond", "US", "MBS"),
    "IEMB": _m("IEMB.L", "Bond", "Emerging Markets", "USD Sovereign"),
    "IGLN": _m("IGLN.L", "Commodity", "Global", "Gold"),
    # ----------------------------------------------------------------- SGX-listed (Singapore Investing / SG ETF)
    "A35": _m("A35.SI", "Bond", "Singapore", "Aggregate"),
    "MBH": _m("MBH.SI", "Bond", "Singapore", "Government"),
    "G3B": _m("G3B.SI", "Equity", "Singapore", "Broad Market"),
    "CLR": _m("CLR.SI", "Equity", "Asia ex-Japan", "Broad Market"),
    "MMS": _m("MMS.SI", "Equity", "Singapore", "Broad Market"),
    "QL3": _m("QL3.SI", "Equity", "Asia ex-Japan", "Broad Market"),
    # ----------------------------------------------------------------- Stashaway-only funds (no public ticker)
    # These are JPMorgan-branded share classes inside Income Investing — proxied with a
    # generic global aggregate-income ETF for diagnostic purposes only. Returns will
    # not match the true holding's NAV but the asset-class shape is preserved.
    "JINAASH": _m("AGG", "Bond", "Global", "Aggregate (proxy: AGG)"),
    "JPEMDSG": _m("EMB", "Bond", "Emerging Markets", "USD Sovereign (proxy: EMB)"),
    "JPGCBAS": _m("AGG", "Bond", "Global", "Aggregate (proxy: AGG)"),
    "JPGHYHS": _m("HYG", "Bond", "Global", "High Yield (proxy: HYG)"),
    "JPMGASA": _m("AGG", "Bond", "Global", "Aggregate (proxy: AGG)"),
    # Lion Global / OCBC SGD money-market funds inside Guitsa / Simple SGD.
    # Now HOLDING-tagged (since 2026-05) so they're visible as cash-replacement
    # candidates; both resolve to the synthetic CASH_SGD series (~1.5% p.a.).
    "LNWELIA": _m("CASH_SGD", "Cash", "Singapore", "Money Market"),
    "OCBSGDM": _m("CASH_SGD", "Cash", "Singapore", "Money Market"),
    # Simple USD's underlying is US 0-3M Treasury bills. BIL (SPDR Bloomberg 1-3
    # Month T-Bill ETF) is a clean Yahoo proxy with real market data — gives
    # Simple USD genuine price variation and a current ~5% yield.
    "BB3M": _m("BIL", "Cash", "US", "Treasury 0-3M (proxy: BIL)"),
}
"""Stashaway identifier → AssetMetadata.

Seeded from the user's 2026-04 statement (55 tickers, 15 portfolios). Some
LSE-listed UCITS ETFs and SGX-listed funds carry exchange suffixes (`.L`,
`.SI`) on Yahoo Finance. Stashaway-only managed funds without public tickers
are mapped to the closest tradeable proxy with the proxy noted in the
``sector`` field.
"""


def resolve(stashaway_id: str, *, source: str | None = None) -> AssetMetadata:
    """Look up *stashaway_id* in :data:`STASHAWAY_UNIVERSE`.

    Raises :class:`UnknownAssetError` (with the source portfolio name if given)
    when the identifier is missing, rather than silently dropping the holding.
    """
    try:
        return STASHAWAY_UNIVERSE[stashaway_id]
    except KeyError as exc:
        raise UnknownAssetError(stashaway_id, source=source) from exc
