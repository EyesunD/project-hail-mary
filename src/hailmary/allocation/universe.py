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
    # iShares Core MSCI EMU UCITS ETF EUR (Acc), USD-quoted Amsterdam listing
    # (ISIN IE00BKBF6616). Same fund as CEU1.L/IEMU.L but Amsterdam is the
    # USD-quoted primary on Yahoo — matches the EXCH.AS convention.
    "CEUU": _m("CEUU.AS", "Equity", "Eurozone", "Broad Market"),
    "CPXJ": _m("CPXJ.L", "Equity", "Pacific ex-Japan", "Broad Market"),
    # iShares MSCI EM ex-China UCITS ETF USD Acc — Amsterdam listing is the
    # USD-quoted share class (matches ISAC.L/CSPX.L/CCAU.L convention). The
    # LSE listing EXCS.L is GBP-quoted and would inject GBP/USD noise.
    "EXCH": _m("EXCH.AS", "Equity", "Emerging Markets ex-China", "Broad Market"),
    # iShares MSCI China UCITS ETF — Amsterdam USD-quoted listing (matches the
    # EXCH.AS / CEUU.AS Amsterdam convention; same fund Stashaway holds via the
    # ICHN code). Yahoo longName: "iShares MSCI China UCITS ETF", currency USD.
    "ICHN": _m("ICHN.AS", "Equity", "China", "Broad Market UCITS USD"),
    "IDTM": _m("IDTM.L", "Bond", "US", "Treasury 7-10Y"),
    "IDTL": _m("IDTL.L", "Bond", "US", "Treasury 20+Y"),
    "IBTU": _m("IBTU.L", "Bond", "US", "Treasury 0-1Y"),
    "TIP5": _m("TIP5.L", "Bond", "US", "TIPS 0-5Y"),
    "IMBS": _m("IMBS.L", "Bond", "US", "MBS"),
    "IEMB": _m("IEMB.L", "Bond", "Emerging Markets", "USD Sovereign"),
    "IGLN": _m("IGLN.L", "Commodity", "Global", "Gold"),
    # ----------------------------------------------------------------- SGX-listed (Singapore Investing / SG ETF)
    "A35": _m("A35.SI", "Bond", "Singapore", "Aggregate"),
    # MBH.SI is the Amova SGD Investment Grade Corporate Bond Index ETF (not Government).
    "MBH": _m("MBH.SI", "Bond", "Singapore", "Investment Grade Corporate"),
    "G3B": _m("G3B.SI", "Equity", "Singapore", "Broad Market"),
    # CLR.SI is Lion-Phillip S-REIT ETF — Singapore REITs, not pan-Asia equity.
    "CLR": _m("CLR.SI", "Equity", "Singapore", "REITs"),
    # MMS.SI is Phillip SGD Money Market ETF — cash-equivalent, not broad equity.
    "MMS": _m("MMS.SI", "Cash", "Singapore", "Money Market"),
    # QL3.SI = iShares USD Asia High Yield Bond ETF (SGD share class).
    # Stashaway's xlsx mislabels this as "Singapore Equities" but the underlying
    # is actually Asia HY USD bonds — confirmed via Yahoo longName.
    "QL3": _m("QL3.SI", "Bond", "Asia ex-Japan", "High Yield USD"),
    # ----------------------------------------------------------------- ISIN-verified non-trivial mappings (2026-05-30, user-confirmed)
    # Each ticker below was confirmed via Yahoo `Search(ISIN)` returning the
    # listed ticker AND by cross-checking Yahoo's `longName` against the fund
    # name on the JPM SG / LionGlobal product page. DO NOT change any of these
    # mappings without re-verifying via ISIN→Yahoo search and re-confirming
    # with the user — earlier guess-based swaps produced wrong-subfund and
    # wrong-share-class errors that took multiple rounds to catch.
    #
    # ISIN              Stashaway ID  Yahoo ticker        Yahoo longName (acc class, SGD-hedged unless noted)
    # LU2031211308      JINAASH       0P0001I87K.SI       JPM Income A acc SGD Hdg
    # LU1823571978      JPMGASA       0P0001DWBA.SI       JPM Global Bd Opps A acc SGD H
    # LU2646069851      JPEMDSG       0P0001RG8N.SI       JPM Emerging Markets Debt A acc SGDH (hard-currency, not local-ccy)
    # LU2646069695      JPGCBAS       0P0001RG8Q.SI       JPM Global Corporate Bd A acc SGD Hdg
    # LU2778065503      JPGHYHS       0P0001SOG4.SI       JPM Global High Yield Bd A acc SGD Hdg
    # IE00BMD8KM66      BB3M          BB3M.L              JPM BetaBuilders US Treasury Bond 0-3 Months UCITS ETF USD Acc
    # (no ISIN given)   LNWELIA       0P0001DB5Z.SI       LionGlobal SGD Enhanced Liquidity (LSE2; proxy for Simple SGD 30/70 blend)
    # (no ISIN given)   OCBSGDM       0P0001DB5Z.SI       LionGlobal SGD Enhanced Liquidity (LSE2; proxy for Guitsa 30/70 blend)
    # Sector field is intentionally short — it labels the exposure pie-chart
    # bucket. Fund full names + ISINs live in the comment block above.
    "JINAASH": _m("0P0001I87K.SI", "Bond", "Global", "Multi-Asset Income"),
    "JPMGASA": _m("0P0001DWBA.SI", "Bond", "Global", "Unconstrained / Multi-Sector"),
    "JPEMDSG": _m("0P0001RG8N.SI", "Bond", "Emerging Markets", "Hard-Currency Sovereign"),
    "JPGCBAS": _m("0P0001RG8Q.SI", "Bond", "Global", "Investment Grade Corporate"),
    "JPGHYHS": _m("0P0001SOG4.SI", "Bond", "Global", "High Yield"),
    # Stashaway's "BB3M" is the LSE-listed USD share class. ISIN IE00BMD8KM66
    # resolves on Yahoo to BBM3.L (GBP-quoted); BB3M.L is the USD-quoted listing
    # of the same fund — match the trading currency in your USD-base statement.
    "BB3M": _m("BB3M.L", "Cash", "US", "Treasury 0-3M"),
    # LionGlobal SGD funds inside Guitsa / Simple SGD. Stashaway's 70/30 blend
    # per stashaway.sg/simple-var3 maps to the two underlying funds — split per
    # Stashaway ID so each row in user-link file ties to the right NAV. Both
    # funds annualised ~identically (~2.33% / ~1.89% over 5y) so metric impact
    # is negligible vs the prior collapsed mapping.
    "LNWELIA": _m("0P0001DB5Z.SI", "Cash", "Singapore", "LionGlobal SGD Enhanced Liquidity A"),
    "OCBSGDM": _m("0P00006FZD.SI", "Cash", "Singapore", "LionGlobal SGD Money Market A"),
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
