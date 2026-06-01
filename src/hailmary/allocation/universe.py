"""Stashaway asset universe — keyed by Yahoo Ticker.

Each entry maps a **Yahoo Finance ticker** (with exchange suffix where it
matters: `.L`, `.SI`, `.AS`) to asset-class / region / sector metadata used
by the diagnostic engine. The Yahoo Ticker is the canonical identifier
throughout the allocation pipeline — Stashaway's bare-symbol IDs (e.g.
``FLOT``, ``BB3M``, ``CSPX``) are ambiguous about exchange, so the legacy
Stashaway-PDF parser in ``statements.py`` translates them via its own
``_SID_TO_TICKER`` table before constructing ``ParsedHolding`` objects.

Holdings sheet (``data/holding.xlsx``) is the user-maintained source of
truth — its ``Yahoo Ticker`` column matches these keys directly. No
translation needed for sheet-driven loads.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AssetMetadata:
    """Static metadata for a Stashaway-tradeable asset.

    ``data_source`` indicates how truthful the mapped ticker is:
    - ``"real"`` — same fund Stashaway holds (identical NAV)
    - ``"proxy"`` — different fund, structurally similar (composition or
      blend approximation; expect modest drift)
    - ``"synthetic"`` — no real ticker, return series generated locally
      (CASH_USD/CASH_SGD with a published yield)
    """

    ticker: str
    asset_class: str
    region: str
    sector: str | None = None
    data_source: str = "real"  # "real" | "proxy" | "synthetic"


class UnknownAssetError(KeyError):
    """Raised when a ticker is not present in the universe map."""

    def __init__(self, ticker: str, source: str | None = None) -> None:
        self.ticker = ticker
        self.source = source
        msg = f"Unknown asset ticker {ticker!r}"
        if source:
            msg += f" (source: {source})"
        super().__init__(msg)


def _m(
    ticker: str,
    asset_class: str,
    region: str,
    sector: str | None = None,
    *,
    data_source: str = "real",
) -> AssetMetadata:
    return AssetMetadata(
        ticker=ticker,
        asset_class=asset_class,
        region=region,
        sector=sector,
        data_source=data_source,
    )


STASHAWAY_UNIVERSE: dict[str, AssetMetadata] = {
    # ----------------------------------------------------------------- Synthetic cash
    "CASH_USD": _m("CASH_USD", "Cash", "US", "Cash", data_source="synthetic"),
    "CASH_SGD": _m("CASH_SGD", "Cash", "Singapore", "Cash", data_source="synthetic"),
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
    "AAXJ": _m("AAXJ", "Equity", "Asia ex-Japan", "Broad Market"),
    # ----------------------------------------------------------------- Thematic / sector ETFs (custom stacks)
    # ARK Genomic Revolution ETF — Cathie Wood genomics/biotech, used in
    # user's Longevity Stack (25% alongside XLV 74% + cash 1%).
    "ARKG": _m("ARKG", "Equity", "US", "Genomics/Biotech"),
    # VanEck Semiconductors ETF — used in user's AI Power Stack (64%).
    "SMH": _m("SMH", "Equity", "US", "Semiconductors"),
    # First Trust NASDAQ Clean Edge Smart Grid Infrastructure Index ETF —
    # used in user's AI Power Stack (35%).
    "GRID": _m("GRID", "Equity", "US", "Smart Grid Infrastructure"),
    # ----------------------------------------------------------------- US-listed bonds & cash-equivalent
    # iShares $ Floating Rate Bond UCITS ETF — Stashaway uses the UCITS
    # variant (London, USD) for SG investors, same pattern as JEPQ → JEPQ.L
    # and CSPX → CSPX.L. User verified 2026-06-01 ($5.02 @ 2026-05-29).
    "FLOT.L": _m("FLOT.L", "Bond", "UCITS", "Floating Rate"),
    # ----------------------------------------------------------------- US-listed thematic / income
    # Stashaway uses the UCITS variant for SG investors (15% withholding via
    # Ireland-US treaty vs 30% on the US-listed JEPQ). User confirmed 2026-05.
    "JEPQ.L": _m("JEPQ.L", "Equity", "US", "Nasdaq Covered Call (proxy: JEPQ.L UCITS)"),
    # ----------------------------------------------------------------- US-listed alts
    "GLDM": _m("GLDM", "Commodity", "Global", "Gold"),
    # Real Fidelity spot-crypto ETFs Stashaway actually holds. Earlier we used
    # BTC-USD/ETH-USD spot for longer history (FBTC launched 2024-01, FETH
    # 2024-07), but Stashaway buys the ETF wrapper — using the real tickers
    # gives ~0.17pp accuracy on the SRS/Crypto reconciliation. Switch back to
    # spot for any pre-launch historical analytics (task #98).
    "FBTC": _m("FBTC", "Crypto", "Global", "Bitcoin"),
    "FETH": _m("FETH", "Crypto", "Global", "Ethereum"),
    # ----------------------------------------------------------------- LSE-listed UCITS ETFs (BlackRock sleeve)
    "ISAC.L": _m("ISAC.L", "Equity", "Global", "Broad Market"),
    "CSUS.L": _m("CSUS.L", "Equity", "US", "Broad Market"),
    "CSPX.L": _m("CSPX.L", "Equity", "US", "Broad Market"),
    "SASU.L": _m("SASU.L", "Equity", "US", "ESG Screened"),
    "IUIS.L": _m("IUIS.L", "Equity", "US", "Industrials"),
    "IJPA.L": _m("IJPA.L", "Equity", "Japan", "Broad Market"),
    "IJPD.L": _m("IJPD.L", "Equity", "Japan", "Hedged"),
    "ISFD.L": _m("ISFD.L", "Equity", "UK", "Broad Market"),
    "CCAU.L": _m("CCAU.L", "Equity", "Canada", "Broad Market"),
    # iShares Core MSCI EMU UCITS ETF EUR (Acc), USD-quoted Amsterdam listing
    # (ISIN IE00BKBF6616). Same fund as CEU1.L/IEMU.L but Amsterdam is the
    # USD-quoted primary on Yahoo — matches the EXCH.AS convention.
    "CEUU.AS": _m("CEUU.AS", "Equity", "Eurozone", "Broad Market"),
    "CPXJ.L": _m("CPXJ.L", "Equity", "Pacific ex-Japan", "Broad Market"),
    # iShares MSCI EM ex-China UCITS ETF USD Acc — Amsterdam listing is the
    # USD-quoted share class (matches ISAC.L/CSPX.L/CCAU.L convention). The
    # LSE listing EXCS.L is GBP-quoted and would inject GBP/USD noise.
    "EXCH.AS": _m("EXCH.AS", "Equity", "Emerging Markets ex-China", "Broad Market"),
    # iShares MSCI China UCITS ETF — Amsterdam USD-quoted listing (matches the
    # EXCH.AS / CEUU.AS Amsterdam convention; same fund Stashaway holds via the
    # ICHN code). Yahoo longName: "iShares MSCI China UCITS ETF", currency USD.
    "ICHN.AS": _m("ICHN.AS", "Equity", "China", "Broad Market UCITS USD"),
    "IDTM.L": _m("IDTM.L", "Bond", "US", "Treasury 7-10Y"),
    "IDTL.L": _m("IDTL.L", "Bond", "US", "Treasury 20+Y"),
    "IBTU.L": _m("IBTU.L", "Bond", "US", "Treasury 0-1Y"),
    "TIP5.L": _m("TIP5.L", "Bond", "US", "TIPS 0-5Y"),
    "IMBS.L": _m("IMBS.L", "Bond", "US", "MBS"),
    "IEMB.L": _m("IEMB.L", "Bond", "Emerging Markets", "USD Sovereign"),
    "IGLN.L": _m("IGLN.L", "Commodity", "Global", "Gold"),
    # ----------------------------------------------------------------- SGX-listed (Singapore Investing / SG ETF)
    "A35.SI": _m("A35.SI", "Bond", "Singapore", "Aggregate"),
    # MBH.SI is the Amova SGD Investment Grade Corporate Bond Index ETF (not Government).
    "MBH.SI": _m("MBH.SI", "Bond", "Singapore", "Investment Grade Corporate"),
    "G3B.SI": _m("G3B.SI", "Equity", "Singapore", "Broad Market"),
    # CLR.SI is Lion-Phillip S-REIT ETF — Singapore REITs, not pan-Asia equity.
    "CLR.SI": _m("CLR.SI", "Equity", "Singapore", "REITs"),
    # MMS.SI is Phillip SGD Money Market ETF — cash-equivalent, not broad equity.
    "MMS.SI": _m("MMS.SI", "Cash", "Singapore", "Money Market"),
    # QL3.SI = iShares USD Asia High Yield Bond ETF (SGD share class).
    # Stashaway's xlsx mislabels this as "Singapore Equities" but the underlying
    # is actually Asia HY USD bonds — confirmed via Yahoo longName.
    "QL3.SI": _m("QL3.SI", "Bond", "Asia ex-Japan", "High Yield USD"),
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
    "0P0001I87K.SI": _m("0P0001I87K.SI", "Bond", "Global", "Multi-Asset Income"),
    "0P0001DWBA.SI": _m("0P0001DWBA.SI", "Bond", "Global", "Unconstrained / Multi-Sector"),
    "0P0001RG8N.SI": _m("0P0001RG8N.SI", "Bond", "Emerging Markets", "Hard-Currency Sovereign"),
    "0P0001RG8Q.SI": _m("0P0001RG8Q.SI", "Bond", "Global", "Investment Grade Corporate"),
    "0P0001SOG4.SI": _m("0P0001SOG4.SI", "Bond", "Global", "High Yield"),
    # Stashaway's "BB3M" is the LSE-listed USD share class. ISIN IE00BMD8KM66
    # resolves on Yahoo to BBM3.L (GBP-quoted); BB3M.L is the USD-quoted listing
    # of the same fund — match the trading currency in your USD-base statement.
    "BB3M.L": _m("BB3M.L", "Cash", "US", "Treasury 0-3M"),
    # LionGlobal SGD funds inside Guitsa / Simple SGD. Stashaway's 70/30 blend
    # per stashaway.sg/simple-var3 maps to the two underlying funds — split per
    # Stashaway ID so each row in user-link file ties to the right NAV. Both
    # funds annualised ~identically (~2.33% / ~1.89% over 5y) so metric impact
    # is negligible vs the prior collapsed mapping.
    "0P0001DB5Z.SI": _m("0P0001DB5Z.SI", "Cash", "Singapore", "LionGlobal SGD Enhanced Liquidity A"),
    # OCBSGDM (the 30% MMF leg) is wired to the real LionGlobal SGD MMF.
    "0P00006FZD.SI": _m("0P00006FZD.SI", "Cash", "Singapore", "LionGlobal SGD Money Market A"),
}
"""Yahoo Ticker → AssetMetadata.

Maintained against the user's Holdings sheet (`data/holding.xlsx`). LSE-listed
UCITS ETFs and SGX-listed funds carry exchange suffixes (`.L`, `.SI`, `.AS`)
on Yahoo Finance; Stashaway-only managed funds without public tickers use
their `0P0...` Yahoo identifiers (e.g. JPM SGD-hedged share classes).
"""


def resolve(ticker: str, *, source: str | None = None) -> AssetMetadata:
    """Look up *ticker* in :data:`STASHAWAY_UNIVERSE`.

    Raises :class:`UnknownAssetError` (with the source portfolio name if given)
    when the ticker is missing, rather than silently dropping the holding.
    """
    try:
        return STASHAWAY_UNIVERSE[ticker]
    except KeyError as exc:
        raise UnknownAssetError(ticker, source=source) from exc
