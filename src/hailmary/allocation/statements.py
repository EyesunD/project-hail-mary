"""Stashaway PDF statement parser, JSON fallback, and parquet cache.

The parser uses :mod:`pdfplumber` to extract per-portfolio holdings tables.
Stashaway statement layouts vary between portfolio types (Simple, General
Investing, Custom, etc.) but share a common shape: a header line giving the
portfolio name and total value, followed by a holdings table whose rows look
roughly like ``ticker | name | weight% | value``.

This implementation aims to be *robust enough for the user's actual
statements* rather than format-perfect. The ``StatementParseError`` carries
enough context (file, portfolio, reason) to identify the failing statement
quickly, and a manual JSON fallback (:func:`load_holdings_from_json`)
produces the same shape so the rest of the pipeline keeps working when the
PDF parser blows up on a new format.

Cache: parsed output is persisted to parquet keyed by ``(filename, mtime)``,
mirroring the pattern in :mod:`hailmary.data.cache`. Repeat parses of the same
file skip pdfplumber entirely.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    pass


WEIGHT_TOLERANCE = 1e-4
DEFAULT_CACHE_DIR = Path.home() / ".hailmary" / "cache" / "statements"


# ---------------------------------------------------------------------------
# Errors & dataclasses
# ---------------------------------------------------------------------------


class StatementParseError(Exception):
    """Raised when a Stashaway statement cannot be parsed.

    Carries the source file path and (when known) the portfolio whose table
    failed, so the caller can identify the failing statement precisely.
    """

    def __init__(
        self, message: str, *, path: Path | str | None = None, portfolio: str | None = None
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.portfolio = portfolio
        prefix_parts = []
        if self.path is not None:
            prefix_parts.append(str(self.path.name))
        if portfolio:
            prefix_parts.append(f"portfolio={portfolio!r}")
        prefix = f"[{' '.join(prefix_parts)}] " if prefix_parts else ""
        super().__init__(prefix + message)


@dataclass(frozen=True, slots=True)
class ParsedHolding:
    """One row of a parsed holdings table."""

    stashaway_id: str
    weight: float
    value: float


@dataclass(slots=True)
class ParsedPortfolio:
    """One portfolio extracted from a Stashaway statement."""

    name: str
    statement_date: date
    currency: str
    total_value: float
    holdings: list[ParsedHolding] = field(default_factory=list)
    statement_fx_usd_sgd: float | None = None
    """Statement-date USDSGD rate as reported by Stashaway (e.g. 1.2732).

    Yahoo's USDSGD=X labels dates in UK time, so its "30 Apr close" lands ~7
    hours before Stashaway's Singapore-EOD snapshot. Using this rate (parsed
    from the PDF) keeps statement-date SGD totals consistent with the app.
    """

    def weight_sum(self) -> float:
        return sum(h.weight for h in self.holdings)

    def validate_weights(self, *, tolerance: float = WEIGHT_TOLERANCE) -> None:
        """Raise :class:`StatementParseError` if weights don't sum to 1."""
        s = self.weight_sum()
        if abs(s - 1.0) > tolerance:
            raise StatementParseError(
                f"Holdings weights sum to {s:.6f} (expected 1.0 ± {tolerance})",
                portfolio=self.name,
            )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CacheKey:
    name: str
    mtime_ns: int

    def safe_name(self) -> str:
        # Replace dots and any path-unfriendly chars so suffixing later is unambiguous.
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", self.name)
        return f"{safe}__{self.mtime_ns}"


class StatementCache:
    """Parquet cache keyed by ``(filename, mtime_ns)``.

    Stores per-portfolio rows as parquet plus a small JSON metadata file with
    the per-portfolio header info (name, date, currency, total_value).
    """

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: _CacheKey) -> tuple[Path, Path]:
        stem = key.safe_name()
        return self.cache_dir / f"{stem}.parquet", self.cache_dir / f"{stem}.meta.json"

    def get(self, key: _CacheKey) -> list[ParsedPortfolio] | None:
        parquet, meta = self._paths(key)
        if not parquet.exists() or not meta.exists():
            return None
        try:
            df = pd.read_parquet(parquet)
            metadata = json.loads(meta.read_text())
        except Exception as exc:
            logger.warning("Statement cache read failed for {}: {}", key.name, exc)
            return None
        portfolios: list[ParsedPortfolio] = []
        for header in metadata["portfolios"]:
            sub = df[df["portfolio"] == header["name"]]
            holdings = [
                ParsedHolding(
                    stashaway_id=row.stashaway_id,
                    weight=float(row.weight),
                    value=float(row.value),
                )
                for row in sub.itertuples(index=False)
            ]
            stmt_fx = header.get("statement_fx_usd_sgd")
            portfolios.append(
                ParsedPortfolio(
                    name=header["name"],
                    statement_date=date.fromisoformat(header["statement_date"]),
                    currency=header["currency"],
                    total_value=float(header["total_value"]),
                    holdings=holdings,
                    statement_fx_usd_sgd=float(stmt_fx) if stmt_fx is not None else None,
                )
            )
        logger.debug("Statement cache hit for {}", key.name)
        return portfolios

    def set(self, key: _CacheKey, portfolios: list[ParsedPortfolio]) -> None:
        parquet, meta = self._paths(key)
        rows = [
            {"portfolio": p.name, **asdict(h)}
            for p in portfolios
            for h in p.holdings
        ]
        df = pd.DataFrame(rows, columns=["portfolio", "stashaway_id", "weight", "value"])
        df.to_parquet(parquet)
        meta.write_text(
            json.dumps(
                {
                    "portfolios": [
                        {
                            "name": p.name,
                            "statement_date": p.statement_date.isoformat(),
                            "currency": p.currency,
                            "total_value": p.total_value,
                            "statement_fx_usd_sgd": p.statement_fx_usd_sgd,
                        }
                        for p in portfolios
                    ]
                }
            )
        )
        logger.debug("Statement cache write {} portfolios for {}", len(portfolios), key.name)


_default_cache: StatementCache | None = None


def _get_default_cache() -> StatementCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = StatementCache()
    return _default_cache


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


# Stashaway holdings are reported as-of the period-end. The PDF contains two
# sentences with "as of DD MMM YYYY": one for the opening balance (prior month
# end) and one for the closing balance (this period's end, which is what we
# want). Take the LATEST date from all such matches.
_AS_OF_DATE_PATTERN = re.compile(
    r"as\s+of\s+(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})",
    re.IGNORECASE,
)
# Stashaway reports both opening and closing FX rates: "(1 USD = 1.2868 SGD)"
# for 31 Mar and "(1 USD = 1.2732 SGD)" for 30 Apr. We want the closing
# (Singapore-EOD on the statement date), so take the LAST match in the text.
_USDSGD_RATE_PATTERN = re.compile(
    r"\(?\s*1\s*USD\s*=\s*(\d+\.\d+)\s*SGD\s*\)?",
    re.IGNORECASE,
)
_DATE_RANGE_PATTERNS = [
    re.compile(
        r"\b(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})\s*[-–—to]+\s*"
        r"(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})\b"
    ),
]
_DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})\b"),  # "30 Sep 2024"
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),  # ISO
]
_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}
_MONTHS.update({m[:3].lower(): i for m, i in list(_MONTHS.items())})

_CURRENCY_RE = re.compile(r"\b(SGD|USD|EUR|GBP|HKD|JPY)\b")
_PCT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")
_MONEY_RE = re.compile(r"-?\$?-?[\d,]+\.\d{2}")
_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.]{0,7})\)")
_PORTFOLIO_HEADER_RE = re.compile(r"^PORTFOLIO DETAILS\s*/")
_BLOCK_TERMINATOR_RE = re.compile(
    r"^(TRANSACTIONS\s*/|FEES\s*/|DISCLOSURES?|IMPORTANT\s+NOTES?|"
    r"PERFORMANCE\s*/|INVESTMENT\s+PORTFOLIOS?\s*$|CASH\s+MANAGEMENT)"
)
_REPORTING_CCY_RE = re.compile(r"Reporting Currency:\s*([A-Z]{3})")
_HEADER_NOISE_PATTERNS = (
    "Opening Balance",
    "Bought Sold",
    "Closing Balance",
    "Profit/Loss",
    "Outstanding",
    "Reporting Currency",
    "(Incl. Dividend",
    "(Paid and",
    "Entitlements)",
    "Reinvested)",
    "Units",
)


def _parse_date(text: str) -> date | None:
    as_of_dates: list[date] = []
    for m in _AS_OF_DATE_PATTERN.finditer(text):
        month = _MONTHS.get(m.group(2).lower())
        if month is None:
            continue
        try:
            as_of_dates.append(date(int(m.group(3)), month, int(m.group(1))))
        except ValueError:
            continue
    if as_of_dates:
        return max(as_of_dates)
    for pat in _DATE_RANGE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        day = int(m.group(4))
        month = _MONTHS.get(m.group(5).lower())
        year = int(m.group(6))
        if month is not None:
            return date(year, month, day)
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        groups = m.groups()
        if len(groups) == 3 and groups[1].isalpha():
            day = int(groups[0])
            month = _MONTHS.get(groups[1].lower())
            year = int(groups[2])
            if month is not None:
                return date(year, month, day)
        elif len(groups) == 3:
            return date(int(groups[0]), int(groups[1]), int(groups[2]))
    return None


def _parse_statement_fx_rate(text: str) -> float | None:
    """Latest "1 USD = X SGD" rate in *text*. Stashaway prints opening then
    closing; the last match is the closing (statement-date) rate."""
    matches = list(_USDSGD_RATE_PATTERN.finditer(text))
    if not matches:
        return None
    try:
        return float(matches[-1].group(1))
    except (ValueError, IndexError):
        return None


def _parse_number(s: str) -> float | None:
    s = s.replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_statement(
    path: Path | str,
    *,
    use_cache: bool = True,
    cache: StatementCache | None = None,
) -> list[ParsedPortfolio]:
    """Parse a Stashaway monthly PDF statement into a list of ``ParsedPortfolio``.

    Stashaway statements are positioned-text PDFs; ``pdfplumber.extract_tables``
    returns nothing useful. Instead we walk extracted text lines, split on
    ``PORTFOLIO DETAILS / <month> <year>`` headers, and parse holding rows by
    finding ``(TICKER)`` regex matches and the associated dollar figures.

    Weights are derived from ``closing_value / total_value`` since the PDF's
    "% Total" column is profit/loss percent, not portfolio weight. Cash rows
    (``Cash USD`` / ``Cash SGD``) are emitted as synthetic ``CASH_<CCY>``
    holdings so weights still sum to 1.

    The result is cached by ``(filename, mtime)`` so repeat reads skip
    pdfplumber. Pass ``use_cache=False`` to force a re-parse.
    """
    p = Path(path)
    if not p.exists():
        raise StatementParseError(f"File not found: {p}", path=p)

    cache = cache or _get_default_cache()
    key = _CacheKey(name=p.name, mtime_ns=p.stat().st_mtime_ns)
    if use_cache:
        cached = cache.get(key)
        if cached is not None:
            return cached

    try:
        import pdfplumber
    except ImportError as exc:
        raise StatementParseError(
            "pdfplumber is required to parse PDF statements. "
            'Install with: pip install -e ".[allocation]"',
            path=p,
        ) from exc

    with pdfplumber.open(p) as pdf:
        all_text = "\n".join((page.extract_text() or "") for page in pdf.pages)

    statement_date = _parse_date(all_text[:4000])
    if statement_date is None:
        raise StatementParseError("Could not find statement date on cover page", path=p)

    statement_fx = _parse_statement_fx_rate(all_text[:8000])

    portfolios = _parse_portfolio_blocks(all_text, statement_date=statement_date, path=p)
    if not portfolios:
        raise StatementParseError(
            "No PORTFOLIO DETAILS sections found; PDF format may have changed.",
            path=p,
        )
    for portfolio in portfolios:
        portfolio.statement_fx_usd_sgd = statement_fx

    for portfolio in portfolios:
        portfolio.validate_weights()

    if use_cache:
        cache.set(key, portfolios)
    return portfolios


def _parse_portfolio_blocks(
    all_text: str,
    *,
    statement_date: date,
    path: Path,
) -> list[ParsedPortfolio]:
    """Split *all_text* on ``PORTFOLIO DETAILS /`` headers and parse each block.

    Multiple consecutive headers may appear when a portfolio's holdings table
    spans pages — in that case the second header repeats the column titles but
    not the portfolio name; we merge those into the previous block.
    """
    lines = all_text.split("\n")
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        stripped = line.strip()
        if _PORTFOLIO_HEADER_RE.match(stripped):
            if current is not None:
                blocks.append(current)
            current = []
            continue
        if current is not None and _BLOCK_TERMINATOR_RE.match(stripped):
            blocks.append(current)
            current = None
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        blocks.append(current)

    # Merge continuation blocks: if a block has no Reporting Currency line in
    # its first ~5 lines AND the previous block had Cash/Total still pending,
    # treat it as a continuation. Safer heuristic: merge if the block's first
    # non-empty line is a column-header noise line (no portfolio name).
    merged: list[list[str]] = []
    for block in blocks:
        first_meaningful = _first_meaningful_line(block)
        if (
            merged
            and first_meaningful is not None
            and (
                any(noise in first_meaningful for noise in _HEADER_NOISE_PATTERNS)
                or "Reporting Currency" not in "\n".join(block[:5])
            )
            and "Reporting Currency" not in "\n".join(block[:5])
        ):
            merged[-1].extend(block)
        else:
            merged.append(block)

    portfolios: list[ParsedPortfolio] = []
    for block in merged:
        portfolio = _parse_portfolio_block(block, statement_date=statement_date, path=path)
        if portfolio is not None:
            portfolios.append(portfolio)
    return portfolios


def _first_meaningful_line(block: list[str]) -> str | None:
    for ln in block:
        s = ln.strip()
        if s:
            return s
    return None


def _parse_portfolio_block(
    block: list[str],
    *,
    statement_date: date,
    path: Path,
) -> ParsedPortfolio | None:
    """Parse one ``PORTFOLIO DETAILS`` block into a :class:`ParsedPortfolio`."""
    name = _first_meaningful_line(block)
    if not name or any(noise in name for noise in _HEADER_NOISE_PATTERNS):
        return None

    block_text = "\n".join(block)
    ccy_match = _REPORTING_CCY_RE.search(block_text)
    currency = ccy_match.group(1) if ccy_match else "USD"

    holdings, total_value = _parse_holdings_lines(block, currency=currency)
    if not holdings:
        return None
    if total_value is None or total_value <= 0:
        # Fall back to summing what we extracted.
        total_value = sum(h.value for h in holdings)

    return ParsedPortfolio(
        name=name,
        statement_date=statement_date,
        currency=currency,
        total_value=total_value,
        holdings=_normalise_weights(holdings, total_value),
    )


def _normalise_weights(
    holdings: list[ParsedHolding], total_value: float
) -> list[ParsedHolding]:
    """Recompute weights from values so they sum cleanly to 1."""
    raw_sum = sum(h.value for h in holdings)
    denom = total_value if total_value > 0 else raw_sum
    if denom <= 0:
        return holdings
    rescaled = [
        ParsedHolding(stashaway_id=h.stashaway_id, weight=h.value / denom, value=h.value)
        for h in holdings
    ]
    drift = 1.0 - sum(h.weight for h in rescaled)
    if rescaled and abs(drift) > 1e-12:
        last = rescaled[-1]
        rescaled[-1] = ParsedHolding(
            stashaway_id=last.stashaway_id,
            weight=last.weight + drift,
            value=last.value,
        )
    return rescaled


def _parse_holdings_lines(
    block: list[str], *, currency: str
) -> tuple[list[ParsedHolding], float | None]:
    """Walk lines in *block* and return (holdings, total_value).

    A holding row is identified by either:
      - a line containing both a ``$`` figure AND a ``(TICKER)`` token, or
      - a line containing ``$`` figures whose next non-empty line contains a
        bare ``(TICKER)`` token (name wrapped onto the next line).
    """
    holdings: list[ParsedHolding] = []
    total_value: float | None = None
    cash_value = 0.0

    i = 0
    n = len(block)
    while i < n:
        line = block[i]
        stripped = line.strip()

        if not stripped or any(noise in stripped for noise in _HEADER_NOISE_PATTERNS):
            i += 1
            continue

        # Total row — captures portfolio total
        if stripped.startswith("Total ") or stripped == "Total":
            total_value = _last_money_in(stripped)
            i += 1
            continue

        # Cash row — synthetic holding
        cash_match = re.match(r"^Cash\s+([A-Z]{3})", stripped)
        if cash_match:
            cash_ccy = cash_match.group(1)
            value = _last_money_in(stripped) or 0.0
            cash_value += value
            holdings.append(
                ParsedHolding(stashaway_id=f"CASH_{cash_ccy}", weight=0.0, value=value)
            )
            i += 1
            continue

        # Real holding row — needs a $ value
        if "$" not in stripped:
            i += 1
            continue

        ticker = _find_ticker(stripped)
        consumed = 1
        if ticker is None:
            # Name may have wrapped over the next 1–3 lines. Walk forward until
            # we find a `(TICKER)` token, OR we hit another holding row (line
            # with `$` figures) — in which case give up on the current row.
            for ahead in range(1, 4):
                if i + ahead >= n:
                    break
                next_line = block[i + ahead].strip()
                if not next_line:
                    continue
                if "$" in next_line:
                    break
                t = _find_ticker(next_line)
                if t is not None:
                    ticker = t
                    consumed = ahead + 1
                    break

        if ticker is None:
            i += 1
            continue

        value = _extract_closing_value(stripped)
        if value is None:
            i += consumed
            continue

        holdings.append(ParsedHolding(stashaway_id=ticker, weight=0.0, value=value))
        i += consumed

    return holdings, total_value


def _find_ticker(line: str) -> str | None:
    m = _TICKER_RE.search(line)
    return m.group(1) if m else None


def _extract_closing_value(line: str) -> float | None:
    """Extract the *closing balance* value from a holding line.

    Format: ``<name> <opening> <bought> <sold> <dividends> <closing> <pl%> <units>``
    The closing balance is the last $-value before the percent sign. If no
    percent is present, fall back to the last $-value on the line.
    """
    pct_match = _PCT_RE.search(line)
    haystack = line[: pct_match.start()] if pct_match else line
    money_matches = list(_MONEY_RE.finditer(haystack))
    if not money_matches:
        return None
    return _parse_money(money_matches[-1].group(0))


def _last_money_in(line: str) -> float | None:
    matches = list(_MONEY_RE.finditer(line))
    if not matches:
        return None
    return _parse_money(matches[-1].group(0))


def _parse_money(s: str) -> float:
    cleaned = s.replace("$", "").replace(",", "")
    return float(cleaned)


# ---------------------------------------------------------------------------
# Manual JSON fallback
# ---------------------------------------------------------------------------


def load_holdings_from_json(path: Path | str) -> list[ParsedPortfolio]:
    """Load manually-curated holdings JSON when pdfplumber fails on a new format.

    The JSON shape is::

        {
          "statement_date": "2024-09-30",
          "currency": "SGD",
          "portfolios": [
            {
              "name": "Custom — Defensive",
              "total_value": 123456.78,
              "holdings": [
                {"stashaway_id": "VTI", "weight": 0.40, "value": 49382.71},
                {"stashaway_id": "BND", "weight": 0.60, "value": 74074.07}
              ]
            }
          ]
        }
    """
    p = Path(path)
    if not p.exists():
        raise StatementParseError(f"File not found: {p}", path=p)
    try:
        payload: dict[str, Any] = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise StatementParseError(f"Invalid JSON: {exc}", path=p) from exc

    try:
        statement_date = date.fromisoformat(payload["statement_date"])
        currency = payload["currency"]
        raw_portfolios = payload["portfolios"]
    except KeyError as exc:
        raise StatementParseError(f"Missing required key: {exc}", path=p) from exc

    portfolios: list[ParsedPortfolio] = []
    for raw in raw_portfolios:
        try:
            holdings = [
                ParsedHolding(
                    stashaway_id=h["stashaway_id"],
                    weight=float(h["weight"]),
                    value=float(h.get("value", 0.0)),
                )
                for h in raw["holdings"]
            ]
            portfolio = ParsedPortfolio(
                name=raw["name"],
                statement_date=statement_date,
                currency=currency,
                total_value=float(raw["total_value"]),
                holdings=holdings,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StatementParseError(
                f"Malformed portfolio entry: {exc}",
                path=p,
                portfolio=raw.get("name") if isinstance(raw, dict) else None,
            ) from exc
        portfolio.validate_weights()
        portfolios.append(portfolio)
    return portfolios
