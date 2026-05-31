"""One-shot audit of every holding in the book.

For each Stashaway ID actually present in the latest statement, prints a row
showing:
- the PDF fund name (best-effort extracted from the raw statement text)
- the currently wired ticker (per universe.py)
- the Yahoo longName for that ticker (or 'synthetic' for CASH_*)
- the universe-map sector annotation

Use this to eyeball mismatches between what Stashaway holds and what we
proxy it to. Run: `.venv/Scripts/python.exe scripts/audit_holdings.py`
"""

from __future__ import annotations

import re
import warnings

import pdfplumber
import yfinance as yf

from hailmary.allocation.book_config import ROLES
from hailmary.allocation.statements import parse_statement
from hailmary.allocation.universe import STASHAWAY_UNIVERSE

warnings.filterwarnings("ignore")

PDF = "data/statements/2026-04 StashAway Monthly Statement.pdf"


def extract_pdf_names() -> dict[str, str]:
    """Walk the PDF and pair each stashaway-code (in parens) with the preceding fund name."""
    ticker_re = re.compile(r"\(([A-Z0-9_]{3,15})\)")
    with pdfplumber.open(PDF) as pdf:
        full_text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    lines = full_text.splitlines()
    pdf_names: dict[str, str] = {}
    for i, line in enumerate(lines):
        for m in ticker_re.finditer(line):
            sid = m.group(1)
            if sid in pdf_names:
                continue
            name_chunk = line[: m.start()].strip()
            # If the line starts with the code, peek at the previous line(s)
            if not name_chunk and i > 0:
                name_chunk = lines[i - 1].strip()
            else:
                # Some funds wrap across two lines — prepend previous line if it
                # ends mid-name (no $ amount, no trailing punctuation)
                if i > 0:
                    prev = lines[i - 1].strip()
                    if prev and not prev.startswith("$") and "$" not in prev.split()[-1]:
                        if any(k in prev for k in ("Fund", "ETF", "Bond", "Equity", "Hedged", "SGD", "USD", "Bills", "Cash", "Trust")):
                            name_chunk = (prev + " " + name_chunk).strip()
            pdf_names[sid] = name_chunk
    return pdf_names


def yahoo_long_name(ticker: str) -> str:
    if ticker.startswith("CASH_"):
        return f"(synthetic — {ticker})"
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or "?"
    except Exception as e:
        return f"ERR: {e}"


def main() -> None:
    parsed = parse_statement(PDF)
    sids_in_book = sorted(
        {h.stashaway_id for p in parsed for h in p.holdings if p.name in ROLES}
    )
    pdf_names = extract_pdf_names()

    header_fmt = "{:12s} {:18s} {:50s} | {:60s}"
    print(header_fmt.format("Stashaway ID", "Wired ticker", "Yahoo longName", "PDF fund name"))
    print("-" * 145)
    for sid in sids_in_book:
        meta = STASHAWAY_UNIVERSE.get(sid)
        pdf = pdf_names.get(sid, "?")
        if meta is None:
            print(header_fmt.format(sid, "(unmapped)", "-", pdf[:60]))
            continue
        ticker = meta.ticker
        ln = yahoo_long_name(ticker)
        print(header_fmt.format(sid, ticker, ln[:50], pdf[:60]))


if __name__ == "__main__":
    main()
