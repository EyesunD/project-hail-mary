"""Snapshot Holdings tickers vs Yahoo: highlight new ones, list verified ones.

Run after editing `data/holding.xlsx` (Holdings sheet) — e.g. adding a new
sleeve, swapping a ticker, or re-allocating within a sleeve:

    .venv/Scripts/python.exe scripts/holdings_prices.py

Writes `reports/new_tickers.xlsx` with two sheets:

- **New Tickers** — Yahoo Tickers in the sheet that are NOT yet wired in
  `universe.py`. Cross-check the Last Close against the Stashaway (or
  Yahoo) website to confirm the ticker is the right fund, then wire it
  into `universe.py`. It will stop appearing here next run.
- **Verified Tickers** — every Holdings row whose SID resolves through
  `universe.py`, with the wired ticker's Yahoo longName / currency /
  last close. Asset column hyperlinks to the Stashaway app deep-link
  so you can jump straight to the holding in the app.

yfinance is hit once per unique ticker (cached within the run), so the
script takes ~30-60s to fetch ~60 tickers.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import yfinance as yf

from hailmary.allocation.universe import STASHAWAY_UNIVERSE

warnings.filterwarnings("ignore")

LINKS = Path("data/holding.xlsx")
OUT = Path("reports/new_tickers.xlsx")

_LINK_RE = re.compile(r"/asset-details/([^/]+)/")
_CASH_ALIAS = {"USD": "CASH_USD", "SGD": "CASH_SGD"}


def load_holdings_tickers() -> list[dict[str, str]]:
    """Return one entry per Holdings row: ``{Port, Asset, Yahoo Ticker, Link URL}``.

    The ``Yahoo Ticker`` column is the canonical lookup key against
    :data:`STASHAWAY_UNIVERSE`. For cash rows (which have no Yahoo Ticker)
    we fall back to the Stashaway hyperlink: USD → ``CASH_USD``,
    SGD → ``CASH_SGD``. The Link URL is carried through purely for the
    clickable Asset hyperlink in the rendered Excel.
    """
    if not LINKS.exists():
        return []
    import openpyxl

    wb = openpyxl.load_workbook(LINKS, data_only=False)
    ws = wb["Holdings"]
    HEADER_ROW = 7
    hdr: dict[str, int] = {}
    for c in range(1, ws.max_column + 1):
        v = ws.cell(row=HEADER_ROW, column=c).value
        if v is not None and str(v).strip():
            hdr[str(v).strip()] = c

    out: list[dict[str, str]] = []
    for r in range(HEADER_ROW + 1, ws.max_row + 1):
        port = ws.cell(row=r, column=hdr.get("Port", 6)).value
        asset = ws.cell(row=r, column=hdr.get("Asset", 9)).value
        link_cell = ws.cell(row=r, column=hdr.get("Link", 10))
        if port is None or asset is None or link_cell.value is None:
            continue
        link_url = link_cell.hyperlink.target if link_cell.hyperlink else ""
        yt = ""
        if "Yahoo Ticker" in hdr:
            v = ws.cell(row=r, column=hdr["Yahoo Ticker"]).value
            yt = str(v).strip() if v is not None else ""
        if not yt:
            # Cash row fallback — Stashaway uses USD/SGD as the SID for cash
            m = _LINK_RE.search(str(link_url))
            if m:
                yt = _CASH_ALIAS.get(m.group(1).upper(), "")
        out.append({"Port": str(port).strip(), "Asset": str(asset),
                    "Yahoo Ticker": yt, "Link URL": link_url})
    return out


def fetch_snapshot(ticker: str) -> dict[str, object]:
    tk = yf.Ticker(ticker)
    info = tk.info
    long_name = info.get("longName") or info.get("shortName") or "?"
    ccy = info.get("currency") or "?"
    try:
        h = tk.history(period="5d", auto_adjust=True)
        last_close = float(h.iloc[-1]["Close"]) if not h.empty else float("nan")
        as_of = h.index[-1].date().isoformat() if not h.empty else "?"
    except Exception:
        last_close = float("nan")
        as_of = "?"
    return {"longName": long_name, "currency": ccy,
            "last_close": last_close, "as_of": as_of}


def main() -> None:
    rows = load_holdings_tickers()
    if not rows:
        print("Holdings sheet is empty or unreadable.")
        return

    # Bucket each row by whether its Yahoo Ticker resolves in universe.py.
    new_refs: dict[str, list[dict[str, str]]] = {}
    verified_rows: list[dict[str, object]] = []
    for r in rows:
        yt = r["Yahoo Ticker"]
        if not yt:
            continue
        if yt in STASHAWAY_UNIVERSE:
            verified_rows.append({**r, "Wired Ticker": yt})
        else:
            new_refs.setdefault(yt, []).append(r)

    # Fetch yfinance for every unique ticker we'll display (new + verified).
    needed: set[str] = set(new_refs) | {r["Wired Ticker"] for r in verified_rows}
    cache: dict[str, dict[str, object]] = {}
    print(f"Fetching yfinance for {len(needed)} unique tickers...")
    for i, t in enumerate(sorted(needed), start=1):
        cache[t] = fetch_snapshot(t)
        print(f"  [{i}/{len(needed)}] {t}: {cache[t]['longName']}")

    # Flatten new_refs into one row per (sleeve, asset, ticker) - same shape
    # as verified_rows so both sheets share the rendering helper.
    new_rows: list[dict[str, object]] = []
    for yt, refs in new_refs.items():
        for r in refs:
            new_rows.append({**r, "Display Ticker": yt})
    for vr in verified_rows:
        vr["Display Ticker"] = vr["Wired Ticker"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from datetime import date as _date

    wb = Workbook()
    header_fill = PatternFill("solid", fgColor="161B22")
    header_font = Font(color="E6EDF3", bold=True)
    portfolio_fill = PatternFill("solid", fgColor="1F2730")
    portfolio_font = Font(color="58A6FF", bold=True, size=12)
    link_font = Font(color="58A6FF", underline="single")

    def render_grouped_sheet(
        ws_,
        ticker_header: str,
        rows_: list[dict[str, object]],
        note_text: str | None = None,
    ) -> None:
        """Render a portfolio-grouped sheet: portfolio header rows + clickable
        Asset hyperlinks + yfinance snapshot columns."""
        headers = ["Portfolio", "Asset", ticker_header, "longName", "Currency",
                   "Last Close", "As Of"]
        for c, name in enumerate(headers, start=1):
            cell = ws_.cell(row=1, column=c, value=name)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(
                horizontal="left" if c <= 4 else "right", vertical="center")
        r_idx = 2
        rows_sorted = sorted(rows_, key=lambda r: (str(r["Port"]), str(r["Asset"])))
        for port_name in dict.fromkeys(r["Port"] for r in rows_sorted):
            group = [r for r in rows_sorted if r["Port"] == port_name]
            ws_.merge_cells(start_row=r_idx, start_column=1,
                            end_row=r_idx, end_column=len(headers))
            c = ws_.cell(row=r_idx, column=1,
                         value=f"  {port_name}   ({len(group)} holding"
                               f"{'s' if len(group) > 1 else ''})")
            c.fill = portfolio_fill
            c.font = portfolio_font
            c.alignment = Alignment(horizontal="left", vertical="center")
            r_idx += 1
            for hr in group:
                t = str(hr["Display Ticker"])
                snap = cache[t]
                ws_.cell(row=r_idx, column=1, value=hr["Port"])
                asset_cell = ws_.cell(row=r_idx, column=2, value=hr["Asset"])
                if hr.get("Link URL"):
                    asset_cell.hyperlink = str(hr["Link URL"])
                    asset_cell.font = link_font
                ws_.cell(row=r_idx, column=3, value=t)
                ws_.cell(row=r_idx, column=4, value=snap["longName"])
                ws_.cell(row=r_idx, column=5, value=snap["currency"])
                ws_.cell(row=r_idx, column=6, value=snap["last_close"])
                ws_.cell(row=r_idx, column=7, value=snap["as_of"])
                r_idx += 1
        for col, w in zip("ABCDEFG", [22, 32, 16, 50, 10, 14, 14], strict=True):
            ws_.column_dimensions[col].width = w
        ws_.freeze_panes = "A2"
        for cell in ws_["F"][1:]:
            if cell.value is not None and not isinstance(cell.value, str):
                cell.number_format = "#,##0.0000"
        if note_text:
            note_r = max(r_idx, 3) + 1
            ws_.merge_cells(start_row=note_r, start_column=1,
                            end_row=note_r, end_column=len(headers))
            note = ws_.cell(row=note_r, column=1, value=note_text)
            note.font = Font(italic=True, color="8B949E")
            note.alignment = Alignment(wrap_text=True, vertical="top")
            ws_.row_dimensions[note_r].height = 50

    ws_new = wb.active
    ws_new.title = "New Tickers"
    render_grouped_sheet(
        ws_new,
        ticker_header="Yahoo Ticker",
        rows_=new_rows,
        note_text=(f"Generated {_date.today().isoformat()} from data/holding.xlsx. "
                   "Cross-check each Last Close against the Stashaway (or Yahoo) website. "
                   "Once verified, wire the ticker into "
                   "src/hailmary/allocation/universe.py (STASHAWAY_UNIVERSE) - "
                   "it will stop appearing here on the next run."),
    )

    ws_ver = wb.create_sheet("Verified Tickers")
    render_grouped_sheet(
        ws_ver,
        ticker_header="Wired Ticker",
        rows_=verified_rows,
    )

    wb.save(OUT)
    print()
    if new_refs:
        print(f"NEW: {len(new_refs)} ticker(s) need eyeballing - see 'New Tickers' sheet")
        for yt in new_refs:
            snap = cache[yt]
            print(f"  {yt:10s} {snap['currency']:4s} last={snap['last_close']:.4f} ({snap['as_of']})")
    else:
        print("NEW: none - all Holdings tickers already wired in universe.py")
    print(f"VERIFIED: {len(verified_rows)} holding rows ({len(set(r['Wired Ticker'] for r in verified_rows))} unique tickers)")
    print(f"Wrote {OUT.resolve()}")


if __name__ == "__main__":
    main()
