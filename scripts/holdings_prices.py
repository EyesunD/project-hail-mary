"""Generate a refreshable holdings-price Excel for cross-checking against the Stashaway app.

Run with `.venv/Scripts/python.exe scripts/holdings_prices.py` whenever you want a fresh
snapshot. Writes `reports/holdings_prices.xlsx` with one row per (portfolio, holding),
showing the wired ticker, Yahoo's `longName`, currency, last close, 1d / YTD return,
and history start. Use this to:

- Sanity-check ticker → fund-name pairing one final time
- Verify last close approximately matches the Stashaway app
- Spot funds with stale or missing history
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from hailmary.allocation.book_config import ROLES
from hailmary.allocation.statements import parse_statement
from hailmary.allocation.universe import STASHAWAY_UNIVERSE

warnings.filterwarnings("ignore")

PDF = "data/statements/2026-04 StashAway Monthly Statement.pdf"
LINKS = Path("data/holding links.xlsx")
OUT = Path("reports/holdings_prices.xlsx")

# ISIN-verified mappings the user has confirmed (2026-05-30). These are the
# anchor points — the diagnostic should never silently change any wiring listed
# here without re-asking. Rendered into a "Verified Mappings" sheet so the
# Excel itself carries the lock alongside universe.py.
VERIFIED_MAPPINGS: list[dict[str, str]] = [
    {"Stashaway ID": "JINAASH", "ISIN": "LU2031211308", "Yahoo Ticker": "0P0001I87K.SI",
     "Yahoo longName": "JPM Income A acc SGD Hdg", "Portfolio": "Income Investing"},
    {"Stashaway ID": "JPMGASA", "ISIN": "LU1823571978", "Yahoo Ticker": "0P0001DWBA.SI",
     "Yahoo longName": "JPM Global Bd Opps A acc SGD H", "Portfolio": "Income Investing"},
    {"Stashaway ID": "JPEMDSG", "ISIN": "LU2646069851", "Yahoo Ticker": "0P0001RG8N.SI",
     "Yahoo longName": "JPM Emerging Markets Debt A acc SGDH (hard-currency)",
     "Portfolio": "Income Investing"},
    {"Stashaway ID": "JPGCBAS", "ISIN": "LU2646069695", "Yahoo Ticker": "0P0001RG8Q.SI",
     "Yahoo longName": "JPM Global Corporate Bd A acc SGD Hdg", "Portfolio": "Income Investing"},
    {"Stashaway ID": "JPGHYHS", "ISIN": "LU2778065503", "Yahoo Ticker": "0P0001SOG4.SI",
     "Yahoo longName": "JPM Global High Yield Bd A acc SGD Hdg", "Portfolio": "Income Investing"},
    {"Stashaway ID": "BB3M", "ISIN": "IE00BMD8KM66", "Yahoo Ticker": "BB3M.L",
     "Yahoo longName": "JPM BetaBuilders US Treasury Bond 0-3 Months UCITS ETF USD Acc",
     "Portfolio": "Simple USD"},
    {"Stashaway ID": "CEUU", "ISIN": "IE00BKBF6616", "Yahoo Ticker": "CEUU.AS",
     "Yahoo longName": "iShares Core MSCI EMU UCITS ETF (USD Amsterdam listing)",
     "Portfolio": "BlackRock"},
    {"Stashaway ID": "LNWELIA", "ISIN": "(no public ISIN)", "Yahoo Ticker": "0P0001DB5Z.SI",
     "Yahoo longName": "LionGlobal SGD Enhanced Liquidity (proxy: 70% of 70/30 Simple SGD blend)",
     "Portfolio": "Simple SGD"},
    {"Stashaway ID": "OCBSGDM", "ISIN": "(no public ISIN)", "Yahoo Ticker": "0P00006FZD.SI",
     "Yahoo longName": "LionGlobal SGD Money Market A (30% leg of Guitsa/Simple SGD blend)",
     "Portfolio": "Guitsa"},
]


def load_user_links() -> pd.DataFrame:
    """Read the user's manually-maintained holdings/links file (Portfolios sheet).

    Uses openpyxl directly so we can preserve the Stashaway app hyperlinks
    attached to each Link cell. Columns are discovered by **header name**
    (header is on row 7 of the source) so the user can freely insert optional
    columns like ``ISIN`` or ``Yahoo Ticker`` without breaking the read.

    Always returns ``Port, Asset, Link, Link URL, %, Current Value`` columns;
    any of ``ISIN, Yahoo Ticker`` is added if present in the source.
    Returns empty DataFrame if the file is missing.
    """
    base_cols = ["Port", "Asset", "Link", "Link URL", "%", "Current Value"]
    optional_cols = ["ISIN", "Yahoo Ticker"]
    if not LINKS.exists():
        return pd.DataFrame(columns=base_cols)
    import openpyxl

    wb = openpyxl.load_workbook(LINKS, data_only=False)
    ws = wb["Portfolios"]
    # Header row is row 7 in the source — map each header label to its column index.
    HEADER_ROW = 7
    header_to_col: dict[str, int] = {}
    for col_idx in range(1, ws.max_column + 1):
        v = ws.cell(row=HEADER_ROW, column=col_idx).value
        if v is not None and str(v).strip():
            header_to_col[str(v).strip()] = col_idx
    # Required columns — fall back to fixed offsets if header is unusual
    required = {"Port": 6, "Asset": 9, "Link": 10, "%": 11, "Current Value": 12}
    for name, default in required.items():
        header_to_col.setdefault(name, default)

    rows: list[dict[str, object]] = []
    for r in range(HEADER_ROW + 1, ws.max_row + 1):
        port = ws.cell(row=r, column=header_to_col["Port"]).value
        asset = ws.cell(row=r, column=header_to_col["Asset"]).value
        link_cell = ws.cell(row=r, column=header_to_col["Link"])
        if port is None or asset is None or link_cell.value is None:
            continue
        link_url = link_cell.hyperlink.target if link_cell.hyperlink else ""
        row: dict[str, object] = {
            "Port": str(port),
            "Asset": str(asset),
            "Link": str(link_cell.value),
            "Link URL": str(link_url),
            "%": float(ws.cell(row=r, column=header_to_col["%"]).value or 0),
            "Current Value": ws.cell(row=r, column=header_to_col["Current Value"]).value,
        }
        for opt in optional_cols:
            if opt in header_to_col:
                v = ws.cell(row=r, column=header_to_col[opt]).value
                row[opt] = str(v).strip() if v is not None else ""
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_ticker_snapshot(ticker: str) -> dict[str, object]:
    """Pull the metadata + recent price for one ticker."""
    if ticker.startswith("CASH_"):
        return {
            "longName": f"(synthetic — {ticker})",
            "currency": "USD" if ticker == "CASH_USD" else "SGD",
            "last_close": float("nan"),
            "as_of": pd.NaT,
            "ret_1d": float("nan"),
            "ret_ytd": float("nan"),
            "history_start": pd.NaT,
        }
    tk = yf.Ticker(ticker)
    info = tk.info
    long_name = info.get("longName") or info.get("shortName") or "?"
    ccy = info.get("currency") or "?"
    ret_ytd = info.get("ytdReturn")
    try:
        h_max = tk.history(period="max", auto_adjust=True)
        h_short = tk.history(period="5d", auto_adjust=True)
        last_close = float(h_short.iloc[-1]["Close"]) if not h_short.empty else float("nan")
        as_of = h_short.index[-1].date() if not h_short.empty else pd.NaT
        prev = float(h_short.iloc[-2]["Close"]) if len(h_short) >= 2 else float("nan")
        ret_1d = (last_close / prev - 1.0) if prev and prev == prev else float("nan")
        history_start = h_max.index[0].date() if not h_max.empty else pd.NaT
    except Exception:
        last_close = float("nan")
        as_of = pd.NaT
        ret_1d = float("nan")
        history_start = pd.NaT
    return {
        "longName": long_name,
        "currency": ccy,
        "last_close": last_close,
        "as_of": as_of,
        "ret_1d": ret_1d,
        "ret_ytd": ret_ytd if ret_ytd is not None else float("nan"),
        "history_start": history_start,
    }


# Map user's portfolio-name shorthand → canonical Stashaway portfolio name.
_USER_PORTFOLIO_ALIASES = {"SRS": "General SRS"}


_URL_TICKER_RE = __import__("re").compile(r"/asset-details/([^/]+)/")


def _ticker_in_url(url: str) -> str:
    """Extract the lowercased ticker code from a Stashaway app-deep-link.

    e.g. https://app.stashaway.sg/asset-details/cspx/goal/... -> 'cspx'.
    Returns '' if no match.
    """
    m = _URL_TICKER_RE.search(url or "")
    return m.group(1).lower() if m else ""


def _match_user_label(
    portfolio_name: str,
    stashaway_id: str,
    weight: float,
    links: pd.DataFrame,
    used: set[int],
) -> tuple[str, str, str, str, float, int | None]:
    """Match a holding to a user-link row using URL-embedded ticker (primary)
    or closest weight (fallback). Greedy 1:1 — a user row used once is taken.

    Returns (label, link_url, isin, user_yahoo_ticker, user_weight, matched_row_index)
    or ("", "", "", "", nan, None) if none found.
    """
    empty = ("", "", "", "", float("nan"), None)
    if links.empty:
        return empty
    sub = links[links["Port"] == portfolio_name]
    if sub.empty:
        return empty
    candidates = sub[~sub.index.isin(used)]
    if candidates.empty:
        return empty

    def _row_payload(row: pd.Series, idx: int) -> tuple[str, str, str, str, float, int]:
        return (
            str(row["Link"]),
            str(row.get("Link URL", "") or ""),
            str(row.get("ISIN", "") or ""),
            str(row.get("Yahoo Ticker", "") or ""),
            float(row["%"]),
            int(idx),
        )

    # PRIMARY: URL-based exact match — URL contains the Stashaway ticker
    sid_lower = stashaway_id.lower()
    cash_alias = {"CASH_USD": "usd", "CASH_SGD": "sgd"}.get(stashaway_id)
    for idx, row in candidates.iterrows():
        url_ticker = _ticker_in_url(str(row.get("Link URL", "") or ""))
        if not url_ticker:
            continue
        if url_ticker == sid_lower or (cash_alias and url_ticker == cash_alias):
            return _row_payload(row, idx)

    # FALLBACK: closest weight within 5pp, only for rows without a URL
    no_url = candidates[
        candidates["Link URL"].fillna("").apply(_ticker_in_url) == ""
    ]
    if no_url.empty:
        return empty
    diffs = (no_url["%"].astype(float) - float(weight)).abs()
    idx = int(diffs.idxmin())
    if diffs.loc[idx] > 0.05:
        return empty
    return _row_payload(no_url.loc[idx], idx)


def main() -> None:
    parsed = parse_statement(PDF)
    portfolios = [p for p in parsed if p.name in ROLES]
    user_links = load_user_links()
    user_links_canonical = user_links.copy()
    user_links_canonical["Port"] = user_links_canonical["Port"].replace(_USER_PORTFOLIO_ALIASES)
    print(f"Portfolios in book: {len(portfolios)}")
    print(f"User-link rows loaded: {len(user_links)}")

    # Cache per-ticker so we don't refetch the same Yahoo data for, e.g.,
    # LNWELIA + OCBSGDM (both map to the same LionGlobal ticker).
    cache: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    seen = 0
    used_link_rows: set[int] = set()
    for p in portfolios:
        for h in p.holdings:
            user_label, user_url, user_isin, user_yticker, user_weight, matched_idx = (
                _match_user_label(
                    p.name, h.stashaway_id, h.weight, user_links_canonical, used_link_rows
                )
            )
            if matched_idx is not None:
                used_link_rows.add(matched_idx)
            meta = STASHAWAY_UNIVERSE.get(h.stashaway_id)
            if meta is None:
                rows.append(
                    {
                        "Portfolio": p.name,
                        "Stashaway ID": h.stashaway_id,
                        "ISIN": user_isin,
                        "Wired Ticker": "(unmapped)",
                        "Yahoo longName": "—",
                        "Currency": "?",
                        "Last Close": float("nan"),
                        "As Of": pd.NaT,
                        "1d %": float("nan"),
                        "YTD %": float("nan"),
                        "History Start": pd.NaT,
                        "Weight": h.weight,
                        "Value (native)": h.value,
                        "Portfolio Ccy": p.currency,
                        "User Label": user_label,
                        "User URL": user_url,
                        "User %": user_weight,
                        "User Yahoo Ticker": user_yticker,
                        "Drift?": ("DRIFT" if user_yticker else ""),
                    }
                )
                continue
            t = meta.ticker
            if t not in cache:
                cache[t] = fetch_ticker_snapshot(t)
                seen += 1
                print(f"  [{seen}] fetched {t}: {cache[t]['longName']}")
            snap = cache[t]
            drift = "DRIFT" if (user_yticker and user_yticker != t) else ""
            rows.append(
                {
                    "Portfolio": p.name,
                    "Stashaway ID": h.stashaway_id,
                    "ISIN": user_isin,
                    "Wired Ticker": t,
                    "Yahoo longName": snap["longName"],
                    "Currency": snap["currency"],
                    "Last Close": snap["last_close"],
                    "As Of": snap["as_of"],
                    "1d %": snap["ret_1d"],
                    "YTD %": snap["ret_ytd"],
                    "History Start": snap["history_start"],
                    "Weight": h.weight,
                    "Value (native)": h.value,
                    "Portfolio Ccy": p.currency,
                    "User Label": user_label,
                    "User URL": user_url,
                    "User %": user_weight,
                    "User Yahoo Ticker": user_yticker,
                    "Drift?": drift,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["Portfolio", "Weight"], ascending=[True, False]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Write with section headers per portfolio so the user can scan by sleeve.
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    headers = list(df.columns)
    wb = Workbook()
    ws = wb.active
    ws.title = "Holdings"
    header_fill = PatternFill("solid", fgColor="161B22")
    header_font = Font(color="E6EDF3", bold=True)
    portfolio_fill = PatternFill("solid", fgColor="1F2730")
    portfolio_font = Font(color="58A6FF", bold=True, size=12)
    totals_font = Font(italic=True, color="8B949E")

    # Header row
    for col_idx, name in enumerate(headers, start=1):
        c = ws.cell(row=1, column=col_idx, value=name)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="left" if col_idx <= 4 else "right", vertical="center")

    user_label_col = headers.index("User Label") + 1 if "User Label" in headers else None
    user_url_col = headers.index("User URL") + 1 if "User URL" in headers else None
    link_font = Font(color="58A6FF", underline="single")

    row = 2
    for portfolio_name, group in df.groupby("Portfolio", sort=False):
        # Section header row spanning all columns
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(headers))
        c = ws.cell(row=row, column=1,
                    value=f"  {portfolio_name}   ·   {len(group)} holdings   ·   "
                          f"{group['Portfolio Ccy'].iloc[0]} {group['Value (native)'].sum():,.0f}")
        c.fill = portfolio_fill
        c.font = portfolio_font
        c.alignment = Alignment(horizontal="left", vertical="center")
        row += 1
        # Data rows
        for _, r in group.iterrows():
            for col_idx, name in enumerate(headers, start=1):
                cell = ws.cell(row=row, column=col_idx, value=r[name])
                if (
                    col_idx == user_label_col
                    and user_url_col is not None
                    and r.get("User URL")
                ):
                    cell.hyperlink = r["User URL"]
                    cell.font = link_font
            row += 1

    # Column widths — 19 columns total
    # A Portfolio | B Stashaway ID | C ISIN | D Wired Ticker | E Yahoo longName |
    # F Currency | G Last Close | H As Of | I 1d % | J YTD % | K History Start |
    # L Weight | M Value (native) | N Portfolio Ccy | O User Label | P User URL |
    # Q User % | R User Yahoo Ticker | S Drift?
    widths = [22, 14, 16, 18, 50, 8, 12, 12, 9, 9, 14, 9, 14, 8, 24, 50, 9, 18, 12]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    if user_url_col is not None:
        ws.column_dimensions[get_column_letter(user_url_col)].hidden = True

    # Number formats by column letter
    pct_cols = {"I": "+0.00%;-0.00%;0.00%",
                "J": "+0.00%;-0.00%;0.00%",
                "L": "0.00%",
                "Q": "0.00%"}
    num_cols = {"G": "#,##0.00", "M": "#,##0.00"}
    date_cols = {"H": "yyyy-mm-dd", "K": "yyyy-mm-dd"}
    for col, fmt in {**pct_cols, **num_cols, **date_cols}.items():
        for cell in ws[col][1:]:
            if cell.value is not None and not isinstance(cell.value, str):
                cell.number_format = fmt

    # Tint Drift? cells red when populated
    drift_col_idx = headers.index("Drift?") + 1 if "Drift?" in headers else None
    if drift_col_idx is not None:
        warn_font = Font(color="F85149", bold=True)
        for cell in ws[get_column_letter(drift_col_idx)][1:]:
            if cell.value == "DRIFT":
                cell.font = warn_font
                cell.fill = PatternFill("solid", fgColor="3D1F1F")

    # ----- Sheet 3: raw user-link rows + last-close for any YT they populated
    ws3 = wb.create_sheet("User Holdings (links)")
    if not user_links_canonical.empty:
        link_cols = list(user_links_canonical.columns) + ["Last Close", "Currency", "As Of"]
        for col_idx, name in enumerate(link_cols, start=1):
            c = ws3.cell(row=1, column=col_idx, value=name)
            c.fill = header_fill
            c.font = header_font
        link_text_col = link_cols.index("Link") + 1 if "Link" in link_cols else None
        link_url_col = link_cols.index("Link URL") + 1 if "Link URL" in link_cols else None
        yt_col_idx = link_cols.index("Yahoo Ticker") + 1 if "Yahoo Ticker" in link_cols else None
        for r_idx, (_, r) in enumerate(user_links_canonical.iterrows(), start=2):
            for col_idx, name in enumerate(link_cols, start=1):
                if name in ("Last Close", "Currency", "As Of"):
                    continue
                cell = ws3.cell(row=r_idx, column=col_idx, value=r[name])
                if (
                    col_idx == link_text_col
                    and link_url_col is not None
                    and r.get("Link URL")
                ):
                    cell.hyperlink = r["Link URL"]
                    cell.font = link_font
            # Look up price for the YT the user entered (cache hits if already
            # fetched for the main Holdings sheet)
            yt = str(r.get("Yahoo Ticker") or "").strip()
            last_close = float("nan")
            ccy = ""
            as_of: object = pd.NaT
            if yt:
                if yt not in cache:
                    try:
                        cache[yt] = fetch_ticker_snapshot(yt)
                    except Exception:
                        cache[yt] = {"last_close": float("nan"), "currency": "?", "as_of": pd.NaT}
                snap = cache[yt]
                last_close = snap.get("last_close", float("nan"))
                ccy = snap.get("currency", "?")
                as_of = snap.get("as_of", pd.NaT)
            ws3.cell(row=r_idx, column=link_cols.index("Last Close") + 1, value=last_close)
            ws3.cell(row=r_idx, column=link_cols.index("Currency") + 1, value=ccy)
            ws3.cell(row=r_idx, column=link_cols.index("As Of") + 1, value=as_of)
        ws3.freeze_panes = "A2"
        col_widths_3 = [22, 36, 18, 18, 28, 50, 9, 14, 12, 8, 12]
        for i, w in enumerate(col_widths_3[: len(link_cols)], start=1):
            ws3.column_dimensions[get_column_letter(i)].width = w
        # Hide raw URL column on Sheet 3 too
        if link_url_col is not None:
            ws3.column_dimensions[get_column_letter(link_url_col)].hidden = True
        # Format %
        pct_col_idx = link_cols.index("%") + 1 if "%" in link_cols else None
        if pct_col_idx is not None:
            for cell in ws3[get_column_letter(pct_col_idx)][1:]:
                if cell.value is not None and not isinstance(cell.value, str):
                    cell.number_format = "0.00%"
        # Format Last Close + As Of
        lc_col = link_cols.index("Last Close") + 1
        for cell in ws3[get_column_letter(lc_col)][1:]:
            if cell.value is not None and not isinstance(cell.value, str):
                cell.number_format = "#,##0.0000"
        ao_col = link_cols.index("As Of") + 1
        for cell in ws3[get_column_letter(ao_col)][1:]:
            if cell.value is not None and not isinstance(cell.value, str):
                cell.number_format = "yyyy-mm-dd"

    # ----- Sheet 4: Verified Mappings — pulled primarily from the user's
    # holding-links file (whichever rows have ISIN + Yahoo Ticker populated).
    # Falls back to the hardcoded VERIFIED_MAPPINGS for rows the user hasn't
    # populated yet. Each row carries the live universe.py wiring + drift flag.
    ws4 = wb.create_sheet("Verified Mappings")
    headers4 = ["Source", "Stashaway ID (best)", "ISIN", "Yahoo Ticker",
                "Yahoo longName", "Portfolio",
                "Wired in universe.py", "Drift?"]
    for col_idx, name in enumerate(headers4, start=1):
        c = ws4.cell(row=1, column=col_idx, value=name)
        c.fill = header_fill
        c.font = header_font
    warn_font = Font(color="F85149", bold=True)
    ok_font = Font(color="3FB950")
    soft_font = Font(color="8B949E", italic=True)

    # 1) Pull from user file when ISIN AND Yahoo Ticker columns are present
    user_verified: list[dict[str, object]] = []
    if not user_links_canonical.empty and {"ISIN", "Yahoo Ticker"}.issubset(user_links_canonical.columns):
        for _, lr in user_links_canonical.iterrows():
            isin = str(lr.get("ISIN") or "").strip()
            yticker = str(lr.get("Yahoo Ticker") or "").strip()
            if not isin and not yticker:
                continue
            sid = _ticker_in_url(str(lr.get("Link URL") or "")).upper() or lr.get("Link")
            user_verified.append({
                "Source": "user file",
                "Stashaway ID (best)": str(sid),
                "ISIN": isin,
                "Yahoo Ticker": yticker,
                "Yahoo longName": "",
                "Portfolio": str(lr.get("Port") or ""),
            })

    # 2) Add hardcoded entries for Stashaway IDs not already in the user-verified set
    user_sids = {str(r["Stashaway ID (best)"]).upper() for r in user_verified}
    for m in VERIFIED_MAPPINGS:
        if m["Stashaway ID"].upper() in user_sids:
            continue
        user_verified.append({
            "Source": "script default",
            "Stashaway ID (best)": m["Stashaway ID"],
            "ISIN": m["ISIN"],
            "Yahoo Ticker": m["Yahoo Ticker"],
            "Yahoo longName": m["Yahoo longName"],
            "Portfolio": m["Portfolio"],
        })

    drift_count = 0
    for r_idx, m in enumerate(user_verified, start=2):
        sid = str(m["Stashaway ID (best)"])
        wired = STASHAWAY_UNIVERSE.get(sid)
        in_universe = wired is not None
        wired_ticker = wired.ticker if in_universe else "(outside Stashaway book)"
        # Drift only counts when SID is actually in universe.py AND user-yticker disagrees
        drift = in_universe and bool(m["Yahoo Ticker"]) and wired_ticker != m["Yahoo Ticker"]
        if drift:
            drift_count += 1
        for col_idx, name in enumerate(headers4, start=1):
            if name == "Wired in universe.py":
                value = wired_ticker
            elif name == "Drift?":
                if drift:
                    value = "DRIFT — RE-VERIFY"
                elif not in_universe:
                    value = "outside book"
                elif m["Yahoo Ticker"]:
                    value = "ok"
                else:
                    value = "—"
            else:
                value = m.get(name, "")
            cell = ws4.cell(row=r_idx, column=col_idx, value=value)
            if name == "Drift?":
                if drift:
                    cell.font = warn_font
                elif in_universe and m["Yahoo Ticker"]:
                    cell.font = ok_font
                else:
                    cell.font = soft_font
            if name == "Source" and m.get("Source") == "script default":
                cell.font = soft_font
    ws4.freeze_panes = "A2"
    widths4 = [14, 22, 18, 18, 55, 20, 20, 22]
    for i, w in enumerate(widths4, start=1):
        ws4.column_dimensions[get_column_letter(i)].width = w
    # Lead with a short note
    note_row = len(user_verified) + 3
    ws4.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=len(headers4))
    note = ws4.cell(row=note_row, column=1,
                    value=("'user file' rows are pulled from data/holding links.xlsx — add ISIN + "
                           "Yahoo Ticker columns to that file to mark a row as verified. "
                           "'script default' rows are baked-in fallbacks for mappings the user "
                           "hasn't recorded yet. Any DRIFT means universe.py disagrees with the "
                           "user file — re-verify via yfinance.Search(<ISIN>) before changing. "
                           "See memory/feedback_ticker_verification.md."))
    note.font = Font(italic=True, color="8B949E")
    note.alignment = Alignment(wrap_text=True, vertical="top")
    ws4.row_dimensions[note_row].height = 60

    wb.save(OUT)
    matched = sum(1 for r in rows if r.get("User Label"))
    unmatched_user = len(user_links_canonical) - len(used_link_rows)
    print()
    print(f"User-link match: {matched}/{len(rows)} holdings matched a user-link row")
    print(f"  {unmatched_user} user-link rows had no Stashaway holding (extra sleeves you track elsewhere)")
    if drift_count:
        print(f"  ⚠ {drift_count} verified mapping(s) DRIFTED from universe.py — see 'Verified Mappings' sheet")
    else:
        print(f"  {len(VERIFIED_MAPPINGS)} verified mappings match universe.py — no drift")

    # Also emit a deduplicated ticker sheet for one-glance verification
    unique = df.drop_duplicates("Wired Ticker")[
        ["Wired Ticker", "Yahoo longName", "Currency", "Last Close", "As Of", "1d %", "YTD %", "History Start"]
    ].copy()
    with pd.ExcelWriter(OUT, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        unique.to_excel(writer, index=False, sheet_name="Unique Tickers")
        ws2 = writer.sheets["Unique Tickers"]
        widths2 = {"A": 18, "B": 50, "C": 10, "D": 12, "E": 12, "F": 9, "G": 9, "H": 14}
        for col, w in widths2.items():
            ws2.column_dimensions[col].width = w
        ws2.freeze_panes = "A2"
        for cell in ws2["D"][1:]:
            cell.number_format = "#,##0.00"
        for cell in ws2["F"][1:]:
            cell.number_format = "+0.00%;-0.00%;0.00%"
        for cell in ws2["G"][1:]:
            cell.number_format = "+0.00%;-0.00%;0.00%"

    today = date.today().isoformat()
    print()
    print(f"Wrote {OUT.resolve()}")
    print(f"  {len(df)} holding rows across {df['Portfolio'].nunique()} portfolios")
    print(f"  {len(unique)} unique tickers")
    print(f"  Refresh by re-running this script ({today} snapshot).")


if __name__ == "__main__":
    main()
