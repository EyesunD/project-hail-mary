"""Smoke check: load_book(holding.xlsx) vs parse_statement(latest PDF).

PDF gives statement-date actuals (with drifted cash). Sheet gives target
weights. They will not match exactly — that's the whole point. This script
verifies the WEAKER conditions that should hold:

- Same portfolio set (sheet may have NEW sleeves not in PDF — that's fine).
- Same ticker set per portfolio, modulo zero-target cash rows omitted by the
  sheet target (PDF retains them as drifted residuals).
- All weight rows sum to 1.0 (already enforced by Portfolio.__post_init__).
- Per-ticker drift between PDF actual and sheet target stays inside a sane
  band (default 5% absolute); larger drifts mean stale targets or bad data.

Run: ``.venv/Scripts/python.exe scripts/parity_load_book.py``
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from hailmary.allocation.book_config import MGMT_FEES_ANNUAL, ROLES
from hailmary.allocation.holdings_book import load_book
from hailmary.allocation.portfolios import from_parsed
from hailmary.allocation.statements import parse_statement

PDF = Path("data/statements/2026-04 StashAway Monthly Statement.pdf")
AS_OF = date(2026, 4, 30)
DRIFT_BAND = 0.05  # 5% absolute per ticker — anything bigger needs eyeballing


def _pdf_book() -> dict[str, dict[str, float]]:
    parsed = parse_statement(PDF, use_cache=False)
    out: dict[str, dict[str, float]] = {}
    for p in parsed:
        if p.name not in ROLES:
            continue
        port = from_parsed(p, roles=ROLES[p.name])
        out[port.name] = {h.metadata.ticker: h.weight for h in port.holdings}
    return out


def _sheet_book() -> tuple[dict[str, dict[str, float]], dict[str, tuple[str, float, float]]]:
    book = load_book(as_of=AS_OF)
    weights = {p.name: {h.metadata.ticker: h.weight for h in p.holdings} for p in book}
    meta = {
        p.name: (
            p.currency,
            p.total_value,
            float(p.metadata.get("statement_fx_usd_sgd", 0.0)),
        )
        for p in book
    }
    return weights, meta


def main() -> int:
    pdf = _pdf_book()
    sheet, meta = _sheet_book()

    pdf_names = set(pdf)
    sheet_names = set(sheet)
    common = pdf_names & sheet_names
    only_sheet = sheet_names - pdf_names
    only_pdf = pdf_names - sheet_names

    print(f"PDF portfolios:   {len(pdf_names)}")
    print(f"Sheet portfolios: {len(sheet_names)}")
    if only_sheet:
        print(f"  NEW (sheet only): {sorted(only_sheet)}")
    if only_pdf:
        print(f"  MISSING from sheet: {sorted(only_pdf)}  <- needs investigation")
    print()

    big_drifts = 0
    cash_drops = 0
    for name in sorted(common):
        p_w = pdf[name]
        s_w = sheet[name]
        p_t = set(p_w)
        s_t = set(s_w)

        only_p_cash = {t for t in p_t - s_t if t.startswith("CASH_")}
        only_p_real = (p_t - s_t) - only_p_cash
        only_s = s_t - p_t
        cash_drops += len(only_p_cash)

        drifts = []
        for t in p_t & s_t:
            d = s_w[t] - p_w[t]
            if abs(d) > DRIFT_BAND:
                drifts.append((t, p_w[t], s_w[t], d))

        s_sum = sum(s_w.values())
        flag = "OK"
        notes = []
        if only_p_real:
            flag = "WARN"
            notes.append(f"non-cash missing from sheet: {sorted(only_p_real)}")
        if only_s:
            flag = "WARN"
            notes.append(f"ticker only in sheet: {sorted(only_s)}")
        if drifts:
            flag = "DRIFT"
            big_drifts += len(drifts)
        if abs(s_sum - 1.0) > 1e-3:
            flag = "WARN"
            notes.append(f"sheet weight sum {s_sum:.4f} != 1.0")

        print(f"[{flag:5s}] {name:30s} pdf={len(p_t):2d}  sheet={len(s_t):2d}  cash-dropped={len(only_p_cash)}")
        for n in notes:
            print(f"          {n}")
        for t, a, b, d in drifts:
            print(f"          drift>{DRIFT_BAND:.0%}: {t} pdf={a:.4f} sheet={b:.4f} delta={d:+.4f}")

    print()
    print("--- Currency / FX (sheet) ---")
    for name in sorted(meta):
        cur, val, fx = meta[name]
        fx_s = f"fx={fx:.4f}" if fx else "fx=n/a (SGD-base)"
        print(f"  {name:30s} {cur} total={val:>14,.2f}  {fx_s}")

    print()
    print(f"Summary: {len(common)} common portfolios, {cash_drops} zero-target cash rows "
          f"dropped by sheet (expected), {big_drifts} per-ticker drifts > {DRIFT_BAND:.0%} (review if non-zero).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
