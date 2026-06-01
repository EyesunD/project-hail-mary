"""Reconcile report — persistent monthly trust-check artifact.

Pulls together everything we proved in the 2026-05-31 validation session into
a single self-contained HTML the user can regenerate whenever something
changes (new deposit, target update, fresh stmt PDF). Sections:

1. Per-sleeve table — App SGD | Our model | + Deposits | = Total | Gap | Status
2. Deposits log — chronological audit trail from holding-links Deposits sheet
3. Per-sleeve drill-down — holdings reconciliation (reuses diagnostic helpers)
4. Methodology footnote — what's used where, briefly

The user re-runs `render_reconcile_report(...)` after any change. If the gap
column lights red, something's off and we investigate.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from hailmary.allocation.portfolios import Holding, Portfolio, Role

_PORTFOLIO_ALIASES = {"SRS": "General SRS"}


# ---------------------------------------------------------------------------
# Deposits loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Deposit:
    """One row from the Deposits sheet — post-stmt cash flow into/out of a sleeve."""

    date: date
    portfolio: str
    currency: str
    amount: float
    notes: str = ""


def load_deposits(
    links_path: Path = Path("data/holding.xlsx"),
    *,
    since: date | None = None,
) -> list[Deposit]:
    """Read the Deposits sheet. Returns rows newer than ``since`` (typically
    the statement date) — anything dated on/before ``since`` is already in
    the statement's ``total_value`` and would double-count if re-added.
    """
    if not links_path.exists():
        return []
    try:
        import openpyxl
    except ImportError:
        return []
    out: list[Deposit] = []
    try:
        wb = openpyxl.load_workbook(links_path, data_only=False)
        if "Deposits" not in wb.sheetnames:
            return []
        ws = wb["Deposits"]
        for r in range(2, ws.max_row + 1):
            d = ws.cell(row=r, column=1).value
            p = ws.cell(row=r, column=2).value
            c = ws.cell(row=r, column=3).value
            a = ws.cell(row=r, column=4).value
            n = ws.cell(row=r, column=5).value
            if d is None or p is None or a is None:
                continue
            d_val = d.date() if hasattr(d, "date") else d
            if since is not None and d_val <= since:
                continue
            out.append(
                Deposit(
                    date=d_val,
                    portfolio=_PORTFOLIO_ALIASES.get(str(p), str(p)),
                    currency=str(c).upper() if c else "SGD",
                    amount=float(a),
                    notes=str(n) if n else "",
                )
            )
    except Exception as exc:
        warnings.warn(f"Could not read Deposits sheet: {exc}", stacklevel=2)
    return out


# ---------------------------------------------------------------------------
# App-values loader (from the Stashaway "current values" PDF the user saves)
# ---------------------------------------------------------------------------


_APP_VALUE_RE = re.compile(r"^\$([\d,]+\.\d{2})$")
_APP_DELTA_RE = re.compile(r"([+-]?\$[\d,]+\.\d{2})")


def load_app_values_all_snapshots(
    links_path: Path = Path("data/holding.xlsx"),
) -> dict[date, dict[str, float]]:
    """Read every app-value snapshot from the PortfolioValueSGD sheet.

    Returns ``{snapshot_date: {portfolio_name: sgd_value}}``. When two columns
    share the same date, values are merged (rightmost wins for ties).
    Empty / missing values are skipped (e.g. new sleeves with 0 on Apr 30
    are kept; truly empty cells are skipped).
    """
    if not links_path.exists():
        return {}
    try:
        import openpyxl
    except ImportError:
        return {}
    try:
        wb = openpyxl.load_workbook(links_path, data_only=False)
        if "PortfolioValueSGD" not in wb.sheetnames:
            return {}
        ws = wb["PortfolioValueSGD"]
        HEADER_ROW = 6
        name_col = None
        date_cols: list[tuple[date, int]] = []
        for c in range(1, ws.max_column + 1):
            v = ws.cell(row=HEADER_ROW, column=c).value
            if v is None:
                continue
            if hasattr(v, "date"):
                date_cols.append((v.date(), c))
            elif isinstance(v, str) and v.strip() not in ("Id",):
                if name_col is None:
                    name_col = c
        if not date_cols or name_col is None:
            return {}
        out: dict[date, dict[str, float]] = {}
        for snap_date, col in date_cols:
            snap = out.setdefault(snap_date, {})
            for r in range(HEADER_ROW + 1, ws.max_row + 1):
                name = ws.cell(row=r, column=name_col).value
                if name is None or not str(name).strip():
                    continue
                v = ws.cell(row=r, column=col).value
                if v is None or v == "":
                    continue
                try:
                    snap[_PORTFOLIO_ALIASES.get(str(name).strip(), str(name).strip())] = float(v)
                except (TypeError, ValueError):
                    continue
        return out
    except Exception as exc:
        warnings.warn(f"Could not read PortfolioValueSGD sheet: {exc}", stacklevel=2)
        return {}


def load_app_values_at_date(
    target_date: date,
    links_path: Path = Path("data/holding.xlsx"),
) -> tuple[dict[str, float], date | None]:
    """Get the snapshot whose date is CLOSEST to ``target_date``.

    Used to source the start-of-period (stmt date) and end-of-period (today)
    values for the reconcile table. Falls back to closest if exact date
    missing.
    """
    snapshots = load_app_values_all_snapshots(links_path)
    if not snapshots:
        return {}, None
    # Closest by absolute days
    best_date = min(snapshots.keys(), key=lambda d: abs((d - target_date).days))
    return snapshots[best_date], best_date


def load_app_values_from_sheet(
    links_path: Path = Path("data/holding.xlsx"),
) -> tuple[dict[str, float], date | None]:
    """Read the LATEST app-value snapshot. Kept for backward compatibility;
    new code should prefer :func:`load_app_values_at_date` or
    :func:`load_app_values_all_snapshots`."""
    snapshots = load_app_values_all_snapshots(links_path)
    if not snapshots:
        return {}, None
    latest_date = max(snapshots.keys())
    return snapshots[latest_date], latest_date


def load_app_values(
    pdf_dir: Path = Path("data"),
    pattern: str = "current values *.pdf",
    links_path: Path = Path("data/holding.xlsx"),
) -> tuple[dict[str, float], date | None]:
    """Resolve app SGD values per sleeve.

    Primary source: ``PortfolioValueSGD`` sheet in holding-links file
    (user-maintained, time-series of snapshots). Fallback: parse the latest
    ``current values YYYY-MM-DD.pdf`` in ``data/``.
    """
    # Prefer the sheet (more reliable than PDF text-extraction)
    vals, dt = load_app_values_from_sheet(links_path)
    if vals:
        return vals, dt
    # PDF fallback below
    return _load_app_values_from_pdf(pdf_dir, pattern)


def _load_app_values_from_pdf(
    pdf_dir: Path = Path("data"),
    pattern: str = "current values *.pdf",
) -> tuple[dict[str, float], date | None]:
    """Parse the latest "current values YYYY-MM-DD.pdf" from Stashaway.

    Returns ``({portfolio_name: app_sgd_value}, snapshot_date)``. Snapshot
    date is taken from the filename. If no PDF found, returns empty +
    None — caller renders the report with "—" for app values.
    """
    candidates = sorted(pdf_dir.glob(pattern), reverse=True)
    if not candidates:
        return {}, None
    path = candidates[0]
    try:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
        snap_date = datetime.strptime(m.group(1), "%Y-%m-%d").date() if m else None
    except Exception:
        snap_date = None

    try:
        import pdfplumber
    except ImportError:
        return {}, snap_date
    try:
        with pdfplumber.open(path) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception as exc:
        warnings.warn(f"Could not parse {path}: {exc}", stacklevel=2)
        return {}, snap_date

    out: dict[str, float] = {}
    lines = [ln.strip() for ln in text.splitlines()]
    # Stashaway's per-sleeve format collapses to ~3 lines after pdfplumber's
    # text extraction:
    #   N    : "<Sleeve name>"
    #   N+1  : "Risk Level Current value (SGD)" (or "Underlying currency Current value (SGD)")
    #   N+2  : "<risk-or-currency-value> $X,XXX.XX +/-$X,XXX.XX"
    # The "+/-$X,XXX.XX" is the lifetime return; we want the FIRST $ value
    # on that line — the current SGD value.
    skip_labels = {
        "Public Market Investments", "Cash Management", "Powered by StashAway",
        "Powered by BlackRock®", "All figures are in SGD",
        "Net deposits (SGD)", "Total value (SGD)", "Total returns after fees (SGD)",
        "Underlying currency", "Risk Level",
        "Risk Level Current value (SGD)", "Underlying currency Current value (SGD)",
        "StashAway Risk Index", "StashAway Risk Index Current value (SGD)",
        "Contact my Wealth Advisor",
    }
    risk_or_ccy_values = {
        "Conservative", "Aggressive", "Very aggressive", "Very conservative",
        "SGD", "USD",
    }
    dollar_re = re.compile(r"\$([\d,]+\.\d{2})")
    greeting_re = re.compile(r"^(Good (morning|afternoon|evening)|Hi|Hello)\b", re.IGNORECASE)
    for i, line in enumerate(lines):
        if not line or line in skip_labels or line in risk_or_ccy_values:
            continue
        # Skip any line that itself contains a dollar value, a date, a URL,
        # or starts with a greeting / boilerplate — these are not portfolio
        # names even if they didn't make the skip_labels list.
        if dollar_re.search(line):
            continue
        if line.startswith(("$", "+$", "-$", "Excluding ", "Powered ", "https://", "5/", "Have a ")):
            continue
        if greeting_re.match(line):
            continue
        if "Current value" in line:
            continue
        if line.startswith("Janelle"):
            continue
        # Reject pure number/percent lines (e.g. "36%" = StashAway risk index value)
        if re.fullmatch(r"[\d.]+%?", line):
            continue
        # Plausibly a portfolio name. Peek 1-3 lines ahead for the value line.
        for offset in (1, 2, 3):
            if i + offset >= len(lines):
                break
            nxt = lines[i + offset]
            if "Current value" in nxt and not dollar_re.search(nxt):
                continue  # label line — keep looking
            ms = dollar_re.findall(nxt)
            if not ms:
                continue
            val = float(ms[0].replace(",", ""))
            # Strip trailing badge tokens Stashaway appends inline ("SRS").
            clean_name = re.sub(r"\s+(SRS|JOINT|INDIVIDUAL)$", "", line).strip()
            out[_PORTFOLIO_ALIASES.get(clean_name, clean_name)] = val
            break
    return out, snap_date


# ---------------------------------------------------------------------------
# Reconcile data assembly
# ---------------------------------------------------------------------------


def _synthesize_new_sleeves(
    existing_portfolios: Sequence[Portfolio],
    deposits: Sequence[Deposit],
    app_values_end: dict[str, float],
    target_weights_by_portfolio: dict[str, list[tuple[date, dict[str, float]]]] | None,
    fx_series_usd_sgd: pd.Series,
) -> tuple[list[Portfolio], list[Deposit]]:
    """Synthesize Portfolio objects for sleeves added AFTER the statement date.

    A sleeve qualifies when it appears in the ``PortfolioValueSGD`` sheet
    (``app_values_end``) but isn't in the parsed statement. We treat the first
    deposit to that sleeve as a synthetic statement event:

      - ``statement_date = first_deposit.date``
      - ``total_value = first_deposit.amount`` (in deposit currency)
      - ``holdings`` synthesized from the target snapshot effective on or
        nearest the deposit date
      - ``currency = first_deposit.currency``

    The seed deposit is removed from the returned ``residual_deposits`` so the
    "+ Deposits" column doesn't double-count it (it's already the model's basis).
    """
    from hailmary.allocation.diagnostic import _target_weights_at
    from hailmary.allocation.universe import STASHAWAY_UNIVERSE as UNIV

    existing_names = {p.name for p in existing_portfolios}
    targets = target_weights_by_portfolio or {}

    deposits_by_port: dict[str, list[Deposit]] = {}
    for d in deposits:
        deposits_by_port.setdefault(d.portfolio, []).append(d)
    for name in deposits_by_port:
        deposits_by_port[name].sort(key=lambda d: d.date)

    new_portfolios: list[Portfolio] = []
    seed_keys: set[tuple[str, date, float]] = set()

    fx_clean = fx_series_usd_sgd.copy()
    if hasattr(fx_clean.index, "tz") and fx_clean.index.tz is not None:
        fx_clean.index = fx_clean.index.tz_localize(None)

    for name, deps in deposits_by_port.items():
        if name in existing_names:
            continue
        if name not in app_values_end:
            continue  # No app value to reconcile against
        seed = deps[0]
        port_targets = targets.get(name)
        if not port_targets:
            warnings.warn(
                f"New sleeve {name!r} has a deposit on {seed.date} but no target "
                "weights — can't synthesize a stmt basis. Skipping.",
                stacklevel=2,
            )
            continue
        # Pick target effective on or before seed date; if none, use the
        # earliest available snapshot (deps are sorted desc, so [-1] = earliest).
        weights = _target_weights_at(port_targets, seed.date)
        if weights is None:
            weights = port_targets[-1][1]

        holdings: list[Holding] = []
        skipped_sids: list[str] = []
        for sid, w in weights.items():
            if w <= 0:
                continue
            meta = UNIV.get(sid)
            if meta is None:
                skipped_sids.append(sid)
                continue
            holdings.append(
                Holding(
                    ticker=sid,
                    weight=w,
                    value=w * seed.amount,
                    metadata=meta,
                )
            )
        wsum = sum(h.weight for h in holdings)
        if not holdings or wsum <= 0:
            warnings.warn(
                f"New sleeve {name!r}: target sids {skipped_sids} not in "
                "STASHAWAY_UNIVERSE — can't synthesize. Add them to "
                "universe.py or skip the sleeve. Skipping for now.",
                stacklevel=2,
            )
            continue
        # Partial drop — synthesis can still proceed but the model will track
        # a different basket than what the user actually holds. Warn loudly
        # with the missing sids and the weight they accounted for so the user
        # knows to add them to universe.py rather than silently renormalising
        # away the gap.
        if skipped_sids:
            dropped_weight = sum(
                w for sid, w in weights.items() if sid in skipped_sids
            )
            warnings.warn(
                f"New sleeve {name!r}: dropped {skipped_sids} "
                f"({dropped_weight:.2%} of target weight) — not in "
                "STASHAWAY_UNIVERSE. Remaining weights will be renormalised. "
                "Add the missing tickers to universe.py for an accurate model.",
                stacklevel=2,
            )
        if abs(wsum - 1.0) > 1e-4:
            holdings = [
                Holding(
                    ticker=h.ticker,
                    weight=h.weight / wsum,
                    value=(h.weight / wsum) * seed.amount,
                    metadata=h.metadata,
                )
                for h in holdings
            ]

        currency = seed.currency.upper()
        metadata: dict[str, Any] = {}
        if currency == "USD":
            stmt_fx_value = float(fx_clean.asof(pd.Timestamp(seed.date)))
            if stmt_fx_value == stmt_fx_value:  # not NaN
                metadata["statement_fx_usd_sgd"] = stmt_fx_value

        new_portfolios.append(
            Portfolio(
                name=name,
                statement_date=seed.date,
                total_value=seed.amount,
                currency=currency,
                holdings=holdings,
                roles={Role.HOLDING},
                metadata=metadata,
            )
        )
        seed_keys.add((seed.portfolio, seed.date, seed.amount))

    residual_deposits = [
        d for d in deposits
        if (d.portfolio, d.date, d.amount) not in seed_keys
    ]
    return new_portfolios, residual_deposits


@dataclass
class SleeveReconcile:
    portfolio: str
    currency: str
    app_start_sgd: float | None  # App snapshot near stmt_date (start of period)
    app_end_sgd: float | None    # App snapshot near end_date (today)
    our_model_sgd: float          # Piecewise-compounded sleeve value (validation)
    deposits_sgd: float
    total_sgd: float              # our_model + deposits
    gap_sgd: float | None         # total - app_end (model accuracy)
    gap_pct: float | None
    status: str  # "exact" | "ok" | "noise" | "warn" | "no app value"


def _classify_status(gap_pct: float | None) -> str:
    if gap_pct is None:
        return "no app value"
    abs_pct = abs(gap_pct)
    if abs_pct < 0.05:
        return "exact"
    if abs_pct < 0.3:
        return "ok"
    if abs_pct < 1.0:
        return "noise"
    return "warn"


def build_reconcile(
    portfolios: Sequence[Portfolio],
    *,
    end: date,
    price_source: Any,
    fx_series_usd_sgd: pd.Series,
    target_weights_by_portfolio: dict[str, list[tuple[date, dict[str, float]]]] | None = None,
    deposits: Sequence[Deposit] | None = None,
    app_values_start: dict[str, float] | None = None,
    app_values_end: dict[str, float] | None = None,
) -> list[SleeveReconcile]:
    """Per-sleeve reconciliation.

    Performance lens: ``Δ = app_end - app_start - deposits`` — what really
    happened to your money, sourced entirely from the PortfolioValueSGD
    sheet (start = snapshot closest to stmt_date, end = closest to today).

    Validation lens: ``gap = our_model + deposits - app_end`` — does our
    piecewise-compounded model agree with the app within tolerance? Status
    badges classify the gap by % of app_end.
    """
    from hailmary.allocation.diagnostic import portfolio_reconciliation

    fx_today = float(fx_series_usd_sgd.dropna().iloc[-1])
    app_start = app_values_start or {}
    app_end = app_values_end or {}

    # Synthesize new sleeves (added after stmt) from their first deposit
    # + target weights. The seed deposit becomes the model's stmt basis;
    # it's removed from the residual list so "+ Deposits" doesn't
    # double-count the seeding.
    new_ports, residual_deps = _synthesize_new_sleeves(
        list(portfolios),
        list(deposits or []),
        app_end,
        target_weights_by_portfolio,
        fx_series_usd_sgd,
    )
    all_portfolios = list(portfolios) + new_ports

    deposits_by_port: dict[str, float] = {}
    for d in residual_deps:
        amt_sgd = d.amount if d.currency == "SGD" else d.amount * fx_today
        deposits_by_port[d.portfolio] = deposits_by_port.get(d.portfolio, 0.0) + amt_sgd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recon = portfolio_reconciliation(
            all_portfolios,
            end=end,
            price_source=price_source,
            fx_series_usd_sgd=fx_series_usd_sgd,
            target_weights_by_portfolio=target_weights_by_portfolio,
        )

    rows: list[SleeveReconcile] = []
    for _, r in recon.iterrows():
        name = r["portfolio"]
        our = float(r["today_sgd"])
        dep = deposits_by_port.get(name, 0.0)
        total = our + dep
        end_v = app_end.get(name)
        start_v = app_start.get(name)
        if end_v is not None:
            gap = total - end_v
            gap_pct = gap / end_v * 100.0 if end_v else None
        else:
            gap = None
            gap_pct = None
        rows.append(
            SleeveReconcile(
                portfolio=name,
                currency=str(r["currency"]),
                app_start_sgd=start_v,
                app_end_sgd=end_v,
                our_model_sgd=our,
                deposits_sgd=dep,
                total_sgd=total,
                gap_sgd=gap,
                gap_pct=gap_pct,
                status=_classify_status(gap_pct),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_RECONCILE_TEMPLATE_PATH = Path(__file__).parent / "_reconcile_report_template.html.j2"


_STATUS_BADGES = {
    "exact": ('<span class="badge ok">exact</span>', "#3fb950"),
    "ok": ('<span class="badge ok">ok</span>', "#3fb950"),
    "noise": ('<span class="badge noise">noise</span>', "#d29922"),
    "warn": ('<span class="badge warn">⚠ investigate</span>', "#f85149"),
    "no app value": ('<span class="badge muted">no app value</span>', "#8b949e"),
}


def _fmt_money(v: float | None) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    sign = "-" if v < 0 else ""
    av = abs(v)
    return f"{sign}${av:,.2f}"


def _fmt_money_signed(v: float | None) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:+,.2f}"


def _fmt_pct(v: float | None) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:+.2f}%"


def _reconcile_table_html(
    rows: list[SleeveReconcile],
    *,
    start_date: date | None,
    end_date: date | None,
) -> str:
    """Per-sleeve reconciliation table.

    Two-column-group structure:
      Performance (what really happened): App start → App end, deposits netted
      Validation (model accuracy):       Our model + deposits vs App end = gap
    """
    start_lbl = start_date.isoformat() if start_date else "start"
    end_lbl = end_date.isoformat() if end_date else "end"
    out: list[str] = ['<table class="ds-table reconcile-table"><thead><tr>']
    headers = [
        "Sleeve", "Ccy",
        f"App ({start_lbl})", f"App ({end_lbl})", "+ Deposits", "Δ Perf $", "Δ Perf %",
        "Our Model", "Gap $", "Gap %", "Status",
    ]
    for h in headers:
        out.append(f"<th>{h}</th>")
    out.append("</tr></thead><tbody>")
    total_start = total_end = total_our = total_dep = total_gap = total_perf = 0.0
    n_app = 0
    for r in rows:
        badge, _ = _STATUS_BADGES.get(r.status, _STATUS_BADGES["no app value"])
        start_v = r.app_start_sgd or 0.0
        perf = (
            (r.app_end_sgd - start_v - r.deposits_sgd)
            if (r.app_end_sgd is not None and r.app_start_sgd is not None)
            else None
        )
        perf_pct = (perf / start_v * 100.0) if (perf is not None and start_v > 0) else None
        out.append("<tr>")
        out.append(f"<td>{r.portfolio}</td>")
        out.append(f"<td>{r.currency}</td>")
        out.append(f"<td>{_fmt_money(r.app_start_sgd)}</td>")
        out.append(f"<td>{_fmt_money(r.app_end_sgd)}</td>")
        out.append(f"<td>{_fmt_money_signed(r.deposits_sgd)}</td>")
        out.append(f"<td>{_fmt_money_signed(perf)}</td>")
        out.append(f"<td>{_fmt_pct(perf_pct)}</td>")
        out.append(f"<td>{_fmt_money(r.our_model_sgd)}</td>")
        out.append(f"<td>{_fmt_money_signed(r.gap_sgd)}</td>")
        out.append(f"<td>{_fmt_pct(r.gap_pct)}</td>")
        out.append(f"<td>{badge}</td>")
        out.append("</tr>")
        if r.app_start_sgd is not None:
            total_start += r.app_start_sgd
        if r.app_end_sgd is not None:
            total_end += r.app_end_sgd
            n_app += 1
        total_our += r.our_model_sgd
        total_dep += r.deposits_sgd
        if r.gap_sgd is not None:
            total_gap += r.gap_sgd
        if perf is not None:
            total_perf += perf
    out.append("</tbody><tfoot><tr style=\"font-weight:600;border-top:2px solid #30363d;\">")
    out.append(f'<td>TOTAL ({n_app} sleeves with app)</td><td></td>')
    out.append(f"<td>{_fmt_money(total_start)}</td>")
    out.append(f"<td>{_fmt_money(total_end)}</td>")
    out.append(f"<td>{_fmt_money_signed(total_dep)}</td>")
    out.append(f"<td>{_fmt_money_signed(total_perf)}</td>")
    perf_pct_tot = (total_perf / total_start * 100.0) if total_start > 0 else None
    out.append(f"<td>{_fmt_pct(perf_pct_tot)}</td>")
    out.append(f"<td>{_fmt_money(total_our)}</td>")
    out.append(f"<td>{_fmt_money_signed(total_gap)}</td>")
    gap_pct_tot = (total_gap / total_end * 100.0) if total_end > 0 else None
    out.append(f"<td>{_fmt_pct(gap_pct_tot)}</td><td></td>")
    out.append("</tr></tfoot></table>")
    return "".join(out)


def _deposits_log_html(deposits: list[Deposit]) -> str:
    if not deposits:
        return "<p>No post-statement deposits recorded.</p>"
    out: list[str] = ['<table class="ds-table deposits-log"><thead><tr>']
    for h in ["Date", "Portfolio", "Ccy", "Amount", "Notes"]:
        out.append(f"<th>{h}</th>")
    out.append("</tr></thead><tbody>")
    sorted_deps = sorted(deposits, key=lambda d: d.date)
    for d in sorted_deps:
        out.append(
            "<tr>"
            f"<td>{d.date.isoformat()}</td>"
            f"<td>{d.portfolio}</td>"
            f"<td>{d.currency}</td>"
            f"<td>{_fmt_money_signed(d.amount)}</td>"
            f"<td>{d.notes}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _methodology_html() -> str:
    return """
<p><strong>Architecture (signed off 2026-05-31):</strong> reconcile.html answers ONE
question — "does our model match the app?". App values (start AND end) come
from the user-maintained <code>PortfolioValueSGD</code> sheet in
<code>holding.xlsx</code>; our model is the validation column.</p>
<ul>
  <li><strong>App (start) / App (end)</strong>: both sourced from the
      <code>PortfolioValueSGD</code> sheet — start = snapshot closest to
      the statement date; end = snapshot closest to today. These are the
      "what really happened" numbers, untouched by our reconstruction.</li>
  <li><strong>Δ Perf</strong>: <code>app_end - app_start - deposits</code>.
      Pure deposit-adjusted performance per sleeve, in SGD.</li>
  <li><strong>Weight source: target % (piecewise)</strong> — uses the target
      snapshot effective at each sub-period within (stmt_date, end_date]. When
      you change a target mid-period (e.g. removed Crypto from GI on May 26),
      the period is sliced and each sub-period uses its own targets, with
      cumulative compounding across segments. Falls back to stmt-date actual
      weights for segments where target coverage is incomplete.</li>
  <li><strong>Value movement: buy-and-hold per holding</strong> within each
      sub-period — <code>seg_return = Σ_h w_seg[h] × (p_end / p_start)</code>.
      The portfolio cum return is the product across segments.</li>
  <li><strong>FX</strong> pinned to your PDF stmt's rate at start (so stmt SGD
      matches the app exactly); mark-to-market Yahoo daily USDSGD for every
      other date.</li>
  <li><strong>Cash / MMF forward-accrual</strong>: where Yahoo NAV for a
      low-vol holding (Cash, Money Market, Treasury 0-3M) lags by 1-2 days,
      we project missing days at the holding's own realised yield. Marked in
      drill-down with <code>*</code> and <code>(+Nd)</code>.</li>
  <li><strong>Synthetic cash dropped (M5, 2026-05-31)</strong>:
      <code>CASH_USD</code> / <code>CASH_SGD</code> placeholders now return
      zero. Real cash positions use real Yahoo tickers (LionGlobal SGD MMF +
      Enhanced Liquidity, OCBC SGD MMF, BB3M.L) that flow through the
      buy-and-hold path with forward-accrual.</li>
  <li><strong>Deposits</strong> summed from <code>holding.xlsx</code>'s
      Deposits sheet (rows with date &gt; stmt date). Subtracted from Δ Perf
      and added into our-model+deposits for the gap calculation.</li>
  <li><strong>Drill-down vs sleeve total</strong>: per-holding values in the
      drill-down use simple point-to-point (stmt → today) ratios. The sleeve
      Total there sums those — for sleeves that DID change targets mid-period
      this may differ from "Our Model" in the table above by the rebalancing
      effect. Treat the table's "Our Model" as the authoritative piecewise
      number; the drill-down is the per-holding break-out.</li>
  <li><strong>Status badges</strong>: exact (&lt;0.05%), ok (&lt;0.3%),
      noise (&lt;1%), ⚠ investigate (≥1%).</li>
</ul>
""".strip()


def render_reconcile_report(
    portfolios: Sequence[Portfolio],
    output_path: Path | str,
    *,
    end: date | None = None,
    price_source: Any,
    fx_series_usd_sgd: pd.Series,
    target_weights_by_portfolio: dict[str, list[tuple[date, dict[str, float]]]] | None = None,
    title: str = "Allocation Reconciliation",
) -> Path:
    """Render the reconcile HTML report.

    Both the start-of-period and end-of-period sleeve values come from the
    user-maintained ``PortfolioValueSGD`` sheet in ``data/holding.xlsx``.
    The start snapshot is the column closest to the latest statement date; the
    end snapshot is the column closest to ``end`` (or today). Deposits and our
    piecewise model are used to compute the performance delta and the model
    accuracy gap.
    """
    end_date = end or date.today()
    holdings_books = [p for p in portfolios if Role.HOLDING in p.roles]
    stmt_date = max(p.statement_date for p in holdings_books)

    deposits = load_deposits(since=stmt_date)
    app_start, start_snap_date = load_app_values_at_date(stmt_date)
    app_end, end_snap_date = load_app_values_at_date(end_date)

    # Synthesize new sleeves once and pass the seed-excluded deposits list
    # to build_reconcile. build_reconcile's internal synthesis pass becomes a
    # noop (all sleeves are now in existing) so the seed deposits stay out of
    # the "+ Deposits" column.
    new_ports, residual_deps = _synthesize_new_sleeves(
        list(portfolios), list(deposits), app_end,
        target_weights_by_portfolio, fx_series_usd_sgd,
    )
    all_portfolios = list(portfolios) + new_ports

    rows = build_reconcile(
        all_portfolios,
        end=end_date,
        price_source=price_source,
        fx_series_usd_sgd=fx_series_usd_sgd,
        target_weights_by_portfolio=target_weights_by_portfolio,
        deposits=residual_deps,
        app_values_start=app_start,
        app_values_end=app_end,
    )

    from hailmary.allocation.diagnostic import _holdings_drilldown_to_html
    drilldown_html = _holdings_drilldown_to_html(
        all_portfolios,
        end=end_date,
        price_source=price_source,
        fx_series_usd_sgd=fx_series_usd_sgd,
        target_weights_by_portfolio=target_weights_by_portfolio,
    )

    template = _load_reconcile_template()
    rendered = template.render(
        title=title,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        statement_date=stmt_date.isoformat(),
        end_date=end_date.isoformat(),
        snapshot_date=end_snap_date.isoformat() if end_snap_date else "—",
        start_snapshot_date=start_snap_date.isoformat() if start_snap_date else "—",
        n_deposits=len(deposits),
        reconcile_table=_reconcile_table_html(
            rows, start_date=start_snap_date, end_date=end_snap_date
        ),
        deposits_log=_deposits_log_html(deposits),
        drilldown=drilldown_html,
        methodology=_methodology_html(),
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    return out


def _load_reconcile_template() -> Any:
    import jinja2

    if _RECONCILE_TEMPLATE_PATH.exists():
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_RECONCILE_TEMPLATE_PATH.parent)),
            autoescape=jinja2.select_autoescape(["html"]),
        )
        return env.get_template(_RECONCILE_TEMPLATE_PATH.name)
    raise FileNotFoundError(
        f"Reconcile template not found at {_RECONCILE_TEMPLATE_PATH}"
    )
