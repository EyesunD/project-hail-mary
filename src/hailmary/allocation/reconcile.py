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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
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
    fx_to_base: float | None = None  # Stashaway's actual rate at deposit (deposit_amount × fx_to_base = native amount in portfolio's base currency); None → fall back to Yahoo midmarket with spread haircut


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
            fx_raw = ws.cell(row=r, column=6).value  # "FX Rate to base" column (optional)
            if d is None or p is None or a is None:
                continue
            d_val = d.date() if hasattr(d, "date") else d
            if since is not None and d_val <= since:
                continue
            try:
                fx_to_base = float(fx_raw) if fx_raw not in (None, "") else None
            except (TypeError, ValueError):
                fx_to_base = None
            out.append(
                Deposit(
                    date=d_val,
                    portfolio=_PORTFOLIO_ALIASES.get(str(p), str(p)),
                    currency=str(c).upper() if c else "SGD",
                    amount=float(a),
                    notes=str(n) if n else "",
                    fx_to_base=fx_to_base,
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


_VALUE_COL_RE = re.compile(
    r"^\s*(\d{1,2}/\d{1,2}/\d{4})\s*\(\s*SGD\s*\)\s*$", re.IGNORECASE
)
_VALUE_COL_RE_BOTH = re.compile(
    r"^\s*(\d{1,2}/\d{1,2}/\d{4})\s*\(\s*(SGD|Base)\s*\)\s*$", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class _ObservedSnapshot:
    """One PortfolioValue row read at a given date."""

    base_value: float
    sgd_value: float
    currency: str


def load_portfolio_values_full(
    links_path: Path = Path("data/holding.xlsx"),
) -> dict[date, dict[str, _ObservedSnapshot]]:
    """Read `PortfolioValue` returning native+SGD+currency per (date, sleeve).

    Returns ``{snapshot_date: {portfolio_name: _ObservedSnapshot}}`` covering
    every column pair ``MM/DD/YYYY (SGD)`` + ``MM/DD/YYYY (Base)``. Used by the
    multi-anchor drill-down which works in native currency per sleeve.
    """
    if not links_path.exists():
        return {}
    try:
        import openpyxl
    except ImportError:
        return {}
    wb = openpyxl.load_workbook(links_path, data_only=False)
    if "PortfolioValue" not in wb.sheetnames:
        return {}
    ws = wb["PortfolioValue"]
    HEADER_ROW = 6
    name_col: int | None = None
    cur_col: int | None = None
    sgd_cols: dict[date, int] = {}
    base_cols: dict[date, int] = {}
    for c in range(1, ws.max_column + 1):
        v = ws.cell(row=HEADER_ROW, column=c).value
        if v is None:
            continue
        label = str(v).strip()
        if label == "Public Market Investments":
            name_col = c
            continue
        if label == "Base Currency":
            cur_col = c
            continue
        m = _VALUE_COL_RE_BOTH.match(label)
        if m:
            mm, dd, yyyy = m.group(1).split("/")
            d = date(int(yyyy), int(mm), int(dd))
            if m.group(2).upper() == "SGD":
                sgd_cols[d] = c
            else:
                base_cols[d] = c
    if name_col is None or cur_col is None:
        return {}
    snap_dates = sorted(set(sgd_cols) & set(base_cols))
    out: dict[date, dict[str, _ObservedSnapshot]] = {d: {} for d in snap_dates}
    for r in range(HEADER_ROW + 1, ws.max_row + 1):
        raw_name = ws.cell(row=r, column=name_col).value
        if raw_name is None or not str(raw_name).strip():
            continue
        name = _PORTFOLIO_ALIASES.get(str(raw_name).strip(), str(raw_name).strip())
        currency = ws.cell(row=r, column=cur_col).value
        if currency is None:
            continue
        currency = str(currency).strip().upper()
        for d in snap_dates:
            sgd_v = ws.cell(row=r, column=sgd_cols[d]).value
            base_v = ws.cell(row=r, column=base_cols[d]).value
            if sgd_v in (None, "") or base_v in (None, ""):
                continue
            try:
                out[d][name] = _ObservedSnapshot(
                    base_value=float(base_v),
                    sgd_value=float(sgd_v),
                    currency=currency,
                )
            except (TypeError, ValueError):
                continue
    return out


def load_app_values_all_snapshots(
    links_path: Path = Path("data/holding.xlsx"),
) -> dict[date, dict[str, float]]:
    """Read every SGD app-value snapshot from the ``PortfolioValue`` sheet.

    Returns ``{snapshot_date: {portfolio_name: sgd_value}}`` sourced from
    columns named ``MM/DD/YYYY (SGD)``. The ``(Base)`` companion columns are
    ignored — reconcile.html is SGD-base throughout.
    """
    if not links_path.exists():
        return {}
    try:
        import openpyxl
    except ImportError:
        return {}
    try:
        wb = openpyxl.load_workbook(links_path, data_only=False)
        if "PortfolioValue" not in wb.sheetnames:
            return {}
        ws = wb["PortfolioValue"]
        HEADER_ROW = 6
        name_col: int | None = None
        sgd_cols: list[tuple[date, int]] = []
        for c in range(1, ws.max_column + 1):
            v = ws.cell(row=HEADER_ROW, column=c).value
            if v is None:
                continue
            label = str(v).strip()
            if label == "Public Market Investments":
                name_col = c
                continue
            m = _VALUE_COL_RE.match(label)
            if m:
                mm, dd, yyyy = m.group(1).split("/")
                sgd_cols.append((date(int(yyyy), int(mm), int(dd)), c))
        if not sgd_cols or name_col is None:
            return {}
        out: dict[date, dict[str, float]] = {}
        for snap_date, col in sgd_cols:
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
        warnings.warn(f"Could not read PortfolioValue sheet: {exc}", stacklevel=2)
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
    observed_by_date: dict[date, dict[str, _ObservedSnapshot]] | None = None,
) -> list[SleeveReconcile]:
    """Per-sleeve reconciliation.

    Performance lens: ``Δ Perf = app_end − app_start − deposits`` — what really
    happened to your money, sourced from the ``PortfolioValue`` sheet.

    Validation lens: ``gap = our_model − app_end`` — does our multi-anchor
    BH-with-deposit-injection model agree with the app within tolerance? Status
    badges classify the gap by % of app_end. The model already includes the
    cash flows shown in ``+ Deposits``, so the column is informational and is
    NOT added on top of ``our_model`` for the gap calculation.
    """
    fx_today = float(fx_series_usd_sgd.dropna().iloc[-1])
    app_start = app_values_start or {}
    app_end = app_values_end or {}

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
        if d.currency == "SGD":
            amt_sgd = d.amount
        else:
            # USD-denominated deposit — convert to SGD at the FX prevailing on
            # the deposit date (not today's FX, which would be stale and the
            # mismatch with the SGD-base table would show up as a phantom drift).
            fx_at_d = _fx_asof(fx_series_usd_sgd, d.date) or fx_today
            amt_sgd = d.amount * fx_at_d
        deposits_by_port[d.portfolio] = deposits_by_port.get(d.portfolio, 0.0) + amt_sgd

    # Multi-anchor BH model from the earliest PortfolioValue snapshot to end.
    # This is what the drill-down also uses — single source of truth for "Our
    # Model" across both tables.
    if observed_by_date is None:
        observed_by_date = {}
    snap_dates = sorted(d for d in observed_by_date if d <= end)
    bars: pd.DataFrame | None = None
    if snap_dates and price_source is not None:
        earliest = snap_dates[0]
        fetch_start = earliest - timedelta(days=7)
        from hailmary.allocation.universe import STASHAWAY_UNIVERSE as _UNIV
        extra_sids: set[str] = set()
        if target_weights_by_portfolio:
            for series in target_weights_by_portfolio.values():
                for _d, w in series:
                    extra_sids.update(w.keys())
        all_symbols = sorted({
            h.metadata.ticker
            for p in all_portfolios
            for h in p.holdings
            if not h.metadata.ticker.startswith("CASH_")
        } | {
            _UNIV[sid].ticker for sid in extra_sids
            if sid in _UNIV and not _UNIV[sid].ticker.startswith("CASH_")
        })
        if all_symbols:
            try:
                bars = price_source.get_bars(all_symbols, fetch_start, end)
            except Exception as exc:  # pragma: no cover — provider failure
                warnings.warn(f"Could not fetch price bars: {exc}", stacklevel=2)

    # First-deposit lookup for new-sleeve anchor synthesis (same as drilldown).
    first_deposit_by_port: dict[str, Deposit] = {}
    for d in (deposits or []):
        prev = first_deposit_by_port.get(d.portfolio)
        if prev is None or d.date < prev.date:
            first_deposit_by_port[d.portfolio] = d

    rows: list[SleeveReconcile] = []
    for p in all_portfolios:
        if Role.HOLDING not in p.roles:
            continue
        # Build anchors: PortfolioValue snapshots + end_date (trailing endpoint)
        anchors: list[_Anchor] = []
        for d in snap_dates:
            snap = observed_by_date.get(d, {}).get(p.name)
            if snap and snap.base_value > 0:
                anchors.append(_Anchor(
                    snap_date=d,
                    observed_native=snap.base_value,
                    observed_sgd=snap.sgd_value,
                    fx_native_to_sgd=(snap.sgd_value / snap.base_value) if snap.base_value > 0 else None,
                ))
            else:
                anchors.append(_Anchor(snap_date=d, observed_native=None))
        endpoint_fx = fx_today if p.currency.upper() == "USD" else 1.0
        if end not in snap_dates:
            anchors.append(_Anchor(
                snap_date=end, observed_native=None,
                fx_native_to_sgd=endpoint_fx,
            ))

        # Inject deposit anchor for new sleeves (same logic as the drilldown).
        first_real_idx = next(
            (i for i, a in enumerate(anchors) if a.observed_native is not None), None
        )
        dep = first_deposit_by_port.get(p.name)
        if first_real_idx is not None and first_real_idx > 0 and dep is not None:
            prev_anchor_date = anchors[first_real_idx - 1].snap_date
            next_anchor_date = anchors[first_real_idx].snap_date
            if prev_anchor_date <= dep.date < next_anchor_date and dep.amount > 0:
                native_amount, sgd_amount, fx_dep = _deposit_anchor_values(
                    dep, p.currency, fx_series_usd_sgd, fx_today
                )
                synthetic = _Anchor(
                    snap_date=dep.date,
                    observed_native=native_amount,
                    observed_sgd=sgd_amount,
                    fx_native_to_sgd=fx_dep,
                )
                anchors = anchors[first_real_idx:]
                anchors.insert(0, synthetic)
        while anchors and anchors[0].observed_native is None:
            anchors.pop(0)
        if not anchors:
            continue

        target_series = (
            _resolve_target_series(target_weights_by_portfolio.get(p.name))
            if target_weights_by_portfolio else None
        )
        view = _compute_multi_anchor(
            p, anchors, target_series, bars,
            deposits=deposits, fx_series_usd_sgd=fx_series_usd_sgd,
        )
        if view is None:
            continue
        sleeve_bh_native_end = view["sleeve_bh"][-1]
        # Convert to SGD using the FX at the end-date anchor (today's FX for USD sleeves)
        if p.currency.upper() == "USD":
            our_model_sgd = sleeve_bh_native_end * fx_today
        else:
            our_model_sgd = sleeve_bh_native_end

        dep_sgd = deposits_by_port.get(p.name, 0.0)
        start_v = app_start.get(p.name)
        end_v = app_end.get(p.name)
        if end_v is not None:
            gap = our_model_sgd - end_v
            gap_pct = gap / end_v * 100.0 if end_v else None
        else:
            gap = None
            gap_pct = None
        rows.append(
            SleeveReconcile(
                portfolio=p.name,
                currency=p.currency.upper(),
                app_start_sgd=start_v,
                app_end_sgd=end_v,
                our_model_sgd=our_model_sgd,
                deposits_sgd=dep_sgd,
                total_sgd=our_model_sgd,  # No separate "+ deposits" add — model already includes them
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
    out: list[str] = [
        '<table class="ds-table reconcile-table sortable" data-default-sort="2" data-default-dir="desc">'
        '<thead><tr>'
    ]
    # (label, data-type) — "num" for sortable numeric columns, "str" for text,
    # "status" for the badge column (sorted by class hierarchy).
    headers: list[tuple[str, str]] = [
        ("Sleeve", "str"),
        ("Ccy", "str"),
        (f"App ({start_lbl})<br><small>SGD</small>", "num"),
        (f"App ({end_lbl})<br><small>SGD</small>", "num"),
        ("+ Deposits<br><small>SGD</small>", "num"),
        ("Δ Perf<br><small>SGD</small>", "num"),
        ("Δ Perf %", "num"),
        ("Our Model<br><small>SGD</small>", "num"),
        ("Model Δ<br><small>SGD</small>", "num"),
        ("Model Δ %", "num"),
        ("Gap<br><small>SGD</small>", "num"),
        ("Gap %", "num"),
        ("Status", "status"),
    ]
    for label, dtype in headers:
        out.append(f'<th data-type="{dtype}">{label}<span class="sort-arrow"></span></th>')
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
        # Δ Perf %: denominator is app_start when the sleeve existed at start;
        # for new sleeves (start = 0) fall back to deposits — the capital
        # actually invested — so the % means "return on deposited capital".
        def _pct_of(numerator: float | None) -> float | None:
            if numerator is None:
                return None
            if start_v > 0:
                return numerator / start_v * 100.0
            if r.deposits_sgd > 0:
                return numerator / r.deposits_sgd * 100.0
            return None

        perf_pct = _pct_of(perf)
        # Model Δ = our_model − app_start − deposits (model's analog of Δ Perf)
        model_perf = (
            r.our_model_sgd - start_v - r.deposits_sgd
            if r.app_start_sgd is not None else None
        )
        model_perf_pct = _pct_of(model_perf)
        def _cls(v: float | None) -> str:
            if v is None or (isinstance(v, float) and v != v):
                return ""
            return "pos" if v >= 0 else "neg"

        def _num_cell(val: float | None, fmt: Any, color: bool = False) -> str:
            """Right-aligned numeric cell with a data-sort hook and optional pos/neg colour."""
            sort_key = "" if val is None or (isinstance(val, float) and val != val) else f'{val:.6f}'
            cls = f' class="{_cls(val)}"' if color else ""
            return f'<td data-sort="{sort_key}"{cls}>{fmt(val)}</td>'

        # Status sort order — exact < ok < noise < warn < no app value
        status_order = {"exact": 0, "ok": 1, "noise": 2, "warn": 3, "no app value": 4}
        status_sort = status_order.get(r.status, 5)

        out.append("<tr>")
        out.append(f'<td data-sort="{r.portfolio}">{r.portfolio}</td>')
        out.append(f'<td data-sort="{r.currency}">{r.currency}</td>')
        out.append(_num_cell(r.app_start_sgd, _fmt_money))
        out.append(_num_cell(r.app_end_sgd, _fmt_money))
        out.append(_num_cell(r.deposits_sgd, _fmt_money_signed, color=True))
        out.append(_num_cell(perf, _fmt_money_signed, color=True))
        out.append(_num_cell(perf_pct, _fmt_pct, color=True))
        out.append(_num_cell(r.our_model_sgd, _fmt_money))
        out.append(_num_cell(model_perf, _fmt_money_signed, color=True))
        out.append(_num_cell(model_perf_pct, _fmt_pct, color=True))
        out.append(_num_cell(r.gap_sgd, _fmt_money_signed, color=True))
        out.append(_num_cell(r.gap_pct, _fmt_pct, color=True))
        out.append(f'<td data-sort="{status_sort}">{badge}</td>')
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
    model_perf_tot = total_our - total_start - total_dep
    out.append(f"<td>{_fmt_money_signed(model_perf_tot)}</td>")
    model_perf_pct_tot = (model_perf_tot / total_start * 100.0) if total_start > 0 else None
    out.append(f"<td>{_fmt_pct(model_perf_pct_tot)}</td>")
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
<p><strong>Reconcile answers one question:</strong> does our model match the
app? App values come from the user-maintained <code>PortfolioValue</code>
sheet; our model compounds forward from there and the gap is the
model-accuracy signal.</p>
<ul>
  <li><strong>App (start) / App (end)</strong>: sourced from the
      <code>PortfolioValue</code> sheet's <code>(SGD)</code> columns. Start
      = earliest snapshot ≤ end_date; end = latest snapshot ≤ end_date. The
      reconciliation window is the gap between these two dates.</li>
  <li><strong>Δ Perf</strong>: <code>app_end − app_start − deposits</code>
      — deposit-adjusted real performance per sleeve, in SGD. For new
      sleeves (start = 0) the % column uses deposits as the denominator so
      the % is "return on capital invested".</li>
  <li><strong>Our Model</strong>: multi-anchor BH with deposit injection.
      At the start anchor we allocate observed value × target weights to
      each holding (fixed shares). We compound forward via Yahoo prices,
      rebalancing shares at every target-change date. Mid-period deposits
      are injected at their date — converted to native currency using the
      explicit Stashaway rate from the Deposits sheet (column "FX Rate to
      base") when available, else Yahoo midmarket with a 0.2% spread
      haircut as a default. Stashaway management fees from
      <code>book_config.MGMT_FEES_ANNUAL</code> are pro-rated daily.</li>
  <li><strong>Gap</strong>: <code>our_model − app_end</code>. The model
      already includes the cash flows shown in "+ Deposits" — adding them
      again would double-count.</li>
  <li><strong>Status badges</strong>: exact (&lt;0.05%), ok (&lt;0.3%),
      noise (&lt;1%), ⚠ investigate (≥1%) — % of app_end.</li>
  <li><strong>Drill-down (per-portfolio)</strong>: anchors at every
      <code>PortfolioValue</code> snapshot date (+ end_date), with a
      synthetic anchor at the deposit date for sleeves seeded mid-period.
      Per holding: $-value at each anchor (native currency with SGD
      subtext for USD sleeves), 1M Δ and Cum Δ. Sleeve footer shows
      Observed / BH-modelled / DR-modelled (daily-rebalanced) with two
      drift rows — BH vs Observed (model accuracy) and BH vs DR
      (rebalancing-assumption effect).</li>
  <li><strong>Data-integrity banner</strong>: <code>book_health_check</code>
      runs on every regeneration and surfaces untagged portfolios, unknown
      tickers, weight-sum violations, mid-period deposits and FX staleness
      as a banner above the main table.</li>
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
    links_path: Path | str = Path("data/holding.xlsx"),
) -> Path:
    """Render the reconcile HTML report.

    Both the start-of-period and end-of-period sleeve values come from the
    user-maintained ``PortfolioValue`` sheet (``(SGD)`` columns) in
    ``data/holding.xlsx``. The start snapshot is the column closest to the
    latest statement date; the end snapshot is the column closest to ``end``
    (or today). Deposits and our piecewise model are used to compute the
    performance delta and the model accuracy gap.
    """
    end_date = end or date.today()
    holdings_books = [p for p in portfolios if Role.HOLDING in p.roles]
    stmt_date = max(p.statement_date for p in holdings_books)
    links_path = Path(links_path)

    if target_weights_by_portfolio is None:
        from hailmary.allocation.diagnostic import load_target_weights
        target_weights_by_portfolio = load_target_weights(links_path)

    observed = load_portfolio_values_full(links_path)
    snap_dates = sorted(d for d in observed if d <= end_date)

    all_deposits = load_deposits(links_path)  # un-filtered, for drilldown anchor injection
    # Use the EARLIEST observed snapshot for the start anchor (was closest-to-stmt
    # which collapses to the same date as end when the book is loaded at today).
    if snap_dates:
        start_snap_date: date | None = snap_dates[0]
        end_snap_date: date | None = snap_dates[-1] if snap_dates[-1] != snap_dates[0] else snap_dates[-1]
        app_start = {n: s.sgd_value for n, s in observed[start_snap_date].items()}
        app_end = {n: s.sgd_value for n, s in observed[end_snap_date].items()}
    else:
        app_start, start_snap_date = load_app_values_at_date(stmt_date, links_path)
        app_end, end_snap_date = load_app_values_at_date(end_date, links_path)

    # Deposits used by the top reconciliation table are bounded by BOTH ends
    # of the reconciliation window: strictly after start_snap_date (anything
    # on/before is already in app_start) AND on/before end_snap_date (anything
    # after isn't reflected in app_end yet, so including it would skew Δ Perf
    # and create a phantom Gap). Drilldown uses `all_deposits` separately for
    # the per-anchor model up to end_date, so future-dated deposits still
    # appear in the multi-anchor view.
    deposit_cutoff = start_snap_date if start_snap_date is not None else stmt_date
    deposits_all = load_deposits(links_path, since=deposit_cutoff)
    if end_snap_date is not None:
        deposits = [d for d in deposits_all if d.date <= end_snap_date]
    else:
        deposits = list(deposits_all)

    # Synthesize new sleeves once and pass the seed-excluded deposits list
    # to build_reconcile. build_reconcile's internal synthesis pass becomes a
    # noop (all sleeves are now in existing) so the seed deposits stay out of
    # the "+ Deposits" column.
    new_ports, residual_deps = _synthesize_new_sleeves(
        list(portfolios), list(deposits), app_end,
        target_weights_by_portfolio, fx_series_usd_sgd,
    )
    all_portfolios = list(portfolios) + new_ports

    # Model end for the top table is the LAST observed anchor (so Gap compares
    # like-with-like). Drilldown still uses end_date (today) so the BH/DR
    # projection extends to the present in the per-holding view.
    top_table_end = end_snap_date if end_snap_date is not None else end_date
    rows = build_reconcile(
        all_portfolios,
        end=top_table_end,
        price_source=price_source,
        fx_series_usd_sgd=fx_series_usd_sgd,
        target_weights_by_portfolio=target_weights_by_portfolio,
        deposits=residual_deps,
        app_values_start=app_start,
        app_values_end=app_end,
        observed_by_date=observed,
    )

    drilldown_html = _multi_anchor_drilldown_to_html(
        all_portfolios,
        observed_by_date=observed,
        end_date=end_date,
        price_source=price_source,
        target_weights_by_portfolio=target_weights_by_portfolio,
        deposits=list(all_deposits),
        fx_series_usd_sgd=fx_series_usd_sgd,
        links_path=links_path,
    )

    health_findings = book_health_check(
        all_portfolios,
        observed_by_date=observed,
        deposits=all_deposits,
        target_weights_by_portfolio=target_weights_by_portfolio,
        end_date=end_date,
        fx_series_usd_sgd=fx_series_usd_sgd,
    )
    health_html = _health_check_html(health_findings)

    template = _load_reconcile_template()
    rendered = template.render(
        title=title,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        end_date=end_date.isoformat(),
        snapshot_date=end_snap_date.isoformat() if end_snap_date else "—",
        start_snapshot_date=start_snap_date.isoformat() if start_snap_date else "—",
        n_deposits=len(deposits),
        health_check=health_html,
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


def book_health_check(
    portfolios: Sequence[Portfolio],
    *,
    observed_by_date: dict[date, dict[str, _ObservedSnapshot]],
    deposits: Sequence[Deposit],
    target_weights_by_portfolio: dict[str, list[tuple[date, dict[str, float]]]] | None,
    end_date: date,
    fx_series_usd_sgd: pd.Series | None = None,
) -> list[tuple[str, str]]:
    """Surface data-integrity gaps that would otherwise corrupt the report silently.

    Returns a list of ``(level, message)`` tuples where ``level`` is one of
    ``"ok"`` / ``"warn"`` / ``"err"``. The renderer turns these into a banner
    at the top of reconcile.html.
    """
    from hailmary.allocation.book_config import ROLES
    from hailmary.allocation.universe import STASHAWAY_UNIVERSE

    findings: list[tuple[str, str]] = []
    holdings_books = [p for p in portfolios if Role.HOLDING in p.roles]

    # 1. Portfolios missing from ROLES (silent-default to HOLDING)
    untagged = sorted(p.name for p in portfolios if p.name not in ROLES)
    if untagged:
        findings.append((
            "warn",
            f"{len(untagged)} portfolio(s) without explicit role in book_config.ROLES "
            f"(defaulted to HOLDING): {', '.join(untagged)}",
        ))

    # 2. Unknown tickers — already filtered by load_book but check for safety
    unknown: list[str] = []
    for p in holdings_books:
        for h in p.holdings:
            if h.ticker not in STASHAWAY_UNIVERSE:
                unknown.append(f"{p.name}/{h.ticker}")
    if unknown:
        findings.append((
            "err",
            f"{len(unknown)} unknown ticker(s) — must be added to STASHAWAY_UNIVERSE: "
            f"{', '.join(unknown[:8])}{' ...' if len(unknown) > 8 else ''}",
        ))

    # 3. Sleeves observed at $0 with no first-deposit anchor (model can't run)
    snap_dates = sorted(d for d in observed_by_date if d <= end_date)
    if snap_dates:
        first_d = snap_dates[0]
        deposits_by_port = {d.portfolio: d for d in sorted(deposits, key=lambda x: x.date)}
        zero_unanchored: list[str] = []
        for p in holdings_books:
            snap = observed_by_date.get(first_d, {}).get(p.name)
            if snap is None or snap.base_value <= 0:
                if p.name not in deposits_by_port:
                    zero_unanchored.append(p.name)
        if zero_unanchored:
            findings.append((
                "warn",
                f"{len(zero_unanchored)} sleeve(s) with $0 at first snapshot AND no logged "
                f"deposit — drill-down will be empty: {', '.join(zero_unanchored)}",
            ))

    # 4. Sleeves missing target weights (model falls back to current actuals)
    if target_weights_by_portfolio is not None:
        no_targets = [
            p.name for p in holdings_books
            if p.name not in target_weights_by_portfolio
        ]
        if no_targets:
            findings.append((
                "warn",
                f"{len(no_targets)} sleeve(s) with no Target % column "
                f"(piecewise model falls back to statement weights): "
                f"{', '.join(no_targets)}",
            ))

    # 5. Target weight sums that don't add to 1.0 ± tolerance per snapshot
    if target_weights_by_portfolio:
        for name, series in target_weights_by_portfolio.items():
            for d, weights in series:
                s = sum(weights.values())
                if abs(s - 1.0) > 0.001:
                    findings.append((
                        "warn",
                        f"{name} target weights on {d.isoformat()} sum to {s:.4f} "
                        f"(should be 1.0)",
                    ))

    # 6. Existing sleeves with mid-period deposits — informational only, since
    # the model now injects deposits at their date (BH allocates at prevailing
    # target weights, DR adds to running sleeve value). Drift should reflect
    # tracking error rather than missing capital.
    deposits_into_existing: dict[str, float] = {}
    if snap_dates:
        first_d = snap_dates[0]
        already_funded = {
            p.name for p in holdings_books
            if (snap := observed_by_date.get(first_d, {}).get(p.name)) and snap.base_value > 0
        }
        for dep in deposits:
            if dep.portfolio in already_funded and dep.date > first_d:
                deposits_into_existing[dep.portfolio] = (
                    deposits_into_existing.get(dep.portfolio, 0.0) + dep.amount
                )
    if deposits_into_existing:
        items = sorted(deposits_into_existing.items(), key=lambda kv: -abs(kv[1]))[:5]
        descs = [f"{n} ({a:+,.0f})" for n, a in items]
        findings.append((
            "ok",
            f"{len(deposits_into_existing)} sleeve(s) had mid-period deposits (auto-injected into model): "
            f"{', '.join(descs)}{' ...' if len(deposits_into_existing) > 5 else ''}",
        ))

    # 7. SGD-base sleeves whose `(Base)` ≠ `(SGD)` in PortfolioValue
    #    For SGD-base sleeves the two columns must be equal (no conversion).
    #    Mismatch usually means a typo or a stale pre-filled value.
    sgd_inconsistent: list[str] = []
    for d, snaps in observed_by_date.items():
        for name, snap in snaps.items():
            if snap.currency == "SGD" and abs(snap.base_value - snap.sgd_value) > 0.01:
                sgd_inconsistent.append(
                    f"{name} on {d.isoformat()} "
                    f"(base={snap.base_value:,.2f} vs sgd={snap.sgd_value:,.2f}; "
                    f"diff {snap.base_value - snap.sgd_value:+,.2f})"
                )
    if sgd_inconsistent:
        findings.append((
            "err",
            f"{len(sgd_inconsistent)} SGD-base sleeve(s) have (Base) ≠ (SGD) in "
            f"PortfolioValue — for SGD-base sleeves the two columns must be equal: "
            f"{'; '.join(sgd_inconsistent)}",
        ))

    # 8. FX series staleness
    if fx_series_usd_sgd is not None and not fx_series_usd_sgd.empty:
        fx_clean = fx_series_usd_sgd.dropna()
        last_fx_date = fx_clean.index.max()
        last_fx_d = last_fx_date.date() if hasattr(last_fx_date, "date") else last_fx_date
        gap_days = (end_date - last_fx_d).days
        if gap_days > 3:
            findings.append((
                "warn",
                f"USDSGD series last value is {last_fx_d.isoformat()} "
                f"({gap_days} days behind end_date {end_date.isoformat()})",
            ))

    if not findings:
        findings.append(("ok", "All integrity checks passed."))
    return findings


def _health_check_html(findings: list[tuple[str, str]]) -> str:
    all_ok = all(lvl == "ok" for lvl, _ in findings)
    cls = "health-check all-clear" if all_ok else "health-check"
    rows = []
    for lvl, msg in findings:
        marker = {"ok": "✓", "warn": "⚠", "err": "✕"}.get(lvl, "·")
        rows.append(
            f'<div class="health-row lvl-{lvl}">{marker} {msg}</div>'
        )
    return (
        f'<div class="{cls}">'
        f"<h3>Data-integrity check ({len(findings)} finding{'s' if len(findings) != 1 else ''})</h3>"
        + "".join(rows)
        + "</div>"
    )


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


# ---------------------------------------------------------------------------
# Multi-anchor drill-down
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Anchor:
    """One point on the per-sleeve timeline."""

    snap_date: date
    observed_native: float | None  # None at the trailing "today" if no observation
    observed_sgd: float | None = None  # SGD-equivalent of observed_native at this date
    fx_native_to_sgd: float | None = None  # = sgd / native; used to convert modeled values


def _fx_asof(fx_series: pd.Series | None, target: date) -> float | None:
    """Most-recent USDSGD rate at or before *target*. Returns None if no data."""
    if fx_series is None or fx_series.empty:
        return None
    fx_clean = fx_series.dropna()
    if hasattr(fx_clean.index, "tz") and fx_clean.index.tz is not None:
        fx_clean = fx_clean.copy()
        fx_clean.index = fx_clean.index.tz_localize(None)
    sub = fx_clean[fx_clean.index <= pd.Timestamp(target)]
    if sub.empty:
        return None
    return float(sub.iloc[-1])


_DEPOSIT_FX_SPREAD = 0.002
"""Default Stashaway-equivalent FX spread for cross-currency deposits when
the user has not provided an explicit ``fx_to_base`` on the Deposits sheet.
Empirically ~0.2% based on a Global Floating Rate USD deposit comparison
(Stashaway rate 0.7791 vs Yahoo midmarket ~0.7807 on 2026-05-21)."""


def _deposit_anchor_values(
    deposit: Deposit,
    portfolio_currency: str,
    fx_series: pd.Series | None,
    fx_today: float,
) -> tuple[float, float, float]:
    """Compute (native_amount, sgd_amount, fx_native_to_sgd) for a deposit used
    as the synthetic first anchor of a new sleeve.

    Priority:
    1. If ``deposit.fx_to_base`` is provided, use it directly — this is
       Stashaway's exact rate (e.g. 0.7791 for SGD→USD on a specific date),
       readable from the user's transaction history.
    2. Else fall back to Yahoo midmarket FX asof deposit date, then apply a
       small ``_DEPOSIT_FX_SPREAD`` haircut on cross-currency conversions to
       approximate Stashaway's spread.
    """
    p_ccy = portfolio_currency.upper()
    dep_ccy = deposit.currency.upper()

    # Determine the native_amount = deposit_amount × conversion_rate
    if deposit.fx_to_base is not None and deposit.fx_to_base > 0:
        native_amount = deposit.amount * deposit.fx_to_base
    elif dep_ccy == p_ccy:
        native_amount = deposit.amount
    else:
        fx_mid = _fx_asof(fx_series, deposit.date) or fx_today
        if p_ccy == "USD" and dep_ccy == "SGD":
            native_amount = (deposit.amount / fx_mid) * (1.0 - _DEPOSIT_FX_SPREAD)
        elif p_ccy == "SGD" and dep_ccy == "USD":
            native_amount = (deposit.amount * fx_mid) * (1.0 - _DEPOSIT_FX_SPREAD)
        else:
            native_amount = deposit.amount  # unknown currency pair — pass through

    # SGD-equivalent of the deposit (used for sgd_subtext in drilldown)
    if dep_ccy == "SGD":
        sgd_amount = deposit.amount
    elif p_ccy == "SGD":
        sgd_amount = native_amount
    else:
        # deposit currency is USD (or similar) and sleeve is USD-base too
        # — convert USD → SGD via Yahoo midmarket asof deposit date
        fx_at_dep = _fx_asof(fx_series, deposit.date) or fx_today
        sgd_amount = native_amount * fx_at_dep

    fx_native_to_sgd = (sgd_amount / native_amount) if native_amount > 0 else 1.0
    return native_amount, sgd_amount, fx_native_to_sgd


def _price_asof(
    bars: pd.DataFrame | None, ticker: str, target: date
) -> float | None:
    if ticker.startswith("CASH_"):
        return 1.0
    if bars is None or ticker not in bars.index.get_level_values(0):
        return None
    s = bars.xs(ticker, level=0)["close"]
    if s.empty:
        return None
    target_ts = (
        pd.Timestamp(target).tz_localize(s.index.tz)
        if s.index.tz else pd.Timestamp(target)
    )
    s = s[s.index <= target_ts]
    if s.empty:
        return None
    return float(s.iloc[-1])


def _segment_dr_factor(
    weights: dict[str, float],
    holding_meta: dict[str, dict[str, Any]],
    bars: pd.DataFrame | None,
    seg_start: date,
    seg_end: date,
) -> float:
    """Daily-rebalanced return factor over [seg_start, seg_end] with constant *weights*.

    Forward-fills missing dates per ticker (treats non-trading days as 0 return).
    Returns the cumulative factor 1 + r_seg.
    """
    if bars is None or seg_end <= seg_start:
        return 1.0
    yahoo_tickers: dict[str, float] = {}
    for sid, w in weights.items():
        if w <= 0:
            continue
        hm = holding_meta.get(sid)
        if hm is None:
            continue
        yt = hm["ticker"]
        if yt.startswith("CASH_"):
            continue  # zero contribution
        if yt not in bars.index.get_level_values(0):
            continue
        yahoo_tickers[yt] = yahoo_tickers.get(yt, 0.0) + w
    if not yahoo_tickers:
        return 1.0
    closes = []
    cols = []
    for yt in yahoo_tickers:
        s = bars.xs(yt, level=0)["close"]
        target_start = (
            pd.Timestamp(seg_start).tz_localize(s.index.tz)
            if s.index.tz else pd.Timestamp(seg_start)
        )
        target_end = (
            pd.Timestamp(seg_end).tz_localize(s.index.tz)
            if s.index.tz else pd.Timestamp(seg_end)
        )
        # Anchor on the close at-or-before seg_start so the first daily return
        # inside the segment uses the prior trading-day close. Without this,
        # a one-trading-day segment yields no compounding and DR underestimates.
        mask_anchor = s.index <= target_start
        mask_in = (s.index > target_start) & (s.index <= target_end)
        if not mask_anchor.any() or not mask_in.any():
            continue
        anchor = s[mask_anchor].iloc[-1:]
        in_seg = s[mask_in]
        seg_series = pd.concat([anchor, in_seg])
        if len(seg_series) < 2:
            continue
        closes.append(seg_series)
        cols.append(yt)
    if not closes:
        return 1.0
    wide = pd.concat(closes, axis=1, keys=cols).ffill()
    rets = wide.pct_change().dropna(how="all").fillna(0.0)
    if rets.empty:
        return 1.0
    w_vec = pd.Series({c: yahoo_tickers[c] for c in rets.columns})
    daily_port = rets.values @ w_vec.values
    return float(np.prod(1.0 + daily_port))


def _compute_multi_anchor(
    portfolio: Portfolio,
    anchors: list[_Anchor],
    target_series: list[tuple[date, dict[str, float]]] | None,
    bars: pd.DataFrame | None,
    deposits: Sequence[Deposit] | None = None,
    fx_series_usd_sgd: pd.Series | None = None,
) -> dict[str, Any] | None:
    """Compute per-holding and sleeve totals at each anchor date.

    Model: at the first anchor, allocate observed_native × target_weight(anchor0)
    to each holding (fixed shares). Buy-and-hold forward through every segment
    boundary; at target-change dates between anchors, rebalance shares to the
    new target weights against the current modeled sleeve value (NO re-anchor to
    observed — drift between modeled and observed accumulates and is visible at
    each anchor with an observed value).

    DR sleeve: daily-rebalanced compounding with constant weights inside each
    segment, snapping to new weights at target changes; sleeve $ tracked as a
    running value (initialised at observed_native, multiplied by segment factor,
    and bumped by deposit amounts at deposit dates).

    Deposit injection: deposits to ``portfolio`` strictly between the first and
    last anchor are added at their date — BH allocates the deposit across
    holdings at the prevailing target weights and converts to native currency
    via ``fx_series_usd_sgd`` when the deposit currency differs from the sleeve;
    DR adds the native-currency amount to ``sleeve_dr_value`` without altering
    the return factor. The first-anchor's own seed deposit (if the anchor is a
    synthetic deposit-date anchor) is already counted as the basis and is
    excluded automatically by the strict-between filter.
    """
    if not anchors or anchors[0].observed_native is None:
        return None

    from hailmary.allocation.universe import STASHAWAY_UNIVERSE

    # Build sid set as UNION of current portfolio.holdings + every sid appearing
    # in any target snapshot in scope (anchors[0] → anchors[-1]). Sids that have
    # been removed from the current book but were held at earlier anchors still
    # need to appear; otherwise their $ allocation silently evaporates at the
    # first anchor and BH ends up compounding a too-small basis.
    holdings_by_sid: dict[str, Any] = {h.ticker: h for h in portfolio.holdings}
    sid_set: set[str] = set(holdings_by_sid)
    if target_series:
        first_d = anchors[0].snap_date
        last_d = anchors[-1].snap_date
        for d, weights in target_series:
            if first_d <= d <= last_d or d <= first_d:
                sid_set.update(weights.keys())
    sid_list = sorted(sid_set)

    def _meta(sid: str) -> tuple[str, str, str, str | None, str]:
        """Return (yahoo_ticker, asset_class, region, sector, data_source) for sid.

        Prefer the live portfolio.holdings entry; fall back to STASHAWAY_UNIVERSE
        for sids that have been removed from the current book.
        """
        h = holdings_by_sid.get(sid)
        if h is not None:
            return (
                h.metadata.ticker,
                h.metadata.asset_class,
                h.metadata.region,
                h.metadata.sector,
                getattr(h.metadata, "data_source", "real"),
            )
        md = STASHAWAY_UNIVERSE.get(sid)
        if md is not None:
            return (md.ticker, md.asset_class, md.region, md.sector, md.data_source)
        return (sid, "?", "?", None, "real")

    # Initial weights at first anchor
    a0 = anchors[0]
    w0 = (
        _target_weights_at(target_series, a0.snap_date) if target_series
        else None
    )
    if w0 is None:
        w0 = {h.ticker: h.weight for h in portfolio.holdings}

    # Initialize shares per holding from first anchor
    shares: dict[str, float] = {}
    holding_meta: dict[str, dict[str, Any]] = {}
    for sid in sid_list:
        yt, asset_class, region, sector, data_source = _meta(sid)
        w_i = w0.get(sid, 0.0)
        px0 = _price_asof(bars, yt, a0.snap_date)
        dollar0 = a0.observed_native * w_i
        if px0 and px0 > 0:
            shares[sid] = dollar0 / px0
        else:
            shares[sid] = 0.0
        holding_meta[sid] = {
            "ticker": yt,
            "asset_class": asset_class,
            "region": region,
            "sector": sector,
            "data_source": data_source,
            "weights": {a0.snap_date: w_i},
            "prices": {a0.snap_date: px0 if px0 is not None else float("nan")},
            "values": {a0.snap_date: dollar0},
        }

    sleeve_bh: list[float] = [a0.observed_native]
    sleeve_dr_value = a0.observed_native
    sleeve_dr: list[float] = [sleeve_dr_value]
    # Cumulative deposits injected up to each anchor — used by the renderer
    # to compute "return net of deposits" (real investment performance).
    deposits_cum: list[float] = [0.0]

    fee_annual = float(portfolio.metadata.get("management_fee_annual", 0.0))

    target_change_dates = sorted(
        {d for d, _ in (target_series or []) if a0.snap_date < d <= anchors[-1].snap_date}
    )

    # Deposits strictly between first and last anchor, converted to native currency.
    # The first-anchor basis (or synthetic deposit anchor) is excluded by the
    # strict-greater check on a0.snap_date.
    port_currency = portfolio.currency.upper()
    fx_clean = (
        fx_series_usd_sgd.dropna() if fx_series_usd_sgd is not None else None
    )
    if fx_clean is not None and hasattr(fx_clean.index, "tz") and fx_clean.index.tz is not None:
        fx_clean = fx_clean.copy()
        fx_clean.index = fx_clean.index.tz_localize(None)

    def _fx_at(d: date) -> float | None:
        if fx_clean is None or fx_clean.empty:
            return None
        ts = pd.Timestamp(d)
        sub = fx_clean[fx_clean.index <= ts]
        if sub.empty:
            return None
        return float(sub.iloc[-1])

    def _to_native(amount: float, src_ccy: str, at_date: date) -> float:
        src = src_ccy.upper()
        if src == port_currency:
            return amount
        fx = _fx_at(at_date) or 1.0
        if port_currency == "USD" and src == "SGD":
            return amount / fx
        if port_currency == "SGD" and src == "USD":
            return amount * fx
        return amount

    deposits_by_date: dict[date, float] = {}
    for dep in (deposits or []):
        if dep.portfolio != portfolio.name:
            continue
        # Inclusive on the right end: a deposit landing exactly on the last
        # anchor (= end_date for drilldown, = end_snap_date for the top-table
        # model) should be reflected in the model's value at that point.
        # The left end stays strict (a deposit on the FIRST anchor would
        # double-count the basis we already set from the synthetic anchor).
        if not (a0.snap_date < dep.date <= anchors[-1].snap_date):
            continue
        native = _to_native(dep.amount, dep.currency, dep.date)
        deposits_by_date[dep.date] = deposits_by_date.get(dep.date, 0.0) + native
    deposit_dates = sorted(deposits_by_date)

    for i in range(1, len(anchors)):
        prev = anchors[i - 1]
        cur = anchors[i]
        # Sub-segment boundaries: union of target-change and deposit dates between
        # prev and cur (exclusive). Each sub-segment is buy-and-hold inside; at
        # the boundary we may rebalance (target change) and/or inject capital
        # (deposit).
        tc_between = {d for d in target_change_dates if prev.snap_date < d < cur.snap_date}
        # Deposits inclusive on the right end so a deposit landing exactly on
        # an anchor (incl. the hidden 1M anchor) still gets injected; the outer
        # filter already excludes deposits at a0/anchors[-1], so this can't
        # double-count.
        dep_between = {d for d in deposit_dates if prev.snap_date < d <= cur.snap_date}
        sub_boundaries = sorted(tc_between | dep_between)
        boundaries = [prev.snap_date, *sub_boundaries, cur.snap_date]
        for j in range(len(boundaries) - 1):
            seg_start = boundaries[j]
            seg_end = boundaries[j + 1]
            seg_weights = (
                _target_weights_at(target_series, seg_start)
                if target_series else w0
            ) or w0

            # DR compounding for this sub-segment (return factor only; capital
            # injection is handled separately below).
            dr_factor = _segment_dr_factor(
                seg_weights, holding_meta, bars, seg_start, seg_end
            )
            # Pro-rated Stashaway management fee over the sub-segment days.
            if fee_annual > 0:
                seg_days = max((seg_end - seg_start).days, 0)
                fee_factor = 1.0 - fee_annual * seg_days / 365.0
                dr_factor *= fee_factor
                # Apply the same fee to BH shares by scaling them down. BH model
                # tracks shares × prices; multiplying shares by fee_factor is
                # equivalent to deducting the fee from each holding's value.
                for sid in shares:
                    shares[sid] *= fee_factor
            sleeve_dr_value *= dr_factor

            is_target_change = seg_end in tc_between
            is_deposit = seg_end in dep_between

            if is_target_change or is_deposit:
                # Compute sleeve_bh value at the boundary
                sleeve_at_boundary = 0.0
                for sid, sh in shares.items():
                    if sh == 0:
                        continue
                    yt = holding_meta[sid]["ticker"]
                    px_t = _price_asof(bars, yt, seg_end)
                    if px_t is None:
                        continue
                    sleeve_at_boundary += sh * px_t
                if is_deposit:
                    dep_amt = deposits_by_date[seg_end]
                    sleeve_at_boundary += dep_amt
                    sleeve_dr_value += dep_amt

                if is_target_change:
                    # Rebalance ALL shares to new targets × current sleeve value
                    w_new = (
                        _target_weights_at(target_series, seg_end)
                        if target_series else None
                    ) or {}
                    for sid in shares:
                        yt = holding_meta[sid]["ticker"]
                        w_i = w_new.get(sid, 0.0)
                        px_t = _price_asof(bars, yt, seg_end)
                        if px_t and px_t > 0:
                            shares[sid] = sleeve_at_boundary * w_i / px_t
                        else:
                            shares[sid] = 0.0
                else:
                    # Deposit-only boundary: allocate the deposit at prevailing
                    # target weights, leaving existing shares untouched (no
                    # off-deposit rebalancing — Stashaway nudges with deposit cash).
                    w_at_dep = (
                        _target_weights_at(target_series, seg_end)
                        if target_series else seg_weights
                    ) or seg_weights or {}
                    dep_native = deposits_by_date[seg_end]
                    for sid in shares:
                        yt = holding_meta[sid]["ticker"]
                        w_i = w_at_dep.get(sid, 0.0)
                        px_t = _price_asof(bars, yt, seg_end)
                        if px_t and px_t > 0 and w_i > 0:
                            shares[sid] += dep_native * w_i / px_t

        # Record per-holding state at cur anchor
        sleeve_bh_at_cur = 0.0
        w_target_at_cur = (
            _target_weights_at(target_series, cur.snap_date)
            if target_series else None
        ) or {}
        for sid in sid_list:
            yt = holding_meta[sid]["ticker"]
            px_cur = _price_asof(bars, yt, cur.snap_date)
            val = shares[sid] * px_cur if (px_cur is not None) else 0.0
            sleeve_bh_at_cur += val
            holding_meta[sid]["prices"][cur.snap_date] = (
                px_cur if px_cur is not None else float("nan")
            )
            holding_meta[sid]["values"][cur.snap_date] = val
            # Target weight at this anchor (constant within a segment; changes
            # only when the target snapshot itself shifts).
            holding_meta[sid]["weights"][cur.snap_date] = w_target_at_cur.get(sid)
        sleeve_bh.append(sleeve_bh_at_cur)
        sleeve_dr.append(sleeve_dr_value)
        deposits_cum.append(
            sum(amt for d, amt in deposits_by_date.items() if d <= cur.snap_date)
        )

    sleeve_obs = [a.observed_native for a in anchors]
    sleeve_obs_sgd = [a.observed_sgd for a in anchors]
    anchor_fx = [a.fx_native_to_sgd for a in anchors]
    return {
        "anchor_dates": [a.snap_date for a in anchors],
        "currency": portfolio.currency.upper(),
        "holdings": holding_meta,
        "deposits_cum": deposits_cum,
        "sleeve_bh": sleeve_bh,
        "sleeve_dr": sleeve_dr,
        "sleeve_obs": sleeve_obs,
        "sleeve_obs_sgd": sleeve_obs_sgd,
        "anchor_fx": anchor_fx,
    }


def _fmt_pct_signed(v: float | None) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:+.2%}"


def _multi_anchor_sleeve_block(
    name: str,
    view: dict[str, Any],
    goal_url: str | None,
) -> str:
    """Render one sleeve's multi-anchor block: per-holding table + summary rows."""
    all_anchor_dates: list[date] = view["anchor_dates"]
    ccy: str = view["currency"]
    holdings: dict[str, dict[str, Any]] = view["holdings"]
    all_sleeve_bh: list[float] = view["sleeve_bh"]
    all_sleeve_dr: list[float] = view["sleeve_dr"]
    all_sleeve_obs: list[float | None] = view["sleeve_obs"]
    all_sleeve_obs_sgd: list[float | None] = view.get(
        "sleeve_obs_sgd", [None] * len(all_anchor_dates)
    )
    all_anchor_fx: list[float | None] = view.get(
        "anchor_fx", [None] * len(all_anchor_dates)
    )
    all_deposits_cum: list[float] = view.get(
        "deposits_cum", [0.0] * len(all_anchor_dates)
    )
    hidden_dates: set[date] = view.get("hidden_dates", set())

    # Split visible vs hidden indices. Hidden anchors are evaluation-only (e.g.
    # the (end_date - 1M) point we use for true calendar-month deltas); they
    # don't render as columns but feed into the 1M Δ calculation below.
    visible_idx = [i for i, d in enumerate(all_anchor_dates) if d not in hidden_dates]
    anchor_dates: list[date] = [all_anchor_dates[i] for i in visible_idx]
    sleeve_bh: list[float] = [all_sleeve_bh[i] for i in visible_idx]
    sleeve_dr: list[float] = [all_sleeve_dr[i] for i in visible_idx]
    sleeve_obs: list[float | None] = [all_sleeve_obs[i] for i in visible_idx]
    sleeve_obs_sgd: list[float | None] = [all_sleeve_obs_sgd[i] for i in visible_idx]
    anchor_fx: list[float | None] = [all_anchor_fx[i] for i in visible_idx]
    deposits_cum: list[float] = [all_deposits_cum[i] for i in visible_idx]
    total_deposits = deposits_cum[-1] if deposits_cum else 0.0

    # 1M evaluation date (hidden anchor closest to end_date - 1 month)
    one_m_date = next(iter(hidden_dates), None) if hidden_dates else None
    one_m_idx = (
        next((i for i, d in enumerate(all_anchor_dates) if d == one_m_date), None)
        if one_m_date else None
    )

    show_sgd = ccy == "USD"

    def _color(val: float | None) -> str:
        if val is None or (isinstance(val, float) and val != val):
            return ""
        return "pos" if val >= 0 else "neg"

    def _sgd(val: float | None, idx: int) -> float | None:
        if val is None or idx >= len(anchor_fx) or anchor_fx[idx] is None:
            return None
        return val * anchor_fx[idx]

    def _val_cell(native: float | None, idx: int) -> str:
        if native is None or (isinstance(native, float) and native != native):
            return "—"
        nv = f"{native:,.2f}"
        if not show_sgd:
            return nv
        s = _sgd(native, idx)
        if s is None:
            return nv
        return f'{nv}<br><small class="sgd-sub">{s:,.0f} SGD</small>'

    def _signed_dual(native: float | None, sgd: float | None) -> str:
        if native is None or (isinstance(native, float) and native != native):
            return "—"
        cls = _color(native)
        nv = f"{native:+,.0f}"
        if not show_sgd or sgd is None:
            return f'<span class="{cls}">{nv}</span>'
        return (
            f'<span class="{cls}">{nv}</span>'
            f'<br><small class="sgd-sub {cls}">{sgd:+,.0f} SGD</small>'
        )

    name_html = (
        f'<a href="{goal_url}" target="_blank" rel="noopener" class="port-link">{name}</a>'
        if goal_url else f"<strong>{name}</strong>"
    )

    first = sleeve_obs[0]
    last_bh = sleeve_bh[-1]
    # "Net" return = strip out injected deposits so the % reflects actual
    # investment performance, not capital additions. Without this, a $5K
    # deposit into a $2M sleeve shows as +0.25% "growth" even with 0% return.
    cum_return = ((last_bh - total_deposits) / first - 1.0) if first else 0.0
    delta_cls = "pos" if cum_return >= 0 else "neg"

    # Build the "Returns summary" panel: per-segment + cumulative, three lenses
    def _fmt_pair(pct: float | None, dollars: float | None) -> str:
        cls = _color(pct)
        pct_s = _fmt_pct_signed(pct)
        if dollars is None or (isinstance(dollars, float) and dollars != dollars):
            d_s = "—"
        else:
            d_s = f"{dollars:+,.0f}"
        return f'<span class="{cls}">{pct_s}</span> <small>{d_s}</small>'

    # All %/$ figures in this panel are NET OF DEPOSITS — capital injected
    # during a segment is subtracted so the return reflects investment
    # performance, not deposit-driven growth.
    returns_rows: list[str] = []
    for i in range(1, len(anchor_dates)):
        obs0, obs1 = sleeve_obs[i - 1], sleeve_obs[i]
        bh0, bh1 = sleeve_bh[i - 1], sleeve_bh[i]
        dr0, dr1 = sleeve_dr[i - 1], sleeve_dr[i]
        seg_dep = deposits_cum[i] - deposits_cum[i - 1]
        # Observed segment return: (obs_end - obs_start - seg_deposits) / obs_start
        if obs0 and obs1 is not None and obs0 > 0:
            obs_d = obs1 - obs0 - seg_dep
            obs_pct = obs_d / obs0
        else:
            obs_d = None
            obs_pct = None
        # Model segment returns net of deposits injected within the segment
        bh_d = bh1 - bh0 - seg_dep if bh0 > 0 else None
        bh_pct = bh_d / bh0 if (bh_d is not None and bh0 > 0) else None
        dr_d = dr1 - dr0 - seg_dep if dr0 > 0 else None
        dr_pct = dr_d / dr0 if (dr_d is not None and dr0 > 0) else None
        period = f"{anchor_dates[i-1].isoformat()} → {anchor_dates[i].isoformat()}"
        returns_rows.append(
            "<tr>"
            f"<td>{period}</td>"
            f"<td>{_fmt_pair(obs_pct, obs_d)}</td>"
            f"<td>{_fmt_pair(bh_pct, bh_d)}</td>"
            f"<td>{_fmt_pair(dr_pct, dr_d)}</td>"
            "</tr>"
        )
    # Cumulative row — net of total deposits across the whole window
    obs_last_real = None
    obs_last_real_sgd: float | None = None
    obs_last_real_idx: int | None = None
    for _i in range(len(sleeve_obs) - 1, -1, -1):
        if sleeve_obs[_i] is not None:
            obs_last_real = sleeve_obs[_i]
            obs_last_real_sgd = sleeve_obs_sgd[_i] if _i < len(sleeve_obs_sgd) else None
            obs_last_real_idx = _i
            break
    obs_cum_d: float | None
    obs_cum_pct: float | None
    if obs_last_real is not None and first:
        deps_to_last_obs = deposits_cum[obs_last_real_idx] if obs_last_real_idx is not None else total_deposits
        obs_cum_d = obs_last_real - first - deps_to_last_obs
        obs_cum_pct = obs_cum_d / first
    else:
        obs_cum_d = None
        obs_cum_pct = None
    bh_cum_d = last_bh - first - total_deposits
    bh_cum_pct = bh_cum_d / first if first else None
    dr_cum_d = sleeve_dr[-1] - first - total_deposits
    dr_cum_pct = dr_cum_d / first if first else None
    returns_rows.append(
        "<tr class=\"cumulative-row\">"
        f"<td><strong>Cumulative</strong></td>"
        f"<td>{_fmt_pair(obs_cum_pct, obs_cum_d)}</td>"
        f"<td>{_fmt_pair(bh_cum_pct, bh_cum_d)}</td>"
        f"<td>{_fmt_pair(dr_cum_pct, dr_cum_d)}</td>"
        "</tr>"
    )
    returns_panel = (
        '<p class="footnote" style="margin:0 0 4px"><em>Returns below are NET '
        "OF DEPOSITS — capital injected during the period is subtracted so the "
        "% reflects investment performance, not deposit-driven growth.</em></p>"
        '<table class="ds-table returns-summary"><thead><tr>'
        '<th>Period</th><th>Observed</th><th>BH modelled</th><th>DR modelled</th>'
        "</tr></thead><tbody>"
        + "".join(returns_rows)
        + "</tbody></table>"
    )

    summary_line = (
        f"{name_html} · {ccy} {first:,.0f} → "
        f'<span class="{delta_cls}">{last_bh:,.0f}</span> '
        f'(<span class="{delta_cls}">{cum_return:+.2%} BH</span>, '
        f'<span class="{_color(dr_cum_pct)}">{(dr_cum_pct or 0):+.2%} DR</span>) · '
        f"{len(holdings)} holdings"
    )

    # 1M = (end_date - 1 month) → end_date. If we have a hidden anchor at that
    # 1M target date, use its modeled values; otherwise fall back to the first
    # visible anchor (for sleeves whose first anchor is later than 1M ago).
    has_1m = one_m_date is not None or len(anchor_dates) >= 2
    one_m_start_date = one_m_date if one_m_date else anchor_dates[0]
    one_m_end_date = anchor_dates[-1]
    seg1_label = (
        f"{one_m_start_date.isoformat()} → {one_m_end_date.isoformat()}"
        if has_1m else ""
    )

    # Header: Ticker | Asset | Wt | $(d1) ... $(dN) | 1M Δ$ | 1M Δ% | Cum Δ$ | Cum Δ%
    header_cells = [
        '<th>Ticker</th>',
        '<th>Asset · Region</th>',
        '<th>Wt</th>',
    ]
    for d in anchor_dates:
        header_cells.append(f'<th>$ ({d.isoformat()})</th>')
    if has_1m:
        header_cells.append(f'<th>1M Δ$<br><small>({seg1_label})</small></th>')
        header_cells.append('<th>1M Δ%</th>')
    header_cells.append('<th>Cum Δ$</th>')
    header_cells.append('<th>Cum Δ%</th>')

    rows_html: list[str] = []
    sorted_holdings = sorted(
        holdings.items(),
        key=lambda kv: -(kv[1]["values"].get(anchor_dates[0], 0.0) or 0.0),
    )
    weight_change_seen = False
    for sid, hm in sorted_holdings:
        ticker = hm["ticker"]
        ds = hm.get("data_source", "real")
        if ds == "synthetic":
            ticker = f"{ticker} 🟡synth"
        elif ds == "proxy":
            ticker = f"{ticker} 🟠proxy"
        label = f'{hm["asset_class"]} · {hm["region"]}'

        # Target weight: show latest anchor's weight. Flag if it changed across anchors.
        weights_seq = [hm["weights"].get(d) for d in anchor_dates]
        weights_present = [w for w in weights_seq if w is not None]
        wt_latest = weights_present[-1] if weights_present else None
        wt_changed = (
            len(set(round(w, 4) for w in weights_present)) > 1
            if len(weights_present) > 1 else False
        )
        if wt_changed:
            weight_change_seen = True
        wt_marker = "*" if wt_changed else ""
        wt_s = f"{wt_latest:.2%}{wt_marker}" if wt_latest is not None else "—"

        cells = [
            f"<td>{ticker}</td>",
            f"<td>{label}</td>",
            f"<td>{wt_s}</td>",
        ]
        first_val = hm["values"].get(anchor_dates[0])
        last_val = hm["values"].get(anchor_dates[-1])
        for idx, d in enumerate(anchor_dates):
            v = hm["values"].get(d)
            cells.append(f"<td>{_val_cell(v, idx)}</td>")
        # 1M Δ = $(end_date) - $(end_date - 1M). Use the hidden 1M anchor's
        # modeled values if present; else fall back to first visible anchor.
        if has_1m:
            # 1M Δ$ per holding = pure price return × $_value_at_1M_start.
            # Excludes any deposit-allocated shares added during the window so
            # the column reflects the underlying's price contribution, not
            # capital additions (consistent with the footer's net-of-deposits
            # 1M row).
            v0 = hm["values"].get(one_m_start_date)
            px0 = hm["prices"].get(one_m_start_date)
            px1 = hm["prices"].get(one_m_end_date)
            v0_fx_idx = one_m_idx if one_m_idx is not None else visible_idx[0]
            v1_fx_idx = visible_idx[-1]
            valid_px = (
                isinstance(px0, (int, float)) and isinstance(px1, (int, float))
                and not (isinstance(px0, float) and px0 != px0)
                and not (isinstance(px1, float) and px1 != px1)
                and px0 > 0
            )
            if v0 is not None and valid_px:
                seg1_pct = (px1 / px0) - 1.0
                seg1_d = v0 * seg1_pct
            else:
                seg1_pct = None
                seg1_d = None
            # SGD subtext for the 1M $ delta — apply end-anchor FX
            if seg1_d is not None and all_anchor_fx[v1_fx_idx] is not None:
                seg1_d_sgd = seg1_d * all_anchor_fx[v1_fx_idx]
            else:
                seg1_d_sgd = None
            cells.append(f"<td>{_signed_dual(seg1_d, seg1_d_sgd)}</td>")
            cls = _color(seg1_pct)
            seg1_pct_s = f"{seg1_pct:+.2%}" if seg1_pct is not None else "—"
            cells.append(f'<td class="{cls}">{seg1_pct_s}</td>')
        # Cum Δ$ per holding = pure price return × $_value_at_first_anchor.
        # Same rationale as 1M — excludes deposit-allocated shares.
        first_px = hm["prices"].get(anchor_dates[0])
        last_px = hm["prices"].get(anchor_dates[-1])
        valid_cum_px = (
            isinstance(first_px, (int, float)) and isinstance(last_px, (int, float))
            and not (isinstance(first_px, float) and first_px != first_px)
            and not (isinstance(last_px, float) and last_px != last_px)
            and first_px > 0
        )
        if first_val is not None and valid_cum_px:
            cum_ret = (last_px / first_px) - 1.0
            cum_delta = first_val * cum_ret
        else:
            cum_ret = None
            cum_delta = None
        cum_delta_sgd = (
            (cum_delta * (all_anchor_fx[-1] or 1.0))
            if (cum_delta is not None and all_anchor_fx[-1] is not None) else None
        )
        cls = _color(cum_ret)
        cum_ret_s = f"{cum_ret:+.2%}" if cum_ret is not None else "—"
        cells.append(f"<td>{_signed_dual(cum_delta, cum_delta_sgd)}</td>")
        cells.append(f'<td class="{cls}">{cum_ret_s}</td>')
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    # Footer note for target-change marker
    wt_change_note = (
        '<p class="footnote"><code>*</code> = target weight changed across the '
        "shown anchors; cell shows the latest target. Expand the methodology section "
        "for the full per-anchor weight series if needed.</p>"
        if weight_change_seen else ""
    )

    # Sleeve summary: 5 rows × N anchor columns + 1M cols + cum cols
    summary_rows: list[str] = []

    def _row(
        label: str,
        per_anchor: list[str],
        cls: str = "",
        seg1_d: str = "",
        seg1_p: str = "",
        cum_d: str = "",
        cum_p: str = "",
    ) -> str:
        cls_attr = f' class="{cls}"' if cls else ""
        cells = [f'<th colspan="3">{label}</th>']
        for v in per_anchor:
            cells.append(f"<td>{v}</td>")
        if has_1m:
            cells.append(f"<td>{seg1_d}</td>")
            cells.append(f"<td>{seg1_p}</td>")
        cells.append(f"<td>{cum_d}</td>")
        cells.append(f"<td>{cum_p}</td>")
        return f"<tr{cls_attr}>" + "".join(cells) + "</tr>"

    def _pct_cell(v: float | None) -> str:
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        return f'<span class="{_color(v)}">{v:+.2%}</span>'

    # Per-anchor value cells with native + SGD subtext (for USD sleeves)
    obs_anchor = [_val_cell(v, i) for i, v in enumerate(sleeve_obs)]
    bh_anchor = [_val_cell(v, i) for i, v in enumerate(sleeve_bh)]
    dr_anchor = [_val_cell(v, i) for i, v in enumerate(sleeve_dr)]

    def _drift_cell(native: float | None, sgd: float | None) -> str:
        return _signed_dual(native, sgd) if native is not None else "—"

    drift_mva_anchor: list[str] = []
    for i, (b, o) in enumerate(zip(sleeve_bh, sleeve_obs)):
        if o is None:
            drift_mva_anchor.append("—")
            continue
        d_native = b - o
        d_sgd = (_sgd(b, i) or 0) - (sleeve_obs_sgd[i] or 0) if sleeve_obs_sgd[i] is not None else None
        drift_mva_anchor.append(_drift_cell(d_native, d_sgd))

    drift_bd_anchor: list[str] = []
    for i, (b, dv) in enumerate(zip(sleeve_bh, sleeve_dr)):
        d_native = b - dv
        d_sgd_b = _sgd(b, i)
        d_sgd_d = _sgd(dv, i)
        d_sgd = (d_sgd_b - d_sgd_d) if (d_sgd_b is not None and d_sgd_d is not None) else None
        drift_bd_anchor.append(_drift_cell(d_native, d_sgd))

    # 1M segment: (end_date - 1 month) → end_date, using the hidden eval anchor
    def _seg1_from(values_all: list[Any]) -> tuple[Any, Any]:
        """Return (v_start, v_end) for the 1M segment from a full-anchor list."""
        if one_m_idx is not None:
            return values_all[one_m_idx], values_all[-1]
        if len(values_all) >= 2:
            return values_all[0], values_all[-1]
        return None, None

    # 1M deposits = total deposits injected within the 1M window. We subtract
    # this from the sleeve-total deltas so the footer's 1M numbers reflect
    # actual investment performance (consistent with the Cumulative row).
    one_m_idx_full = one_m_idx if one_m_idx is not None else 0
    deps_in_1m = all_deposits_cum[-1] - all_deposits_cum[one_m_idx_full]

    def _seg1_deltas(values_all: list[float | None]) -> tuple[float | None, float | None]:
        v0, v1 = _seg1_from(values_all)
        if v0 is None or v1 is None:
            return None, None
        delta = v1 - v0 - deps_in_1m
        pct = (delta / v0) if (isinstance(v0, (int, float)) and v0 > 0) else None
        return delta, pct

    def _seg1_sgd(values_all: list[float | None]) -> float | None:
        v0, v1 = _seg1_from(values_all)
        i0 = one_m_idx if one_m_idx is not None else 0
        i1 = len(all_anchor_dates) - 1
        if v0 is None or v1 is None:
            return None
        fx0 = all_anchor_fx[i0] if i0 < len(all_anchor_fx) else None
        fx1 = all_anchor_fx[i1] if i1 < len(all_anchor_fx) else None
        if fx0 is None or fx1 is None:
            return None
        # Convert each side to SGD then subtract deposits-in-1M (in SGD-equiv).
        # We approximate the deposits' SGD amount with the end FX since deposits
        # are typically same-day in SGD; refinement would require per-deposit FX
        # tracking but the impact is small (<5bp on cross-currency sleeves).
        return (v1 * fx1) - (v0 * fx0) - deps_in_1m * fx1

    obs_1m_d, obs_1m_p = _seg1_deltas(all_sleeve_obs)
    bh_1m_d, bh_1m_p = _seg1_deltas(all_sleeve_bh)
    dr_1m_d, dr_1m_p = _seg1_deltas(all_sleeve_dr)

    obs_1m_sgd = _seg1_sgd(all_sleeve_obs)
    bh_1m_sgd = _seg1_sgd(all_sleeve_bh)
    dr_1m_sgd = _seg1_sgd(all_sleeve_dr)

    obs_cum_sgd = None
    if obs_last_real_sgd is not None and sleeve_obs_sgd[0] is not None:
        obs_cum_sgd = obs_last_real_sgd - sleeve_obs_sgd[0]
    bh_cum_sgd = (
        (_sgd(sleeve_bh[-1], len(anchor_dates) - 1) or 0)
        - (_sgd(sleeve_bh[0], 0) or 0)
    )
    dr_cum_sgd = (
        (_sgd(sleeve_dr[-1], len(anchor_dates) - 1) or 0)
        - (_sgd(sleeve_dr[0], 0) or 0)
    )
    summary_rows.append(_row(
        "Observed (app)", obs_anchor, "obs-row",
        seg1_d=_signed_dual(obs_1m_d, obs_1m_sgd) if has_1m else "",
        seg1_p=_pct_cell(obs_1m_p),
        cum_d=_signed_dual(obs_cum_d, obs_cum_sgd) if obs_cum_d is not None else "—",
        cum_p=_pct_cell(obs_cum_pct),
    ))
    summary_rows.append(_row(
        "BH modelled", bh_anchor, "bh-row",
        seg1_d=_signed_dual(bh_1m_d, bh_1m_sgd) if has_1m else "",
        seg1_p=_pct_cell(bh_1m_p),
        cum_d=_signed_dual(bh_cum_d, bh_cum_sgd),
        cum_p=_pct_cell(bh_cum_pct),
    ))
    summary_rows.append(_row(
        "DR modelled", dr_anchor, "dr-row",
        seg1_d=_signed_dual(dr_1m_d, dr_1m_sgd) if has_1m else "",
        seg1_p=_pct_cell(dr_1m_p),
        cum_d=_signed_dual(dr_cum_d, dr_cum_sgd),
        cum_p=_pct_cell(dr_cum_pct),
    ))
    summary_rows.append(_row(
        "Drift: BH − Observed", drift_mva_anchor, "drift-mva-row",
        cum_d=_signed_dual(
            (bh_cum_d - obs_cum_d) if obs_cum_d is not None else None,
            None,
        ),
    ))
    summary_rows.append(_row(
        "Drift: BH − DR (rebal effect)", drift_bd_anchor, "drift-bd-row",
        cum_d=_signed_dual(bh_cum_d - dr_cum_d, None),
    ))

    return (
        "<details class=\"port-block\">"
        f"<summary>{summary_line}</summary>"
        + returns_panel
        + '<table class="ds-table multi-anchor-table"><thead>'
        + "<tr>" + "".join(header_cells) + "</tr>"
        + "</thead><tbody>"
        + "".join(rows_html)
        + "</tbody><tfoot>"
        + "".join(summary_rows)
        + "</tfoot></table>"
        + wt_change_note
        + "</details>"
    )


def _multi_anchor_drilldown_to_html(
    portfolios: Sequence[Portfolio],
    *,
    observed_by_date: dict[date, dict[str, _ObservedSnapshot]],
    end_date: date,
    price_source: Any | None,
    target_weights_by_portfolio: dict[
        str, list[tuple[date, dict[str, float]]]
    ] | None,
    deposits: Sequence[Deposit] | None = None,
    fx_series_usd_sgd: pd.Series | None = None,
    links_path: Path | str = Path("data/holding.xlsx"),
) -> str:
    """Render the multi-anchor drill-down for the reconcile report."""
    from hailmary.allocation.diagnostic import _portfolio_goal_links

    holdings_books = sorted(
        [p for p in portfolios if Role.HOLDING in p.roles],
        key=lambda p: -p.total_value,
    )
    if not holdings_books or not observed_by_date:
        return "<p>No portfolios to drill into.</p>"

    snap_dates = sorted(d for d in observed_by_date if d <= end_date)
    if not snap_dates:
        return "<p>No PortfolioValue snapshots before end_date.</p>"

    # Pre-fetch all bars in one shot. Include current holdings + any sids that
    # appear in target snapshots but were removed from the current book
    # (otherwise their $ allocation evaporates at earlier anchors).
    from hailmary.allocation.universe import STASHAWAY_UNIVERSE as _UNIV
    earliest = snap_dates[0]
    fetch_start = earliest - timedelta(days=7)
    extra_sids: set[str] = set()
    if target_weights_by_portfolio:
        for series in target_weights_by_portfolio.values():
            for _d, weights in series:
                extra_sids.update(weights.keys())
    all_symbols = sorted({
        h.metadata.ticker
        for p in holdings_books
        for h in p.holdings
        if not h.metadata.ticker.startswith("CASH_")
    } | {
        _UNIV[sid].ticker for sid in extra_sids
        if sid in _UNIV and not _UNIV[sid].ticker.startswith("CASH_")
    })
    bars: pd.DataFrame | None = None
    if all_symbols and price_source is not None:
        try:
            bars = price_source.get_bars(all_symbols, fetch_start, end_date)
        except Exception as exc:  # pragma: no cover — provider failure
            warnings.warn(f"Could not fetch price bars: {exc}", stacklevel=2)

    # Build first-deposit lookup per portfolio (for new-sleeve anchor injection)
    first_deposit_by_port: dict[str, Deposit] = {}
    for d in (deposits or []):
        prev = first_deposit_by_port.get(d.portfolio)
        if prev is None or d.date < prev.date:
            first_deposit_by_port[d.portfolio] = d

    fx_today = (
        float(fx_series_usd_sgd.dropna().iloc[-1])
        if fx_series_usd_sgd is not None and not fx_series_usd_sgd.empty
        else 1.0
    )

    goal_urls = _portfolio_goal_links(portfolios, links_path=links_path)

    intro = (
        '<p class="footnote">'
        "Each sleeve expands to a multi-anchor timeline. <strong>Anchor dates</strong> come "
        "from <code>PortfolioValue</code> snapshots plus the trailing end date; new sleeves "
        "with no first-snapshot observation get a synthetic anchor at their first deposit. "
        "<strong>Wt</strong> = latest target weight (marked <code>*</code> if it shifted across "
        "anchors). <strong>$ ({date})</strong> columns = BH-compounded value at each anchor "
        "(fixed shares × price; weights drift inside segments). "
        "<strong>Cum Δ%</strong> = per-holding cumulative return across the shown anchors. "
        "Footer rows: <strong>Observed</strong> from the app; <strong>BH modelled</strong> "
        "from buy-and-hold; <strong>DR modelled</strong> from daily-rebalanced compounding. "
        "<strong>Drift: BH − Observed</strong> measures model accuracy at each checkpoint; "
        "<strong>Drift: BH − DR</strong> is the rebalancing-assumption effect.</p>"
    )
    sections: list[str] = [intro]

    for p in holdings_books:
        target_series = (
            _resolve_target_series(target_weights_by_portfolio.get(p.name))
            if target_weights_by_portfolio else None
        )

        # Build candidate anchors from observed snapshots
        anchors: list[_Anchor] = []
        for d in snap_dates:
            snap = observed_by_date.get(d, {}).get(p.name)
            if snap and snap.base_value > 0:
                fx = (snap.sgd_value / snap.base_value) if snap.base_value > 0 else None
                anchors.append(_Anchor(
                    snap_date=d,
                    observed_native=snap.base_value,
                    observed_sgd=snap.sgd_value,
                    fx_native_to_sgd=fx,
                ))
            else:
                anchors.append(_Anchor(snap_date=d, observed_native=None))
        # Trailing "today" anchor — model endpoint, no observation
        if end_date not in snap_dates:
            # Use today's FX (or 1.0 for SGD-base) as the conversion rate for the endpoint
            endpoint_fx = fx_today if p.currency.upper() == "USD" else 1.0
            anchors.append(_Anchor(
                snap_date=end_date,
                observed_native=None,
                fx_native_to_sgd=endpoint_fx,
            ))

        # If the earliest anchor is zero/None but the sleeve has a first deposit
        # between adjacent anchors, inject a synthetic anchor at deposit date.
        first_real_idx = next(
            (i for i, a in enumerate(anchors) if a.observed_native is not None), None
        )
        dep = first_deposit_by_port.get(p.name)
        if first_real_idx is not None and first_real_idx > 0 and dep is not None:
            prev_anchor_date = anchors[first_real_idx - 1].snap_date
            next_anchor_date = anchors[first_real_idx].snap_date
            if prev_anchor_date <= dep.date < next_anchor_date and dep.amount > 0:
                native_amount, sgd_amount, fx_dep = _deposit_anchor_values(
                    dep, p.currency, fx_series_usd_sgd, fx_today
                )
                synthetic = _Anchor(
                    snap_date=dep.date,
                    observed_native=native_amount,
                    observed_sgd=sgd_amount,
                    fx_native_to_sgd=fx_dep,
                )
                anchors = anchors[first_real_idx:]
                anchors.insert(0, synthetic)

        # Drop any leading anchors that still don't have an observation
        while anchors and anchors[0].observed_native is None:
            anchors.pop(0)
        if not anchors:
            sections.append(
                f'<details class="port-block"><summary>{p.name} — no observed or deposit anchor</summary></details>'
            )
            continue

        # Inject a HIDDEN evaluation point at (end_date - 1 month) so we can
        # compute a true calendar-month return per sleeve / holding. It only
        # records values; it doesn't trigger any rebalancing.
        try:
            from dateutil.relativedelta import relativedelta
            one_month_target = end_date - relativedelta(months=1)
        except ImportError:
            one_month_target = end_date - timedelta(days=30)
        hidden_dates: set[date] = set()
        if anchors[0].snap_date < one_month_target < anchors[-1].snap_date:
            # Find insertion point (sorted)
            insert_idx = next(
                (i for i, a in enumerate(anchors) if a.snap_date > one_month_target),
                len(anchors),
            )
            if not any(a.snap_date == one_month_target for a in anchors):
                # FX at the 1M anchor — needed for the SGD-subtext on 1M deltas.
                # Use fx_series_usd_sgd lookup (asof or forward-fill) for USD sleeves;
                # SGD sleeves use 1.0.
                if p.currency.upper() == "SGD":
                    fx_at_1m: float | None = 1.0
                elif fx_series_usd_sgd is not None and not fx_series_usd_sgd.empty:
                    ts = pd.Timestamp(one_month_target)
                    fx_clean = fx_series_usd_sgd.dropna()
                    if fx_clean.index.tz is not None:
                        ts = ts.tz_localize(fx_clean.index.tz)
                    sub = fx_clean[fx_clean.index <= ts]
                    fx_at_1m = float(sub.iloc[-1]) if not sub.empty else None
                else:
                    fx_at_1m = None
                anchors.insert(
                    insert_idx,
                    _Anchor(
                        snap_date=one_month_target,
                        observed_native=None,
                        fx_native_to_sgd=fx_at_1m,
                    ),
                )
                hidden_dates.add(one_month_target)

        view = _compute_multi_anchor(
            p, anchors, target_series, bars,
            deposits=deposits, fx_series_usd_sgd=fx_series_usd_sgd,
        )
        if view is None:
            sections.append(
                f'<details class="port-block"><summary>{p.name} — could not model</summary></details>'
            )
            continue
        view["hidden_dates"] = hidden_dates
        sections.append(
            _multi_anchor_sleeve_block(p.name, view, goal_urls.get(p.name))
        )

    return "".join(sections)


def _resolve_target_series(
    entry: dict[str, float] | list[tuple[date, dict[str, float]]] | None,
) -> list[tuple[date, dict[str, float]]] | None:
    if not entry:
        return None
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict):
        return [(date.today(), entry)]
    return None


def _target_weights_at(
    series: list[tuple[date, dict[str, float]]] | None,
    as_of: date,
) -> dict[str, float] | None:
    if not series:
        return None
    for d, w in sorted(series, key=lambda x: x[0], reverse=True):
        if d <= as_of:
            return w
    return None
