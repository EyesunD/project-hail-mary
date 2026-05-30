"""Generate the three allocation notebooks via nbformat.

Idempotent: re-running overwrites notebooks/01_validate_holdings.ipynb,
notebooks/02_validate_returns.ipynb, notebooks/03_allocation_diagnostic.ipynb.
Outputs are not pre-executed here; users open them in JupyterLab.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def write_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Notebook 01 — validate holdings
# ---------------------------------------------------------------------------

NB01 = [
    md(
        "# 01 — Validate parsed holdings\n"
        "\n"
        "Parse the latest Stashaway statement, apply role tags from `book_config`,\n"
        "and check that every parsed portfolio reconciles cleanly against the\n"
        "weight-sum tolerance and resolves against the universe map.\n"
        "\n"
        "**Inputs:** `data/statements/<latest>.pdf`\n"
        "\n"
        "**Outputs:** a summary table; a per-portfolio holdings table; a list of any\n"
        "unmapped tickers that need adding to `STASHAWAY_UNIVERSE`.\n"
    ),
    code(
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "\n"
        "from hailmary.allocation.book_config import ROLES\n"
        "from hailmary.allocation.portfolios import Role, from_parsed\n"
        "from hailmary.allocation.statements import parse_statement\n"
        "\n"
        "STATEMENT_PATH = Path('../../data/statements/2026-04 StashAway Monthly Statement.pdf')\n"
        "assert STATEMENT_PATH.exists(), f'Statement not found at {STATEMENT_PATH}'"
    ),
    md("## Parse the statement"),
    code(
        "parsed = parse_statement(STATEMENT_PATH, use_cache=False)\n"
        "print(f'Parsed {len(parsed)} portfolios from {STATEMENT_PATH.name}')\n"
        "print(f'Statement date: {parsed[0].statement_date}')"
    ),
    md(
        "## Reconcile weights\n"
        "\n"
        "Each portfolio's weights should sum to 1.0 ± 1e-4. If any fail, the parser\n"
        "raised inside `parse_statement`; this cell is a final visual check."
    ),
    code(
        "rows = [\n"
        "    {\n"
        "        'name': pf.name,\n"
        "        'currency': pf.currency,\n"
        "        'total_value': pf.total_value,\n"
        "        'n_holdings': len(pf.holdings),\n"
        "        'weight_sum': sum(h.weight for h in pf.holdings),\n"
        "    }\n"
        "    for pf in parsed\n"
        "]\n"
        "summary = pd.DataFrame(rows).sort_values('total_value', ascending=False)\n"
        "summary"
    ),
    md(
        "## Apply role tags and resolve universe\n"
        "\n"
        "`from_parsed` looks each holding up in `STASHAWAY_UNIVERSE`. An\n"
        "`UnknownAssetError` here means a ticker needs adding to the map."
    ),
    code(
        "portfolios = []\n"
        "missing_roles = []\n"
        "for pf in parsed:\n"
        "    roles = ROLES.get(pf.name)\n"
        "    if roles is None:\n"
        "        missing_roles.append(pf.name)\n"
        "        continue\n"
        "    portfolios.append(from_parsed(pf, roles=roles))\n"
        "\n"
        "if missing_roles:\n"
        "    print('Portfolios with no role tag (edit book_config.ROLES):')\n"
        "    for n in missing_roles:\n"
        "        print(f'  - {n!r}')\n"
        "else:\n"
        "    print(f'All {len(portfolios)} portfolios resolved against the universe map.')"
    ),
    md("## Diagnostic vs hidden portfolios"),
    code(
        "diag_rows = [\n"
        "    {\n"
        "        'name': p.name,\n"
        "        'roles': ','.join(sorted(r.value for r in p.roles)),\n"
        "        'currency': p.currency,\n"
        "        'total_value': p.total_value,\n"
        "        'n_holdings': len(p.holdings),\n"
        "    }\n"
        "    for p in portfolios\n"
        "]\n"
        "diag = pd.DataFrame(diag_rows)\n"
        "in_diag = diag[diag['roles'].str.contains('holding')]\n"
        "hidden = diag[~diag['roles'].str.contains('holding')]\n"
        "print(f'In diagnostic ({len(in_diag)}):')\n"
        "display(in_diag.sort_values('total_value', ascending=False))\n"
        "print(f'Hidden ({len(hidden)}):')\n"
        "display(hidden)"
    ),
    md("## Detailed holdings — first three portfolios"),
    code(
        "for p in portfolios[:3]:\n"
        "    print(f'\\n=== {p.name} ({p.currency}) — total {p.total_value:,.2f} ===')\n"
        "    rows = [\n"
        "        {\n"
        "            'stashaway_id': h.stashaway_id,\n"
        "            'yahoo_ticker': h.metadata.ticker,\n"
        "            'asset_class': h.metadata.asset_class,\n"
        "            'region': h.metadata.region,\n"
        "            'sector': h.metadata.sector,\n"
        "            'weight': h.weight,\n"
        "            'value': h.value,\n"
        "        }\n"
        "        for h in p.holdings\n"
        "    ]\n"
        "    display(pd.DataFrame(rows))"
    ),
    md(
        "## Final assertion\n"
        "\n"
        "If any of these fail, the rest of the diagnostic pipeline can't trust the\n"
        "input — fix before running notebooks 02 and 03."
    ),
    code(
        "from hailmary.allocation.portfolios import Role\n"
        "\n"
        "assert len(portfolios) == 15, f'Expected 15 portfolios, got {len(portfolios)}'\n"
        "for p in portfolios:\n"
        "    assert abs(sum(h.weight for h in p.holdings) - 1.0) < 1e-4, p.name\n"
        "managed = [p for p in portfolios if Role.MANAGED_BENCHMARK in p.roles]\n"
        "assert len(managed) >= 1, 'No MANAGED_BENCHMARK portfolios — diagnostic will skip benchmark deltas'\n"
        "print('All checks passed — proceed to notebook 02 / 03.')"
    ),
]

# ---------------------------------------------------------------------------
# Notebook 02 — validate returns
# ---------------------------------------------------------------------------

NB02 = [
    md(
        "# 02 — Reconstruct portfolio returns\n"
        "\n"
        "Pull historical returns for every tradeable ticker via `YahooFinanceProvider`,\n"
        "then reconstruct each portfolio's daily return series using *current weights\n"
        "× historical underlying-asset returns* (design D2).\n"
        "\n"
        "**Caveat (forward-looking, not realised):** the reconstructed series is what\n"
        "*today's* book *would have* returned over history — it's not a real track\n"
        "record. Stashaway statements don't expose a NAV history we could reconcile\n"
        "against, so this notebook is a sanity check that returns reconstruction works,\n"
        "not a backtest.\n"
    ),
    code(
        "from datetime import date\n"
        "from pathlib import Path\n"
        "import warnings\n"
        "\n"
        "import pandas as pd\n"
        "import plotly.graph_objects as go\n"
        "\n"
        "from hailmary.allocation.book_config import ROLES\n"
        "from hailmary.allocation.portfolios import Role, from_parsed\n"
        "from hailmary.allocation.returns import portfolio_returns\n"
        "from hailmary.allocation.statements import parse_statement\n"
        "from hailmary.data.providers import YahooFinanceProvider\n"
        "from hailmary.viz.theme import apply_theme\n"
        "\n"
        "STATEMENT_PATH = Path('../../data/statements/2026-04 StashAway Monthly Statement.pdf')\n"
        "START = date(2022, 1, 1)\n"
        "END = date.today()"
    ),
    md("## Parse + resolve"),
    code(
        "parsed = parse_statement(STATEMENT_PATH, use_cache=False)\n"
        "portfolios = [\n"
        "    from_parsed(pf, roles=ROLES[pf.name])\n"
        "    for pf in parsed if pf.name in ROLES\n"
        "]\n"
        "holding = [p for p in portfolios if Role.HOLDING in p.roles]\n"
        "print(f'{len(portfolios)} portfolios resolved, {len(holding)} tagged HOLDING')"
    ),
    md(
        "## Fetch returns for the diagnostic universe\n"
        "\n"
        "Pull every Yahoo-resolvable ticker across HOLDING portfolios in one call so\n"
        "the cache key is shared across portfolios."
    ),
    code(
        "provider = YahooFinanceProvider()\n"
        "tickers = sorted({\n"
        "    h.metadata.ticker for p in holding for h in p.holdings\n"
        "    if not h.metadata.ticker.startswith('CASH_')\n"
        "})\n"
        "print(f'Fetching {len(tickers)} unique tickers from Yahoo ({START} → {END})')\n"
        "returns = provider.get_returns(tickers, START, END)\n"
        "print(f'Returns panel: {returns.shape[0]} dates × {returns.shape[1]} tickers')\n"
        "missing = sorted(set(tickers) - set(returns.columns))\n"
        "if missing:\n"
        "    print(f'\\nTickers with NO Yahoo data: {missing}')\n"
        "    print('These holdings will be excluded from return reconstruction; consider mapping to a different proxy in universe.py.')"
    ),
    md("## Reconstruct one return series per portfolio"),
    code(
        "import warnings\n"
        "with warnings.catch_warnings():\n"
        "    warnings.simplefilter('ignore', UserWarning)\n"
        "    series = []\n"
        "    failed = []\n"
        "    for p in holding:\n"
        "        try:\n"
        "            s = portfolio_returns(p, returns=returns)\n"
        "            series.append(s)\n"
        "        except Exception as exc:\n"
        "            failed.append((p.name, str(exc)))\n"
        "\n"
        "panel = pd.concat(series, axis=1) if series else pd.DataFrame()\n"
        "print(f'Reconstructed {len(series)} portfolio return series across {panel.shape[0]} dates')\n"
        "if failed:\n"
        "    print('\\nFailed:')\n"
        "    for n, e in failed:\n"
        "        print(f'  {n}: {e}')"
    ),
    md("## Summary stats"),
    code(
        "from hailmary.analytics.metrics import PerformanceMetrics\n"
        "stats = []\n"
        "common = panel.dropna(how='any')\n"
        "for col in common.columns:\n"
        "    m = PerformanceMetrics(common[col])\n"
        "    stats.append({\n"
        "        'portfolio': col,\n"
        "        'ann_return': m.annualised_return,\n"
        "        'ann_vol': m.annualised_vol,\n"
        "        'sharpe': m.sharpe,\n"
        "        'max_dd': m.max_drawdown,\n"
        "    })\n"
        "stats_df = pd.DataFrame(stats).set_index('portfolio').sort_values('sharpe', ascending=False)\n"
        "stats_df.style.format({\n"
        "    'ann_return': '{:.2%}', 'ann_vol': '{:.2%}', 'sharpe': '{:.2f}', 'max_dd': '{:.2%}'\n"
        "})"
    ),
    md("## Cumulative-return chart"),
    code(
        "cum = (1 + common).cumprod() - 1\n"
        "fig = go.Figure()\n"
        "for col in cum.columns:\n"
        "    fig.add_trace(go.Scatter(x=cum.index, y=cum[col], mode='lines', name=col))\n"
        "fig.update_layout(yaxis_tickformat='.0%')\n"
        "apply_theme(fig, title='Reconstructed cumulative returns', height=520)"
    ),
    md(
        "## Tracking-error placeholder\n"
        "\n"
        "Stashaway statements don't expose a NAV history per portfolio, so we can't\n"
        "compute a real tracking error against the parser's reconstruction. If a NAV\n"
        "feed becomes available later, plot `(reconstructed - actual)` here."
    ),
]

# ---------------------------------------------------------------------------
# Notebook 03 — full diagnostic + HTML export
# ---------------------------------------------------------------------------

NB03 = [
    md(
        "# 03 — Allocation diagnostic\n"
        "\n"
        "Run all five diagnostic sections (combined-book exposure, correlation,\n"
        "redundancy, risk contribution, benchmark comparison) on the resolved book\n"
        "and export `reports/allocation_diagnostic.html`.\n"
    ),
    code(
        "from datetime import date\n"
        "from pathlib import Path\n"
        "import warnings\n"
        "\n"
        "import pandas as pd\n"
        "\n"
        "from hailmary.allocation.book_config import MGMT_FEES_ANNUAL, ROLES\n"
        "from hailmary.allocation.diagnostic import (\n"
        "    benchmark_comparison, combined_exposure, combined_exposure_figure,\n"
        "    correlation_figure, correlation_matrix, redundancy_pairs,\n"
        "    render_html_report, risk_contribution,\n"
        ")\n"
        "from hailmary.allocation.portfolios import Role, from_parsed\n"
        "from hailmary.allocation.statements import parse_statement\n"
        "from hailmary.data.providers import YahooFinanceProvider\n"
        "\n"
        "from hailmary.allocation.returns import last_business_day_on_or_before\n"
        "\n"
        "STATEMENT_PATH = Path('../../data/statements/2026-04 StashAway Monthly Statement.pdf')\n"
        "REPORT_PATH = Path('../../reports/allocation_diagnostic.html')\n"
        "START = date(2022, 1, 1)\n"
        "END = last_business_day_on_or_before(date.today())\n"
        "REDUNDANCY_THRESHOLD = 0.85\n"
        "\n"
        "# ALIGN_WINDOW = True  (default, recommended)\n"
        "#   Combined-book metrics + windowed table use the common-history window\n"
        "#   (latest first-data date across all HOLDING portfolios). Apples-to-apples,\n"
        "#   shorter history, static weights.\n"
        "# ALIGN_WINDOW = False\n"
        "#   Full available history with dynamic per-timestep weight renormalisation.\n"
        "#   Longer history but early dates use only the older portfolios at boosted\n"
        "#   weights, so 'early book' ≠ 'today's book'. Useful if you want pre-2024\n"
        "#   context at the cost of mixing weight regimes.\n"
        "ALIGN_WINDOW = True\n"
        "\n"
        "# Annual-return target for the traffic-light styling on the benchmark table's\n"
        "# 'Ann. return' column. Green ≥ target, yellow [0..target), red < 0.\n"
        "# High-Sharpe-low-return rows (e.g. Simple SGD with Sharpe ~4 and ann return 1.5%)\n"
        "# will be yellow under a 5% target — they're risk-adjusted-great but won't grow\n"
        "# wealth at your target rate.\n"
        "TARGET_ANN_RETURN = 0.05\n"
        "\n"
        "# Date the portfolio-reconciliation section projects to. Defaults to the last\n"
        "# business day. Stashaway's app values are sometimes delayed by a day —\n"
        "# if the app shows 'as of 22 May' while today is 24 May, set this to\n"
        "# date(2026, 5, 22) to get an exact apples-to-apples reconcile.\n"
        "RECONCILE_AS_OF = END\n"
        "\n"
        "print(f'window: {START}..{END}  |  align_window={ALIGN_WINDOW}  |  target_ann_return={TARGET_ANN_RETURN:.1%}  |  reconcile_as_of={RECONCILE_AS_OF}')"
    ),
    md("## Parse + tag + fetch returns"),
    code(
        "parsed = parse_statement(STATEMENT_PATH, use_cache=False)\n"
        "portfolios = [\n"
        "    from_parsed(\n"
        "        p,\n"
        "        roles=ROLES[p.name],\n"
        "        metadata={'management_fee_annual': MGMT_FEES_ANNUAL.get(p.name, 0.0)},\n"
        "    )\n"
        "    for p in parsed if p.name in ROLES\n"
        "]\n"
        "holding = [p for p in portfolios if Role.HOLDING in p.roles]\n"
        "tickers = sorted({\n"
        "    h.metadata.ticker for p in holding for h in p.holdings\n"
        "    if not h.metadata.ticker.startswith('CASH_')\n"
        "})\n"
        "provider = YahooFinanceProvider()\n"
        "returns = provider.get_returns(tickers, START, END)\n"
        "print(f'Resolved {len(portfolios)} portfolios; fetched {returns.shape[1]} ticker series')"
    ),
    md("## Fetch USDSGD (daily series for return FX adjustment)"),
    code(
        "fx_bars = provider.get_bars(['USDSGD=X'], START, END)\n"
        "fx_series_usd_sgd = fx_bars.xs('USDSGD=X', level=0)['close']\n"
        "# Yahoo's USDSGD is labelled in UK time, so its 'closing' rate lands ~7h\n"
        "# before Stashaway's Singapore-EOD snapshot. For statement-date AUM we use\n"
        "# the rate parsed from the PDF (attached to portfolio.metadata via from_parsed).\n"
        "# Daily series is still used for compounding return adjustments — day-over-day\n"
        "# moves are roughly the same despite the timezone shift.\n"
        "stashaway_fx = portfolios[0].metadata.get('statement_fx_usd_sgd')\n"
        "yahoo_spot = float(fx_series_usd_sgd.iloc[-1])\n"
        "print('Statement-date FX (from PDF, used for AUM):  1 USD = {:.4f} SGD'.format(stashaway_fx))\n"
        "print('Yahoo USDSGD spot ({}, used for daily series): 1 USD = {:.4f} SGD'.format(\n"
        "    fx_series_usd_sgd.index.max().date(), yahoo_spot,\n"
        "))"
    ),
    md("## Validation — no portfolio silently dropped + window summary"),
    code(
        "from hailmary.allocation.diagnostic import _build_returns_panel, PortfolioDroppedError\n"
        "try:\n"
        "    _validation_panel = _build_returns_panel(\n"
        "        holding, returns=returns, fx_series_usd_sgd=fx_series_usd_sgd, strict=True,\n"
        "    )\n"
        "    print(f'All {len(holding)} HOLDING portfolios resolved cleanly '\n"
        "          f'({_validation_panel.shape[1]} series, {_validation_panel.shape[0]:,} dates).')\n"
        "except PortfolioDroppedError as exc:\n"
        "    print('FAIL — would drop portfolios:')\n"
        "    for name, reason in exc.dropped:\n"
        "        print(f'  {name}: {reason}')\n"
        "    raise\n"
        "\n"
        "first_dates = (\n"
        "    _validation_panel.apply(lambda c: c.dropna().index.min().date())\n"
        "    .sort_values(ascending=False)\n"
        ")\n"
        "common_start = first_dates.iloc[0]\n"
        "aligned_days = len(_validation_panel.loc[str(common_start):].dropna(how='any'))\n"
        "print()\n"
        "print(f'Common-history window: [{common_start}..{END}] ({aligned_days:,} aligned dates).')\n"
        "print('Per-portfolio first-data dates (latest first — these are what shrink the window):')\n"
        "for name, first_date in first_dates.head(8).items():\n"
        "    marker = '  ← constrains common_start' if first_date == common_start else ''\n"
        "    print(f'  {name:<22} {first_date}{marker}')\n"
        "if len(first_dates) > 8:\n"
        "    remaining = first_dates.iloc[8:]\n"
        "    print(f'  ...{len(remaining)} more portfolios start between '\n"
        "          f'{remaining.min()} and {remaining.max()}')\n"
        "print()\n"
        "print('Set ALIGN_WINDOW=True (default) → combined-book metrics use this aligned window.')\n"
        "print('Set ALIGN_WINDOW=False → full per-portfolio histories with dynamic-renorm weights.')"
    ),
    md("## Combined-book exposure"),
    code(
        "exposure = combined_exposure(portfolios)\n"
        "for dim, df in exposure.items():\n"
        "    print(f'\\n--- {dim.replace(\"_\", \" \").title()} ---')\n"
        "    display(df)\n"
        "combined_exposure_figure(exposure)"
    ),
    md("## Correlation matrix"),
    code(
        "corr = correlation_matrix(portfolios, returns=returns, fx_series_usd_sgd=fx_series_usd_sgd)\n"
        "display(corr.round(3))\n"
        "correlation_figure(corr)"
    ),
    md(f"## Redundancy (threshold default {0.85})"),
    code(
        "pairs = redundancy_pairs(corr, threshold=REDUNDANCY_THRESHOLD, portfolios=portfolios)\n"
        "if pairs:\n"
        "    pd.DataFrame(pairs, columns=['a', 'b', 'rho', 'candidate'])\n"
        "else:\n"
        "    print(f'No portfolio pairs above ρ = {REDUNDANCY_THRESHOLD}.')\n"
        "    print('If your customs are uncorrelated by design, this is expected — drop the threshold to 0.7 to surface near-redundancy.')"
    ),
    md("## Risk contribution"),
    code(
        "risk = risk_contribution(portfolios, returns=returns)\n"
        "print('--- By portfolio ---')\n"
        "display(risk['by_portfolio'].round(4))\n"
        "print('--- By holding (top 15) ---')\n"
        "display(risk['by_holding'].head(15).round(4))"
    ),
    md("## Benchmark comparison"),
    code(
        "bench = benchmark_comparison(portfolios, returns=returns, fx_series_usd_sgd=fx_series_usd_sgd)\n"
        "bench.round(3)"
    ),
    md("## Export HTML report (strict — fails if any portfolio would drop)"),
    code(
        "out = render_html_report(\n"
        "    portfolios,\n"
        "    REPORT_PATH,\n"
        "    returns=returns,\n"
        "    redundancy_threshold=REDUNDANCY_THRESHOLD,\n"
        "    fx_series_usd_sgd=fx_series_usd_sgd,\n"
        "    align_window=ALIGN_WINDOW,\n"
        "    target_ann_return=TARGET_ANN_RETURN,\n"
        "    reconciliation_as_of=RECONCILE_AS_OF,\n"
        "    title='Stashaway book — allocation diagnostic',\n"
        ")  # fx_rate_usd_sgd left default → picks up Stashaway PDF rate from portfolio.metadata\n"
        "print(f'Wrote {out.resolve()}')"
    ),
]


NB_SCENARIO = [
    md(
        "# 04 — Scenario compare (Phase 2)\n"
        "\n"
        "What-if rebalancing on top of the Phase 1 diagnostic. Express a proposed\n"
        "book via the edit helpers (`drop_portfolio`, `rebalance_into`, `merge_into`,\n"
        "`set_weights`) and diff it against the current book — every Phase 1 metric\n"
        "is recomputed on both sides and surfaced as deltas."
    ),
    md("## Setup — load real book + market data"),
    code(
        "from datetime import date\n"
        "from pathlib import Path\n"
        "\n"
        "import pandas as pd\n"
        "\n"
        "from hailmary.allocation.book_config import MGMT_FEES_ANNUAL, ROLES\n"
        "from hailmary.allocation.portfolios import Role, from_parsed\n"
        "from hailmary.allocation.statements import parse_statement\n"
        "from hailmary.allocation.returns import last_business_day_on_or_before\n"
        "from hailmary.allocation.scenarios import (\n"
        "    Scenario, drop_portfolio, merge_into, rebalance_into, scenario_compare, set_weights,\n"
        ")\n"
        "from hailmary.data.providers import YahooFinanceProvider\n"
        "\n"
        "STATEMENT_PATH = Path('../../data/statements/2026-04 StashAway Monthly Statement.pdf')\n"
        "START = date(2022, 1, 1)\n"
        "END = last_business_day_on_or_before(date.today())\n"
        "\n"
        "parsed = parse_statement(STATEMENT_PATH)\n"
        "portfolios = [\n"
        "    from_parsed(\n"
        "        p,\n"
        "        roles=ROLES[p.name],\n"
        "        metadata={'management_fee_annual': MGMT_FEES_ANNUAL.get(p.name, 0.0)},\n"
        "    )\n"
        "    for p in parsed if p.name in ROLES\n"
        "]\n"
        "holding = [p for p in portfolios if Role.HOLDING in p.roles]\n"
        "tickers = sorted({\n"
        "    h.metadata.ticker for p in holding for h in p.holdings\n"
        "    if not h.metadata.ticker.startswith('CASH_')\n"
        "})\n"
        "provider = YahooFinanceProvider()\n"
        "returns = provider.get_returns(tickers, START, END)\n"
        "fx_bars = provider.get_bars(['USDSGD=X'], START, END)\n"
        "fx_series_usd_sgd = fx_bars.xs('USDSGD=X', level=0)['close']\n"
        "print(f'{len(portfolios)} portfolios loaded, {len(holding)} HOLDING-tagged, '\n"
        "      f'window {START}..{END}')"
    ),
    code(
        "def headline(diff):\n"
        "    d = diff.deltas\n"
        "    print(f'Combined book changes ({diff.current.label}  ->  {diff.proposed.label}):')\n"
        "    print(f'  Sharpe Δ:    {d.book_sharpe_delta:+.3f}')\n"
        "    print(f'  Ann ret Δ:   {d.book_ann_return_delta:+.2%}')\n"
        "    print(f'  Vol Δ:       {d.book_ann_vol_delta:+.2%}')\n"
        "    print(f'  Max DD Δ:    {d.book_max_dd_delta:+.2%}')\n"
        "    print(f'  AUM Δ:       {d.book_aum_delta:+,.0f} SGD')\n"
        "    if d.redundancy_appeared:\n"
        "        print(f'  Redundancy appeared:    {d.redundancy_appeared}')\n"
        "    if d.redundancy_disappeared:\n"
        "        print(f'  Redundancy disappeared: {d.redundancy_disappeared}')\n"
        "\n"
        "cur = Scenario('current book', tuple(portfolios))\n"
        "kw = dict(returns=returns, fx_series_usd_sgd=fx_series_usd_sgd, align_window=True)"
    ),
    md(
        "## Scenario A — What if I dropped Crypto?\n"
        "\n"
        "Removes the Crypto sleeve entirely. The freed capital is just *gone* from the\n"
        "book (use `rebalance_into` if you want to redeploy it). Useful for seeing\n"
        "how much risk Crypto is contributing."
    ),
    code(
        "proposed = drop_portfolio(portfolios, 'Crypto')\n"
        "diff_a = scenario_compare(cur, Scenario('drop Crypto', proposed), **kw)\n"
        "headline(diff_a)"
    ),
    code(
        "print('Asset-class exposure shift:')\n"
        "diff_a.deltas.exposure_delta['asset_class'].head(10)"
    ),
    md(
        "## Scenario B — What if I moved Crypto into BlackRock?\n"
        "\n"
        "`rebalance_into` moves Crypto's whole `total_value` into BlackRock\n"
        "at BlackRock's current composition. Crypto disappears; BlackRock gets bigger.\n"
        "Both are USD sleeves — same-currency rotations only (cross-currency rotations\n"
        "raise `ScenarioEditError`; convert FX first if needed)."
    ),
    code(
        "proposed = rebalance_into(portfolios, 'Crypto', 'BlackRock')\n"
        "diff_b = scenario_compare(cur, Scenario('Crypto -> BlackRock', proposed), **kw)\n"
        "headline(diff_b)"
    ),
    md(
        "## Scenario C — What if I merged Energy + Utilities + HDY into one sleeve?\n"
        "\n"
        "Value-weighted union of the three customs into a single 'Custom Equity Sleeve'.\n"
        "Same total exposure — just consolidated administratively."
    ),
    code(
        "proposed = merge_into(\n"
        "    portfolios,\n"
        "    names=['Energy', 'Utilities', 'High Dividend Yield'],\n"
        "    into='Custom Equity Sleeve',\n"
        ")\n"
        "diff_c = scenario_compare(cur, Scenario('merge customs', proposed), **kw)\n"
        "headline(diff_c)"
    ),
    md(
        "## Scenario D — What if I shifted Crypto weights 50/50 BTC/ETH?\n"
        "\n"
        "`set_weights` replaces one sleeve's weights. Requires explicit weight for\n"
        "every existing holding (pass 0.0 to zero one out)."
    ),
    code(
        "crypto = next(p for p in portfolios if p.name == 'Crypto')\n"
        "print('Current Crypto holdings:')\n"
        "for h in crypto.holdings:\n"
        "    print(f'  {h.stashaway_id:<10} weight={h.weight:.3f}')"
    ),
    code(
        "# Build a 50/50 BTC/ETH proposal (zero out any other holdings)\n"
        "current_weights = {h.stashaway_id: h.weight for h in crypto.holdings}\n"
        "new_weights = {sid: 0.0 for sid in current_weights}\n"
        "if 'FBTC' in new_weights: new_weights['FBTC'] = 0.5\n"
        "if 'FETH' in new_weights: new_weights['FETH'] = 0.5\n"
        "# Sanity: weights sum to 1\n"
        "assert abs(sum(new_weights.values()) - 1.0) < 1e-9, new_weights\n"
        "\n"
        "proposed = set_weights(portfolios, 'Crypto', new_weights)\n"
        "diff_d = scenario_compare(cur, Scenario('Crypto 50/50 BTC/ETH', proposed), **kw)\n"
        "headline(diff_d)"
    ),
    md(
        "## Compare all four scenarios side-by-side"
    ),
    code(
        "import pandas as pd\n"
        "rows = []\n"
        "for label, diff in [\n"
        "    ('A: drop Crypto', diff_a),\n"
        "    ('B: Crypto -> SI', diff_b),\n"
        "    ('C: merge customs', diff_c),\n"
        "    ('D: 50/50 BTC/ETH', diff_d),\n"
        "]:\n"
        "    d = diff.deltas\n"
        "    rows.append({\n"
        "        'scenario': label,\n"
        "        'Sharpe Δ': d.book_sharpe_delta,\n"
        "        'AnnRet Δ': d.book_ann_return_delta,\n"
        "        'Vol Δ':    d.book_ann_vol_delta,\n"
        "        'MaxDD Δ':  d.book_max_dd_delta,\n"
        "        'AUM Δ':    d.book_aum_delta,\n"
        "        'red. appeared':    len(d.redundancy_appeared),\n"
        "        'red. disappeared': len(d.redundancy_disappeared),\n"
        "    })\n"
        "summary = pd.DataFrame(rows).set_index('scenario')\n"
        "summary.style.format({\n"
        "    'Sharpe Δ': '{:+.3f}', 'AnnRet Δ': '{:+.2%}',\n"
        "    'Vol Δ': '{:+.2%}', 'MaxDD Δ': '{:+.2%}', 'AUM Δ': '{:+,.0f}',\n"
        "}, na_rep='-')"
    ),
    md(
        "## Want HTML?\n"
        "\n"
        "Chunk 3 (`render_scenario_report`) is still TODO. Once built, every `diff`\n"
        "above can be exported to a self-contained side-by-side HTML with the same\n"
        "styling as `reports/allocation_diagnostic.html`."
    ),
]


NB_ETF_EXPLORER = [
    md(
        "# ETF Explorer (v1)\n"
        "\n"
        "Discovery view over Stashaway's full ETF Explorer offering (~98 ETFs).\n"
        "Computes multi-window metrics (1Y/3Y/5Y) + correlation with your combined book.\n"
        "Renders to `reports/etf_explorer.html`."
    ),
    code(
        "from datetime import date\n"
        "from pathlib import Path\n"
        "\n"
        "from hailmary.allocation.book_config import MGMT_FEES_ANNUAL, ROLES\n"
        "from hailmary.allocation.etf_explorer import build_etf_explorer, render_etf_explorer_report\n"
        "from hailmary.allocation.portfolios import from_parsed\n"
        "from hailmary.allocation.statements import parse_statement\n"
        "from hailmary.allocation.returns import last_business_day_on_or_before\n"
        "from hailmary.data.providers import YahooFinanceProvider\n"
        "\n"
        "ETF_XLSX = Path('../../data/stashaway_etf_universe.xlsx')\n"
        "STATEMENT_PATH = Path('../../data/statements/2026-04 StashAway Monthly Statement.pdf')\n"
        "REPORT_PATH = Path('../../reports/etf_explorer.html')\n"
        "START = date(2020, 1, 1)\n"
        "END = last_business_day_on_or_before(date.today())\n"
        "TARGET_ANN_RETURN = 0.05\n"
        "print(f'window: {START}..{END}')"
    ),
    md("## Load user's book (for correlation reference)"),
    code(
        "parsed = parse_statement(STATEMENT_PATH)\n"
        "portfolios = [\n"
        "    from_parsed(\n"
        "        p,\n"
        "        roles=ROLES[p.name],\n"
        "        metadata={'management_fee_annual': MGMT_FEES_ANNUAL.get(p.name, 0.0)},\n"
        "    )\n"
        "    for p in parsed if p.name in ROLES\n"
        "]\n"
        "provider = YahooFinanceProvider()\n"
        "fx_bars = provider.get_bars(['USDSGD=X'], START, END)\n"
        "fx_series_usd_sgd = fx_bars.xs('USDSGD=X', level=0)['close']\n"
        "print(f'{len(portfolios)} portfolios loaded')"
    ),
    md("## Build the explorer DataFrame"),
    code(
        "explorer_df = build_etf_explorer(\n"
        "    ETF_XLSX,\n"
        "    portfolios=portfolios,\n"
        "    price_source=provider,\n"
        "    fx_series_usd_sgd=fx_series_usd_sgd,\n"
        "    start=START,\n"
        "    end=END,\n"
        ")\n"
        "print(f'{len(explorer_df)} ETFs, {explorer_df[\"has_data\"].sum()} with Yahoo data')\n"
        "explorer_df.head(20)"
    ),
    md("## Render HTML report"),
    code(
        "out = render_etf_explorer_report(\n"
        "    explorer_df,\n"
        "    REPORT_PATH,\n"
        "    target_ann_return=TARGET_ANN_RETURN,\n"
        ")\n"
        "print(f'Wrote {out.resolve()}')"
    ),
]


def main() -> None:
    write_notebook(Path("notebooks/allocation/01_validate_holdings.ipynb"), NB01)
    write_notebook(Path("notebooks/allocation/02_validate_returns.ipynb"), NB02)
    write_notebook(Path("notebooks/allocation/03_allocation_diagnostic.ipynb"), NB03)
    write_notebook(Path("notebooks/allocation/04_scenario_compare.ipynb"), NB_SCENARIO)
    write_notebook(Path("notebooks/allocation/etf_explorer.ipynb"), NB_ETF_EXPLORER)


if __name__ == "__main__":
    main()
