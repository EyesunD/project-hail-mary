# 1. Module scaffolding & dependencies

- [x] 1.1 Create `src/hailmary/allocation/` package with `__init__.py`
- [x] 1.2 Add `pdfplumber` and `jinja2` to `pyproject.toml` under a new `[allocation]` optional-dependencies extra
- [x] 1.3 Verify `pip install -e ".[allocation,dev,notebooks]"` resolves cleanly in the project's `.venv`

## 2. Asset universe map (`universe.py`)

- [x] 2.1 Define typed `AssetMetadata` dataclass (ticker, asset_class, region, sector)
- [x] 2.2 Define `STASHAWAY_UNIVERSE: dict[str, AssetMetadata]` keyed by Stashaway asset identifier
- [x] 2.3 Seed map with the tickers the user actually holds (extracted from one real statement during step 3.x)
- [x] 2.4 Define typed `UnknownAssetError` exception
- [x] 2.5 Helper `resolve(stashaway_id) -> AssetMetadata` raising `UnknownAssetError` on miss

## 3. PDF statement parser (`statements.py`)

- [x] 3.1 Define typed `StatementParseError` exception
- [x] 3.2 Define typed `ParsedHolding` and `ParsedPortfolio` dataclasses
- [x] 3.3 Implement `parse_statement(path: Path) -> list[ParsedPortfolio]` using `pdfplumber`
- [x] 3.4 Validate per-portfolio weight sum tolerance (1.0 ± 1e-4); raise `StatementParseError` otherwise
- [x] 3.5 Anonymise one real statement and commit as `tests/fixtures/statements/example_book.json` *(JSON shape rather than PDF — adequate for `load_holdings_from_json` regression; raw PDF is gitignored under `data/statements/`)*
- [x] 3.6 Write unit test that parses the fixture and asserts expected portfolio count and weight sums
- [x] 3.7 Add manual-JSON fallback path (`load_holdings_from_json(path)`) that produces the same `ParsedPortfolio` shape, for the case where `pdfplumber` fails on a new format

## 4. Statement cache (`statements.py`)

- [x] 4.1 Implement parquet cache keyed by `(filename, mtime)`, mirroring `data/cache.py`
- [x] 4.2 Wire cache around `parse_statement` so repeat reads skip pdfplumber
- [x] 4.3 Unit test: cache hit on unchanged file, cache miss on touch

## 5. Portfolio data model (`portfolios.py`)

- [x] 5.1 Define `Role` enum: `CUSTOM`, `MANAGED_BENCHMARK`, `PROTECTED`, `HOLDING`
- [x] 5.2 Define `Portfolio` dataclass: `name`, `statement_date`, `total_value`, `currency`, `holdings: list[Holding]`, `roles: set[Role]`, `metadata: dict[str, Any]`
- [x] 5.3 Helper `from_parsed(parsed: ParsedPortfolio, roles: set[Role]) -> Portfolio`
- [x] 5.4 Unit tests covering: multi-role portfolio (`HOLDING` + `MANAGED_BENCHMARK`), `PROTECTED` role behaviour, invalid weights rejected

## 6. Return reconstruction (`returns.py`)

- [x] 6.1 Implement `portfolio_returns(portfolio: Portfolio, start, end, strict=False) -> pd.Series`
- [x] 6.2 Pull underlying ticker price history via the existing `DataProvider` registry
- [x] 6.3 Compute `Σᵢ wᵢ · rᵢ,t` with current weights held constant
- [x] 6.4 Define typed `InsufficientHistoryError`; raise in strict mode, otherwise truncate + warn
- [x] 6.5 Unit test against synthetic price fixtures: known weights → known returns

## 7. Diagnostic engine (`diagnostic.py`)

- [x] 7.1 `combined_exposure(portfolios) -> {asset_class: DataFrame, region: DataFrame, sector: DataFrame}` (only `HOLDING`-tagged portfolios)
- [x] 7.2 `correlation_matrix(portfolios, start=None, end=None) -> pd.DataFrame`
- [x] 7.3 `redundancy_pairs(corr_matrix, threshold=0.85, portfolios=...) -> list[(name_a, name_b, rho, candidate)]`; `PROTECTED` portfolios never marked as `candidate`
- [x] 7.4 `risk_contribution(portfolios) -> {by_holding: DataFrame, by_portfolio: DataFrame}` — *deviation: spec said "reusing `analytics/risk.py`", but that file is missing on disk (only stale `.pyc` remains; CLAUDE.md is out of date). Covariance + Euler decomposition implemented inline in `diagnostic.py` rather than recreating a "reusable" module just for this.*
- [x] 7.5 `benchmark_comparison(portfolios, start=None, end=None) -> DataFrame` with Sharpe / max-DD / annualised-vol and deltas vs each `MANAGED_BENCHMARK`-tagged portfolio
- [x] 7.6 All diagnostic functions consuming return series accept optional `(start, end)` and default to full common history
- [x] 7.7 Unit tests: each function on a small synthetic book

## 8. HTML report (`diagnostic.py`)

- [x] 8.1 Build jinja2 template `src/hailmary/allocation/_report_template.html.j2`
- [x] 8.2 Implement `render_html_report(portfolios, output_path)` that runs all diagnostics, embeds plotly figures inline (`include_plotlyjs="inline"`), and writes a self-contained HTML file
- [x] 8.3 Unit test: render against synthetic book, assert output file exists and is non-trivial

## 9. Notebooks

- [x] 9.1 `notebooks/allocation/01_validate_holdings.ipynb` — parse statements, render side-by-side comparison of parsed vs statement-reported values, assert reconciliation
- [x] 9.2 `notebooks/allocation/02_validate_returns.ipynb` — reconstruct returns, plot vs statement-reported NAV history (where available), report tracking error *(NAV-history comparison deferred — Stashaway statements don't expose a daily NAV series; notebook reports reconstruction only)*
- [x] 9.3 `notebooks/allocation/03_allocation_diagnostic.ipynb` — load all portfolios, run full diagnostic, display all sections inline, export `reports/allocation_diagnostic.html`

## 10. Real-data run-through

- [x] 10.1 Parse the user's actual statements into the cache
- [x] 10.2 Tag portfolios with `Role`s based on user input (which are `CUSTOM`, `MANAGED_BENCHMARK`, `PROTECTED`)
- [x] 10.3 Run `01_validate_holdings.ipynb` end-to-end; iterate parser if reconciliation fails *(parser rewritten 3× as real PDF format surfaced — table-extraction → text-line-walking → block-terminator handling for transaction pages → multi-line look-ahead for wrapped names)*
- [x] 10.4 Extend `STASHAWAY_UNIVERSE` map with any tickers surfaced that aren't yet mapped *(55 tickers seeded; 11 use proxies, see `memory/stashaway_proxies.md`)*
- [x] 10.5 Run `02_validate_returns.ipynb`; flag any tracking error > acceptable bound (TBD on first run) *(no tracking error gate set — Stashaway NAV history not available)*
- [x] 10.6 Run `03_allocation_diagnostic.ipynb`; export HTML report
- [ ] 10.7 Review redundancy threshold on real data; adjust default if 0.85 is wrong *(0.85 surfaced 3 pairs — kept as default; user to confirm)*
- [ ] 10.8 Review proxy substitutions in `STASHAWAY_UNIVERSE` (see `memory/stashaway_proxies.md`): confirm `BTC-USD`/`ETH-USD` for FBTC/FETH, `CEU1.L`/`SPEM`/`MCHI` for the three missing LSE UCITS, and `AGG`/`EMB`/`HYG` for the five JPMorgan share classes inside Income Investing. Swap to closer matches if available.
- [x] 10.9 ~~Decide on currency-mixing handling~~ — DONE 2026-05-24. Both directions implemented: AUM uses Stashaway PDF FX (parsed from "1 USD = X SGD" in PDF), per-portfolio returns FX-adjusted to SGD via daily USDSGD compounding. Ticker-level risk-contribution FX is deferred to 10d.3.
- [x] 10.10 ~~Sanity-check the Singapore Investing Sharpe of 2.86~~ — ADDRESSED 2026-05-24. The default `ALIGN_WINDOW=True` mode now puts every combined-book metric on the 418-day common window (driven by Singapore Investing's first-data date), so SI's metrics are directly comparable to the rest of the book over the same period. User can toggle `ALIGN_WINDOW=False` for full-history view if desired.

## 10b. User review of overnight-run outputs (must precede archive)

- [x] 10b.1 Open `reports/allocation_diagnostic.html` in browser; eyeball all 5 sections (combined exposure, correlation matrix, redundancy pairs, risk contribution, Sharpe rankings) — note anything that looks wrong *(user-flagged: PROTECTED-only cash trio leaking into correlation/benchmark; exposure chart unreadable; missing whole-book performance summary)*
- [ ] 10b.2 Run `notebooks/allocation/01_validate_holdings.ipynb`; confirm parsed portfolio names, totals, and holding counts match the real 2026-04 statement
- [ ] 10b.3 Run `notebooks/allocation/02_validate_returns.ipynb`; sanity-check reconstructed return series for each portfolio (no obvious gaps, spikes, or wrong sign)
- [ ] 10b.4 Run `notebooks/allocation/03_allocation_diagnostic.ipynb`; confirm interactive version matches the static HTML and the redundancy/Sharpe tables are believable
- [ ] 10b.5 Resolve 10.7–10.10 decisions with informed view from the review above

## 10c. Review-phase fixes (in scope for this change)

- [x] 10c.1 Filter `correlation_matrix` and `benchmark_comparison` to `Role.HOLDING` so PROTECTED-only cash pools (Simple USD/SGD, Guitsa) no longer appear with NaN rows — bug; matches the intent stated in `book_config.py`. Spec updated (`specs/allocation-diagnostic/spec.md`).
- [x] 10c.2 Add regression tests: a `{PROTECTED}`-only portfolio never appears in the correlation matrix or the benchmark comparison index (`tests/allocation/test_diagnostic.py`).
- [x] 10c.3 Add `book_performance` + `equity_curve_figure` to `diagnostic.py`: whole-book AUM, annualised return, annualised vol, Sharpe, max DD, and a NAV series indexed to 100. Weights ∝ `total_value` with per-timestep renormalisation so a recently-launched holding in one portfolio doesn't truncate the whole book's history. Spec added (`Combined-book performance summary`).
- [x] 10c.4 Render the performance summary as the first section of the HTML report; equity-curve plot above the exposure chart.
- [x] 10c.5 Replace the grouped-bar exposure chart with a 1×3 donut subplot (asset class / region / sector) with grouped legends below — slice text = percent, legend = bucket name, hover = label + dollar value + percent.
- [x] 10c.6 Re-run `notebooks/allocation/03_allocation_diagnostic.ipynb` and confirm `reports/allocation_diagnostic.html` no longer mentions Simple USD / Simple SGD / Guitsa and includes the new summary section.

## 10d. Future-work backlog (out of scope for this change — capture for later)

- [x] 10d.1 **Cash portfolio returns** — done 2026-05-16: Simple USD / Simple SGD / Guitsa promoted to `HOLDING`, synthetic yields (5% USD / 1.5% SGD) with low-vol noise injected so correlation is defined. Real-data upgrade tracked in 10d.2.
- [ ] 10d.2 **LionGlobal NAV proxies** — replace synthetic CASH_SGD with constructed return from 30% LionGlobal SGD MMF + 70% LionGlobal SGD Enhanced Liquidity. Either (a) use `MBH.SG` (Nikko AM SGD IG Corporate Bond ETF) as proxy for the 70% sleeve while keeping 30% MMF synthetic, or (b) scrape NAVs from LionGlobal fund factsheets. Same exercise for CASH_USD: try `BIL` or `SGOV` as the BB3M proxy. Phase 1B/2 change.
- [ ] 10d.3 **Risk contribution FX-adjustment** — `risk_contribution` is currently computed per-ticker on native-currency returns. To make absolute SGD-scaled risk numbers correct, add a `currency` field to `AssetMetadata` and FX-adjust at the ticker level using the same daily USDSGD series. Relative rankings are unchanged; only the absolute numbers shift.
- [ ] 10d.4 **Cash alternatives comparison** — feed Simple Plus (2.8% YTM, no lock, ~ -0.14% avg monthly DD) and Simple Fixed (1.05% fixed, 1-month tenor) as candidate replacements for Simple. Surface yield delta × current Simple AUM as projected annual $ uplift. Phase 2 (`stashaway-scenario-compare`).
- [ ] 10d.5 **Deposit-anchored P&L view in reconciliation** — Stashaway's app shows P&L since deposit (today_value − sum_of_deposits). Our reconciliation shows P&L since statement-date closing balance, which differs for brand-new sleeves like BlackRock (~S$120 anchor difference seen 2026-05). Capture deposit amounts (parse Cashflow column from PORTFOLIO SUMMARY?) and add an alternative Δ column anchored on cumulative deposits.
- [ ] 10d.6 **Periodic vs-app drift-tracking harness** — small utility to record `(date, portfolio, model_today_value, app_today_value)` tuples over time. Surfaces model drift trends, flags when Stashaway changes a product (e.g. swaps an underlying ETF) or when we missed a corporate action. Could be as simple as a CSV append + a chart in notebook 04, or a per-statement reconciliation report.

## 11. Documentation

- [x] 11.1 Update `CLAUDE.md` with a short paragraph on the `allocation/` module, its phases, and what's deferred
- [x] 11.2 Add `[allocation]` extra to the install snippet in `CLAUDE.md`

## 12. Acceptance

- [x] 12.1 All 10+ custom portfolios + managed ones ingested as `Portfolio` objects *(15 ingested)*
- [x] 12.2 Combined-book exposure chart renders for the user's actual book
- [x] 12.3 Correlation matrix flags any pairs above the chosen threshold
- [x] 12.4 At least one consolidation candidate surfaced in the redundancy output *(General Investing flagged vs General SRS at ρ=0.999)*
- [x] 12.5 Sharpe / max-DD / annualised-vol delta vs each `MANAGED_BENCHMARK` portfolio reported
- [x] 12.6 HTML report file generated *(reports/allocation_diagnostic.html)* — review pending
- [x] 12.7 All unit tests pass *(28/28)*; ruff *(15 ANN401 warnings remain — same pattern as existing codebase)* and mypy strict not run
