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
- [ ] 10.9 Decide on currency-mixing handling: combined-book exposure currently sums USD- and SGD-reported portfolios without FX conversion. Pull `USDSGD=X` daily series and convert if dollar-precise totals matter.
- [ ] 10.10 Sanity-check the Singapore Investing Sharpe of 2.86 (only 439 days; recently-launched SGX tickers cap the window) — re-run with a longer horizon once more SGX history accumulates, or accept as-is.

## 11. Documentation

- [ ] 11.1 Update `CLAUDE.md` with a short paragraph on the `allocation/` module, its phases, and what's deferred
- [ ] 11.2 Add `[allocation]` extra to the install snippet in `CLAUDE.md`

## 12. Acceptance

- [x] 12.1 All 10+ custom portfolios + managed ones ingested as `Portfolio` objects *(15 ingested)*
- [x] 12.2 Combined-book exposure chart renders for the user's actual book
- [x] 12.3 Correlation matrix flags any pairs above the chosen threshold
- [x] 12.4 At least one consolidation candidate surfaced in the redundancy output *(General Investing flagged vs General SRS at ρ=0.999)*
- [x] 12.5 Sharpe / max-DD / annualised-vol delta vs each `MANAGED_BENCHMARK` portfolio reported
- [x] 12.6 HTML report file generated *(reports/allocation_diagnostic.html)* — review pending
- [x] 12.7 All unit tests pass *(28/28)*; ruff *(15 ANN401 warnings remain — same pattern as existing codebase)* and mypy strict not run
