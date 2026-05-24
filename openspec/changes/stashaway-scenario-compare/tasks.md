# 1. Scenario data model (`scenarios.py`)

- [ ] 1.1 Define `Scenario` frozen dataclass: `label: str`, `portfolios: tuple[Portfolio, ...]`, `note: str | None = None`
- [ ] 1.2 Define `ScenarioDeltas` dataclass: per-portfolio Sharpe/vol/max-DD deltas, book-level Sharpe/vol/max-DD/AUM shift, exposure-bucket shift, redundancy-pair appearances/disappearances
- [ ] 1.3 Define `ScenarioDiff` dataclass: `current`, `proposed`, and paired tuples of every Phase 1 diagnostic output + `deltas: ScenarioDeltas`
- [ ] 1.4 Unit tests: `Scenario` immutability; `ScenarioDeltas` field types; `ScenarioDiff` construction with synthetic data

# 2. Edit helpers (`scenarios.py`)

- [ ] 2.1 `drop_portfolio(portfolios, name) -> tuple[Portfolio, ...]` — remove a whole portfolio; raise if not found
- [ ] 2.2 `set_weights(portfolios, portfolio_name, weights: dict[str, float]) -> tuple[Portfolio, ...]` — replace one portfolio's weights; validate sum-to-1 tolerance; raise on unknown ticker
- [ ] 2.3 `rebalance_into(portfolios, from_name, to_name) -> tuple[Portfolio, ...]` — move all `total_value` from `from_name` into `to_name`'s `total_value`; drop `from_name`; preserve `to_name`'s weights
- [ ] 2.4 `merge_into(portfolios, names: Sequence[str], into: str, *, weights: str = "value-weighted") -> tuple[Portfolio, ...]` — combine N portfolios into one (default = value-weighted blend of their holdings)
- [ ] 2.5 All helpers are non-mutating: input `portfolios` is unchanged after call; verified by test
- [ ] 2.6 Unit tests for each helper: happy path + at least one error case (unknown name, bad weights, etc.)

# 3. `scenario_compare` engine

- [ ] 3.1 Implement `scenario_compare(current, proposed, *, start=None, end=None, price_source=None, returns=None, redundancy_threshold=0.85, risk_free_rate=0.0) -> ScenarioDiff`
- [ ] 3.2 Run all six Phase 1 diagnostics on both books with identical kwargs (passes `start`/`end` through)
- [ ] 3.3 Compute `ScenarioDeltas`:
  - `book_performance`: Sharpe / ann_return / ann_vol / max_dd / aum deltas
  - `benchmarks`: per-portfolio Sharpe / vol / max-DD delta (rows present in both)
  - `exposure`: per-bucket weight delta for each of asset_class / region / sector
  - `redundancy`: pairs appearing only in proposed (`appeared`), only in current (`disappeared`), in both (`persisted`)
- [ ] 3.4 Sanity test: `scenario_compare(s, s)` returns zero deltas across the board
- [ ] 3.5 Unit test: drop one portfolio, assert AUM and exposure deltas reflect that portfolio's contribution

# 4. HTML report (`scenarios.py`)

- [ ] 4.1 Build `src/hailmary/allocation/_scenario_report_template.html.j2`:
  - top: "What changed" delta strip (book metrics + redundancy appearances/disappearances)
  - per section: two-column grid (current | proposed) reusing Phase 1's `_*_to_html` helpers
  - inherits Phase 1's caveat block + adds a "scenario projection" caveat
- [ ] 4.2 Implement `render_scenario_report(diff, output_path, *, title="Scenario Comparison") -> Path`
- [ ] 4.3 Embed equity curves for both scenarios side-by-side
- [ ] 4.4 Unit test: render against a small synthetic before/after; assert file exists, non-trivial, includes both scenario labels

# 5. Notebook

- [ ] 5.1 `notebooks/allocation/04_scenario_compare.ipynb`:
  - Load real 2026-04 book via the existing `load_book` path used in notebook 03
  - Section 1: `drop_portfolio` example (e.g. "what if Crypto weren't there?")
  - Section 2: `rebalance_into` example (e.g. "what if Crypto's capital moved to Singapore Investing?")
  - Section 3: `set_weights` example (re-weighting inside one portfolio)
  - Each section: render and display the `ScenarioDiff` deltas table inline; export an HTML per scenario to `reports/`
- [ ] 5.2 Notebook must execute clean end-to-end via `jupyter nbconvert --execute`

# 6. Public API + docs

- [ ] 6.1 Export `Scenario`, `ScenarioDiff`, `ScenarioDeltas`, `scenario_compare`, `render_scenario_report`, and the four edit helpers from `hailmary.allocation.__init__`
- [ ] 6.2 Update `CLAUDE.md` allocation paragraph to mention Phase 2 scenario comparison
- [ ] 6.3 Add a one-liner pointer from `stashaway-allocation-diagnostic`'s memory notes (or replace them with a "Phase 2 superseded by ..." note once archived)

# 7. Acceptance

- [ ] 7.1 `scenario_compare(current, current)` produces all-zero deltas (sanity)
- [ ] 7.2 `drop_portfolio(current, "Crypto")` produces a `ScenarioDiff` where book vol decreases and Crypto vanishes from every Phase 1 output
- [ ] 7.3 The HTML report opens in a browser, shows current vs proposed side-by-side, and surfaces the delta strip at the top
- [ ] 7.4 All unit tests pass; ruff clean; mypy strict not required (consistent with Phase 1)
- [ ] 7.5 Notebook 04 executes clean and writes at least one report to `reports/`
