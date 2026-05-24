## ADDED Requirements

### Requirement: Scenario data type wraps a portfolio book

The system SHALL provide a `Scenario` data type that wraps an immutable collection of `Portfolio` objects with a human-readable label and an optional note. The wrapped collection SHALL be a `tuple[Portfolio, ...]` so accidental mutation requires explicit reconstruction. `Scenario` instances SHALL be hashable and support equality based on field values.

#### Scenario: Construct from current book
- **WHEN** a `Scenario` is constructed from a parsed-statement portfolio list
- **THEN** the wrapped `portfolios` field is a `tuple`, equal to the input portfolios in order, and the input list is unchanged

#### Scenario: Mutation requires reconstruction
- **WHEN** a caller attempts to assign to `Scenario.portfolios` or `Scenario.label`
- **THEN** the assignment raises (frozen dataclass), forcing the caller to construct a new `Scenario`

### Requirement: Non-mutating edit helpers for the four common scenario edits

The system SHALL provide four edit helpers — `drop_portfolio`, `set_weights`, `rebalance_into`, `merge_into` — each of which SHALL accept a `Sequence[Portfolio]` and return a new `tuple[Portfolio, ...]` without modifying the input. Helpers SHALL raise typed errors on invalid input rather than silently producing nonsense outputs.

#### Scenario: drop_portfolio removes a named portfolio
- **WHEN** `drop_portfolio(book, name)` is called with `name` matching exactly one portfolio
- **THEN** the returned book contains all other portfolios in original order, the named portfolio is absent, and the input `book` list is unchanged

#### Scenario: drop_portfolio raises on unknown name
- **WHEN** `drop_portfolio(book, name)` is called with a `name` not present in `book`
- **THEN** a `KeyError` (or equivalent typed error) is raised identifying the missing name and listing the available names

#### Scenario: set_weights validates the new weight vector
- **WHEN** `set_weights(book, portfolio_name, new_weights)` is called with a dict whose values do not sum to 1.0 within `1e-4`
- **THEN** a validation error is raised identifying the portfolio and the actual sum, and the input book is unchanged

#### Scenario: rebalance_into moves all capital between two portfolios
- **WHEN** `rebalance_into(book, "Crypto", "Singapore Investing")` is called
- **THEN** the returned book contains no "Crypto" portfolio; "Singapore Investing"'s `total_value` equals its original `total_value` plus Crypto's original `total_value`; its weights are unchanged; the input book is unchanged

#### Scenario: merge_into combines several portfolios value-weighted
- **WHEN** `merge_into(book, ["Energy", "Utilities", "High Dividend Yield"], into="Custom Equity Sleeve")` is called
- **THEN** the returned book contains a new portfolio whose holdings are the value-weighted union of the inputs' holdings and whose `total_value` is the sum of their `total_value`s; the three source portfolios are absent

### Requirement: scenario_compare runs the full Phase 1 diagnostic on both books

The system SHALL provide a `scenario_compare(current: Scenario, proposed: Scenario, **diagnostic_kwargs) -> ScenarioDiff` function that runs every Phase 1 diagnostic (`combined_exposure`, `correlation_matrix`, `redundancy_pairs`, `risk_contribution`, `benchmark_comparison`, `book_performance`) on both `current.portfolios` and `proposed.portfolios` with identical kwargs and returns paired outputs plus a computed `ScenarioDeltas` summary.

The `ScenarioDiff` SHALL contain, in addition to paired raw outputs, the following computed deltas:
- Book-level: Sharpe Δ, annualised return Δ, annualised vol Δ, max drawdown Δ, AUM Δ
- Per-portfolio (where the portfolio name is present in both books): Sharpe Δ, vol Δ, max DD Δ
- Per exposure bucket (asset class, region, sector): weight Δ
- Redundancy: pairs `appeared` (only in proposed), `disappeared` (only in current), `persisted` (in both)

#### Scenario: Comparing a book against itself produces zero deltas
- **WHEN** `scenario_compare(s, s)` is called for any non-empty `Scenario` `s`
- **THEN** every numeric delta in the returned `ScenarioDiff` is zero (within `1e-9` tolerance), and `redundancy.appeared` and `redundancy.disappeared` are both empty

#### Scenario: Dropping a portfolio reflects in book performance and exposure deltas
- **WHEN** `proposed` is `current` with one HOLDING-tagged portfolio dropped via `drop_portfolio`
- **THEN** `ScenarioDeltas.aum_delta` equals the negative of the dropped portfolio's `total_value`; that portfolio's name does not appear in any of `proposed`'s diagnostic outputs

#### Scenario: Both books share the diagnostic kwargs
- **WHEN** `scenario_compare(current, proposed, start="2023-01-01", end="2025-12-31")` is called
- **THEN** every diagnostic is computed over the same window on both books; per-portfolio delta rows are only present for portfolios whose return history overlaps the window in both books

### Requirement: Side-by-side HTML scenario report

The system SHALL provide a `render_scenario_report(diff: ScenarioDiff, output_path, *, title="Scenario Comparison") -> Path` function that writes a self-contained HTML file presenting the current vs proposed diagnostic outputs side-by-side, with a "what changed" delta strip at the top of the report.

The report SHALL include:
- Title, generation timestamp, and both `Scenario.label`s clearly shown
- A delta strip summarising book-level changes (Sharpe Δ, vol Δ, max-DD Δ, AUM Δ) and counts of redundancy pairs appeared/disappeared
- Per diagnostic section (combined exposure, correlation matrix, redundancy pairs, risk contribution, benchmark comparison, book performance summary, equity curve): a two-column layout showing current on the left and proposed on the right
- The Phase 1 caveat block, plus a second caveat clarifying that proposed-book numbers are projections under the same current-snapshot reconstruction assumption

#### Scenario: Successful render to disk
- **WHEN** `render_scenario_report` is called with a `ScenarioDiff` containing non-empty diagnostic outputs and a writeable `output_path`
- **THEN** a self-contained `.html` file is written at the target path, openable in any modern browser without external assets, containing both scenario labels and the delta strip
