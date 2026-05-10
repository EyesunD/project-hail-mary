## ADDED Requirements

### Requirement: Combined-book exposure across sector, region, and asset class

The system SHALL compute the dollar-weighted exposure of the combined book (all portfolios tagged `holding`) across sector, region, and asset class, using the metadata provided by the asset universe map. Output SHALL be returned as a structured DataFrame and as a `plotly` figure rendered with the project's house theme.

#### Scenario: Single-portfolio book
- **WHEN** the combined book contains a single `holding` portfolio
- **THEN** the exposure breakdown matches that portfolio's holdings exactly, weighted by holding value

#### Scenario: Multi-portfolio book with overlapping holdings
- **WHEN** the combined book contains multiple `holding` portfolios with overlapping tickers
- **THEN** each ticker appears once in the per-asset view with summed dollar value, and the sector / region / asset-class aggregations sum across portfolios without double-counting

### Requirement: Pairwise correlation matrix across all portfolios

The system SHALL compute the pairwise correlation matrix of portfolio return series across all portfolios in the input collection (regardless of role) and return it as a square `pandas.DataFrame` indexed by portfolio name.

#### Scenario: Computed over the full common window
- **WHEN** the correlation matrix is requested without a date-range argument
- **THEN** it is computed over the full common date window of the input portfolios' return series

#### Scenario: Computed over a restricted window
- **WHEN** the correlation matrix is requested with a `(start, end)` argument
- **THEN** it is computed over the intersection of the requested window and the common available history

### Requirement: Redundancy flagging by correlation threshold

The system SHALL flag pairs of portfolios whose pairwise correlation exceeds a configurable threshold (default 0.85). Output SHALL identify the pair and the correlation value. Portfolios tagged `protected` SHALL be excluded from being proposed as the change-candidate side of a pair.

#### Scenario: High-correlation pair surfaced
- **WHEN** two non-protected portfolios A and B have correlation > 0.85 over the analysis window
- **THEN** the pair (A, B, ρ) appears in the redundancy output

#### Scenario: Protected portfolio in a high-correlation pair
- **WHEN** a `protected` portfolio P is highly correlated with a `custom` portfolio C
- **THEN** the pair appears in the output, but only C is marked as a change candidate; P is never marked as the side to consolidate

### Requirement: Risk contribution analysis per holding and per portfolio

The system SHALL compute, for the combined book, the risk contribution of each holding and each portfolio to total book volatility, using the covariance matrix produced by `analytics/risk.py`.

#### Scenario: Risk contributions sum to total
- **WHEN** holding-level risk contributions are computed for the combined book
- **THEN** the sum of contributions equals the total book volatility (within float tolerance)

### Requirement: Performance comparison versus tagged benchmarks

The system SHALL compute Sharpe ratio, maximum drawdown, and annualised volatility for each `custom` portfolio and for each `managed_benchmark` portfolio, and present a comparison table including the delta versus each benchmark over a common analysis window.

#### Scenario: Custom vs single benchmark
- **WHEN** there is at least one `custom` portfolio and at least one `managed_benchmark` portfolio with overlapping return history
- **THEN** the comparison output contains, for each `custom` portfolio, its Sharpe / max-DD / annualised-vol values and the deltas vs. each `managed_benchmark`

#### Scenario: No managed benchmarks present
- **WHEN** no portfolios in the input are tagged `managed_benchmark`
- **THEN** the comparison output is empty for benchmark deltas, the diagnostic emits a warning, and other diagnostic outputs are unaffected

### Requirement: Diagnostic outputs are HTML-renderable

The system SHALL provide a function that takes a collection of `Portfolio` objects and produces a self-contained HTML report containing all diagnostic outputs (combined exposure, correlation matrix, redundancy flags, risk contribution, benchmark comparison) with charts rendered using the project's house theme.

#### Scenario: Successful HTML export
- **WHEN** the report function is called with a non-empty portfolio collection and a target output path
- **THEN** a self-contained `.html` file is written at the target path containing all diagnostic sections, openable in any modern browser without external assets
