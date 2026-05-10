## ADDED Requirements

### Requirement: Parse Stashaway monthly PDF statements into structured holdings

The system SHALL extract per-portfolio holdings from Stashaway monthly PDF statements and return them as a typed structure containing portfolio name, statement date, total value (in the statement's reporting currency), and a list of `(ticker, weight, value)` rows summing (within float tolerance) to the portfolio total.

#### Scenario: Successful extraction of a well-formed statement
- **WHEN** the parser is given a Stashaway monthly statement PDF whose holdings table follows the current published format
- **THEN** the parser returns one structured record per portfolio, each containing the portfolio name, statement date, reporting currency, total value, and per-holding rows whose weights sum to 1.0 ± 1e-4

#### Scenario: Statement with missing or malformed holdings table
- **WHEN** the parser encounters a statement whose holdings table cannot be parsed (corrupted PDF, format change, missing table)
- **THEN** the parser raises a typed `StatementParseError` identifying the file and the failing portfolio, and does NOT return partial data for that portfolio

### Requirement: Cache parsed statement output as parquet

The system SHALL persist parsed statement output to a configurable on-disk cache in parquet format, keyed by source filename and statement date, so re-parsing the same statement is not required across sessions.

#### Scenario: Cache hit on repeat read
- **WHEN** a previously parsed statement is requested again with no source file change
- **THEN** the system loads the cached parquet without invoking the PDF parser

#### Scenario: Cache miss on source change
- **WHEN** the source PDF's filename or modification time has changed since last parse
- **THEN** the cache entry is invalidated and the parser re-runs

### Requirement: Map Stashaway asset universe to tradeable tickers

The system SHALL provide a static mapping from Stashaway asset identifiers (as they appear in statements) to tradeable tickers consumable by the existing `data/` provider layer, plus per-asset metadata (asset class, region, sector where available).

#### Scenario: Known Stashaway asset
- **WHEN** the diagnostic requests price history for a Stashaway asset present in the universe map
- **THEN** the system returns the mapped ticker and its asset-class / region / sector metadata, and downstream price retrieval succeeds via the existing `DataProvider` registry

#### Scenario: Unknown Stashaway asset
- **WHEN** a parsed statement contains an asset identifier not present in the universe map
- **THEN** the system raises a typed `UnknownAssetError` identifying the asset and the source portfolio, and the diagnostic does not silently drop the holding

### Requirement: Portfolio data model with role tagging

The system SHALL model each portfolio as an object carrying its name, statement date, total value, holdings, and a set of role tags drawn from `{custom, managed_benchmark, protected, holding}`. A portfolio MAY carry multiple roles simultaneously (for example, a managed Stashaway portfolio is both a `holding` and a `managed_benchmark`).

#### Scenario: Multi-role portfolio
- **WHEN** a portfolio is tagged with both `holding` and `managed_benchmark`
- **THEN** combined-book exposure analysis includes its holdings and benchmark comparison treats it as a comparison target, with no double-counting

#### Scenario: Protected portfolio
- **WHEN** a portfolio is tagged `protected`
- **THEN** combined-book exposure analysis includes its holdings, but no diagnostic output marks any of its holdings as a consolidation or change candidate

### Requirement: Reconstruct portfolio return series from current weights and historical prices

The system SHALL compute a daily return time series for each portfolio as the weighted sum of the historical returns of its current holdings, using current weights held constant across the entire historical window. The system SHALL NOT attempt to reconstruct historical weight changes from transaction data.

#### Scenario: Return reconstruction over a chosen window
- **WHEN** a return series is requested for a portfolio over a `[start, end]` window with all underlying tickers having price data over that window
- **THEN** the returned series has one entry per trading day in the window equal to `Σᵢ wᵢ · rᵢ,t`, where wᵢ is the current weight of holding i and rᵢ,t is its daily return

#### Scenario: Holding with insufficient price history
- **WHEN** any holding's price history does not cover the requested window (e.g., a recently added ETF)
- **THEN** the system EITHER returns a series truncated to the common available window AND emits a warning identifying the limiting holding(s), OR raises a typed `InsufficientHistoryError` if strict mode is requested

### Requirement: Diagnostic functions accept an optional date-range parameter

All diagnostic functions that consume return series (correlation, risk contribution, performance comparison) SHALL accept an optional `(start, end)` date range parameter and default to the full available history when unspecified. This requirement exists to keep regime-conditional analysis (Phase 3) cheap to add later.

#### Scenario: Default full-history call
- **WHEN** a diagnostic function is called without a date-range parameter
- **THEN** the function uses the full common history of the input series

#### Scenario: Window-restricted call
- **WHEN** a diagnostic function is called with `(start, end)`
- **THEN** the function restricts all input series to that window before computing, with no other behaviour change
