## Why

The user holds ~10 custom Stashaway portfolios plus several managed ones (General Investing, Singapore Investing, cash holdings) with up to ~$1M of deployable capital. There is no systematic view of the combined book: actual sector / region / asset-class exposure, redundancy across portfolios, risk contributions, or whether custom allocations beat Stashaway's managed robo-portfolios on a risk-adjusted basis. Decisions about consolidation, expansion, and "is custom worth the effort?" are being made on intuition, which is the wrong default at this scale.

This change establishes the data foundation and diagnostic layer that turns Stashaway statements into actionable book-level analytics. It is Phase 1 of a longer roadmap (static optimization → regime-conditional allocation), and is intentionally scoped to **diagnosis only** — no optimization, no rebalance recommendations, no execution.

## What Changes

- New `src/hailmary/allocation/` package containing portfolio data model and diagnostic engine
- PDF statement parser (pdfplumber-based) that extracts current holdings from Stashaway monthly statements
- Static configuration for the Stashaway-supported asset universe → tradeable ticker mapping (Yahoo or similar provider already present in `data/`)
- Portfolio data model with role tagging (`custom`, `managed_benchmark`, `protected`, `holding`); a portfolio may carry multiple roles
- Current-snapshot return reconstruction (current weights × historical price series) — no transaction history reconstruction
- Diagnostic suite producing:
  - Combined-book exposure across sector / region / asset class
  - Pairwise correlation matrix across all portfolios
  - Redundancy flagging (configurable threshold; default 0.85)
  - Risk contribution per portfolio and combined
  - Performance comparison vs. each `managed_benchmark` portfolio (Sharpe, max drawdown, annualised vol)
- Two validation notebooks (parsed holdings reconcile with statements; reconstructed returns reconcile with statement-reported NAV) and one diagnostic notebook that exports an HTML report
- A small synthetic test fixture used **only for unit tests** — real statements are the primary input from day one

Out of scope (deferred to future phases, called out so they don't creep in):
- Optimization, sizing recommendations, hedging — Phase 2
- Regime detection and regime-conditional allocation — Phase 3 (will consume this phase's data layer + the crypto signal R&D path as regime classifiers)
- Universe-gap / expansion analysis — gated on enumerating the full Stashaway asset universe; tracked as Phase 1B follow-up
- Transaction history reconstruction (daily-rebalance noise makes this low-value for forward decisions)
- Anything in the crypto / signal R&D path (paused while this ships)

## Capabilities

### New Capabilities

- `stashaway-portfolio-data`: Ingest Stashaway monthly PDF statements into a structured `Portfolio` data model with role tagging, plus a static Stashaway-asset → tradeable-ticker universe map. Provides current-snapshot return reconstruction using current weights and historical price data fetched via the existing `data/` layer.
- `allocation-diagnostic`: Compute combined-book exposure (sector / region / asset class), pairwise portfolio correlation, redundancy flagging, risk contribution, and performance comparison vs. tagged benchmarks for a collection of `Portfolio` objects. Produces both programmatic outputs (DataFrames, plotly figures) and a renderable HTML report.

### Modified Capabilities

None. This is a greenfield additive change.

## Impact

- **New module**: `src/hailmary/allocation/` (package with `universe.py`, `statements.py`, `portfolios.py`, `returns.py`, `diagnostic.py`)
- **New notebooks**: `notebooks/01_validate_holdings.ipynb`, `notebooks/02_validate_returns.ipynb`, `notebooks/03_allocation_diagnostic.ipynb` (the last produces the HTML report)
- **New dependency**: `pdfplumber` (statement parsing). Added to `pyproject.toml` extras under a new `[allocation]` extra to keep the core install lean
- **Reused, no changes**: `data/` (price history via existing `DataProvider`s), `analytics/risk.py` (covariance, risk contribution), `analytics/metrics.py` (Sharpe, drawdown, vol), `viz/theme.py`
- **No changes to existing public APIs.** The crypto signal R&D path (`models/`, `backtest/`, existing notebooks) is untouched and paused for the duration
- **Architectural pre-bend for Phase 3** (small, costs nothing now): diagnostic functions accept an optional date-range parameter (defaults to full history); `Portfolio` carries a `metadata: dict` field for arbitrary tags. These keep regime-conditional analysis cheap to add later
- **Data acquisition risk**: real statement PDFs may have inconsistent formatting or table structures. Buffer of half-day to a full weekend allocated for parser; manual JSON entry remains a fallback if pdfplumber blows up on the actual format
