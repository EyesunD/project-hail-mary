## Context

Project Hail Mary currently exercises a bar-level signal R&D path on crypto (`TrendSignal → BarBacktest → SignalTradePerformance / SignalAllocationPerformance`). The user, however, holds a substantially larger pool of capital (~$1M deployable) at Stashaway in ~10 custom portfolios plus several managed ones. There is no machinery in the codebase for ingesting Stashaway holdings or analysing the combined book.

Constraints shaping the design:

- **Stashaway is a $-denominated, weight-based venue** (not share-level): execution lag is 5–7 days; Singapore Investing, General Investing and crypto-style portfolios rebalance daily internally. We do **not** attempt to reconstruct historical allocations from transactions; we use a **current-snapshot** approach — current weights × historical underlying-asset prices.
- **Asset universe is fixed** by Stashaway, but the user does not have a published universe list for ETFs they don't currently hold. Universe-gap (expansion) analysis is therefore explicitly Phase 1B and gated.
- **Statements are PDFs** (monthly), and the user has multiple statements available locally. They are the primary input — no synthetic-fixture-first development.
- **Day-job constraints** rule out intraday workflows; a slow, batch, weekly/monthly review cadence is the deployment shape.
- **Phase 3 (regime-conditional allocation) is on the roadmap** and will consume both this phase's data layer and the (currently paused) crypto signal R&D path. The design must keep that path cheap, without inflating Phase 1 scope.

## Goals / Non-Goals

**Goals:**

- A `Portfolio` data model that supports role tagging (`custom`, `managed_benchmark`, `protected`, `holding`), with multi-role portfolios as first-class
- A robust-enough PDF parser that ingests current Stashaway monthly statements into structured holdings, with parquet caching for repeat reads
- A static asset-universe map from Stashaway identifiers to tradeable tickers, consumable through the existing `data/` provider layer
- A diagnostic engine that produces, for the combined book: exposure (sector / region / asset-class), pairwise correlation, redundancy flags, risk contribution, and benchmark comparison
- An HTML report exporter that consolidates diagnostic outputs into a single self-contained file
- Architectural shape that makes Phase 2 (optimization) and Phase 3 (regime-conditional allocation) cheap to add later

**Non-Goals:**

- Optimization, sizing recommendations, hedging logic — Phase 2
- Regime detection, regime-conditional allocation — Phase 3
- Universe-gap / expansion analysis — Phase 1B, gated on universe availability
- Transaction history reconstruction (low value for forward decisions; killed by daily-rebalance noise)
- Anything in the crypto signal R&D path (paused while this ships)
- Real-time / event-driven anything

## Decisions

### D1 — Module placement: new top-level `src/hailmary/allocation/` package

We add `allocation/` alongside `data/`, `models/`, `backtest/`, `analytics/`, `viz/`, `cli/`. It depends on `data/` (for prices) and `analytics/` (for risk + metrics) but is independent of `models/` and `backtest/`.

**Alternatives considered:**

- *Folding into `analytics/`*: rejected — allocation work has a distinct data layer (statement parsing, universe mapping) that doesn't belong in `analytics/`.
- *Two separate packages (`stashaway/` + `book_diagnostic/`)*: rejected — the seam isn't real for a Phase 1 of this size; a single package with module-level separation is cleaner.

### D2 — Current-snapshot return reconstruction, not transactional

Each portfolio's return time series is computed as `r_p,t = Σᵢ wᵢ · rᵢ,t`, where `wᵢ` is the **current** weight of holding `i` and `rᵢ,t` is its historical daily return. Weights are held constant across the entire historical window.

**Why:** The user's Stashaway portfolios (especially Simple, General Investing, crypto-flavoured ones) rebalance daily internally; reconstructing actual historical weights from transactions is brittle and pollutes the analysis with rebalance noise that doesn't matter for forward decisions. Recently added ETFs would have no historical weight at all, requiring synthetic inputs anyway.

**Alternative considered:** transaction-based reconstruction — rejected for the reasons above.

**Implication:** the reconstructed return series is a *projection* of the current portfolio onto historical price data, not a real historical track record. This must be clearly labelled in the diagnostic report; Sharpe / DD numbers are forward-looking estimates of the current book, not realised track record.

### D3 — Portfolio role tags, not a class hierarchy

`Portfolio` carries a `roles: set[Role]` field. Diagnostic functions filter or partition by role membership. A managed Stashaway portfolio can be both `holding` and `managed_benchmark` simultaneously without double-counting, because exposure aggregation iterates over `holding`-tagged portfolios while benchmark comparison iterates over `managed_benchmark`-tagged ones.

**Alternative considered:** subclasses (`CustomPortfolio`, `BenchmarkPortfolio`, ...) — rejected because role overlap (the General Investing case) makes a clean subclass hierarchy impossible without virtual or composition tricks. Tags are simpler.

### D4 — Universe map as static config (`universe.py`), not data file

The Stashaway-asset → ticker map lives as a Python dict in `universe.py`, not as JSON / YAML / CSV.

**Why:** Type safety, IDE jump-to-definition, and explicit per-asset metadata (asset class, region, sector) attached at the same site. The map will grow incrementally as new tickers are encountered — being in code makes that easy.

**Alternative considered:** JSON config file — rejected for friction; not enough data to justify a dedicated config format.

**Trade-off:** any change requires editing code and a small commit. Acceptable at this scale.

### D5 — `pdfplumber` for statement parsing, with manual-JSON fallback

PDF parsing is fragile. We commit to `pdfplumber` as the primary path because it's the standard tool for tabular PDF extraction in Python and is permissively licensed. If a statement format defeats it, the parser layer accepts a manual JSON fallback file at the same input position, so the rest of the pipeline keeps working.

**Alternative considered:** `pypdf` (lower-level, no table extraction) — rejected; tables are exactly what we need.

**Risk:** Stashaway may change their statement format. Mitigation: `StatementParseError` identifies the file and portfolio precisely, and a regression test fixture (one anonymised real statement) catches format drift early.

### D6 — Parquet cache mirrors `data/cache.py` pattern

Parsed statements cache to parquet keyed by `(filename, mtime)`. Same pattern as the existing data cache. Cache directory configurable, defaults under `~/.hailmary/cache/statements/`.

### D7 — Architectural pre-bend for Phase 3

Two small choices that cost nothing now and save refactor later:

1. **All diagnostic functions consuming return series accept an optional `(start, end)` argument** defaulting to full common history. Phase 3 will call them with regime-masked windows.
2. **`Portfolio.metadata: dict[str, Any]`** for arbitrary tags. Phase 3 will tag holdings as `defensive` / `growth` / `cyclical` etc.

These are explicitly captured as requirements (see `specs/stashaway-portfolio-data/spec.md`).

### D8 — HTML report via plotly + jinja2

The diagnostic notebook produces interactive plotly charts; the final report is rendered via a small jinja2 template that embeds the figures inline using `plotly.io.to_html(..., include_plotlyjs="inline")` so the file is self-contained.

**Alternative considered:** `nbconvert` of the diagnostic notebook to HTML — rejected because it ties report content to notebook cell ordering and includes irrelevant scaffolding. A dedicated render path is cleaner.

### D9 — Validation notebooks live alongside the diagnostic notebook

`notebooks/01_validate_holdings.ipynb` and `notebooks/02_validate_returns.ipynb` are first-class deliverables, not afterthoughts. They run before the diagnostic and assert reconciliation against statements. A failed assertion means we don't trust the diagnostic and must fix parsing first.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Stashaway statement format drifts and silently breaks the parser | Typed `StatementParseError` per portfolio; regression test on one anonymised real statement; manual-JSON fallback path |
| Some Stashaway ETFs unavailable on Yahoo (e.g., Singapore-domiciled funds, fund-of-funds) | Identify gaps during universe-map construction; substitute with closest tradeable proxy and clearly mark proxied entries; raise `UnknownAssetError` rather than silently dropping |
| Current-snapshot reconstruction misrepresents portfolios that have changed substantially | Document explicitly in the report; treat numbers as forward-looking estimates of the current book, not realised history |
| Highly-correlated portfolios may still differ in tail risk or factor structure (correlation isn't redundancy) | Combined-book exposure decomposition (sector / region / asset class) catches structural differences correlation alone misses; both run in the same report |
| pdfplumber blows up on the actual statement format | Half-day buffer for parser; manual-JSON fallback so the rest of the pipeline can still progress |
| Scope creep into Phase 2 ("just one little optimization function") | Explicit non-goals list; design + specs reviewed before tasks.md so the boundary is firm |

## Migration Plan

Greenfield change; no migration needed. The `pdfplumber` dependency is added under a new `[allocation]` extra in `pyproject.toml` so the core install stays lean for users who don't need allocation work. Notebooks are additive. No existing public APIs change.

Rollback: deleting `src/hailmary/allocation/` and the three new notebooks is sufficient. No data layer changes to undo.

## Open Questions

- **Redundancy threshold default**: locked at 0.85 with a configurable parameter. Will validate against real data; expect to tune once first diagnostic run completes.
- **Benchmark portfolios**: not pre-named in the design. Whichever portfolios come back tagged `managed_benchmark` after first ingest serve this role. The user will tag them in a small config or directly when constructing `Portfolio` objects.
- **Universe map seed list**: built incrementally — first ingest will surface every ticker actually held; we map those, ship Phase 1A. Phase 1B (universe-gap analysis) is gated on enumerating the rest of Stashaway's offerings, which is a separate, small task to be scoped later.
- **Reporting currency**: statements may report in SGD or USD. Decide on a single canonical currency for the report (likely SGD given the user's base) and convert as needed using a daily FX series. To be confirmed when first statement is parsed.
