## Why

The Phase 1 allocation diagnostic (`stashaway-allocation-diagnostic`) answers "what does my book look like *today* and how would it have behaved historically?" — a static snapshot. It does **not** answer the next question the user wants to make decisions on:

> *If I drop Crypto and rotate that capital into Singapore Investing, what changes? Does Sharpe go up? Does exposure concentrate? Does the redundancy disappear? What happens to total book vol?*

That is what-if rebalancing — comparing a **proposed** book against the **current** one across the same diagnostic surface. It is explicitly named as Phase 2 in `stashaway-allocation-diagnostic/design.md` (Goals, line 22; D7 pre-bend) but was deliberately out of scope for Phase 1. With Phase 1 shipped and the user reviewing it now, Phase 2 is the next decision-relevant step.

This change is intentionally scoped to **manual scenario diffing** — the user (not an optimizer) proposes the new weights. Phase 3 (regime-conditional allocation) and an automated optimizer are still out of scope.

## What Changes

- New module `src/hailmary/allocation/scenarios.py` containing:
  - `Scenario` — small dataclass wrapping a `list[Portfolio]` with a label and an optional human-readable note
  - `scenario_compare(current: Scenario, proposed: Scenario, **diagnostic_kwargs) -> ScenarioDiff` — runs every existing diagnostic on both books and computes deltas
  - `ScenarioDiff` — return type carrying paired diagnostic outputs and delta DataFrames (Sharpe Δ, vol Δ, max-DD Δ, exposure Δ, redundancy-pair appearances/disappearances)
  - Helpers for the common edits: `drop_portfolio`, `set_weights`, `rebalance_into`, `merge_into` — each returns a *new* `list[Portfolio]` without mutating the input
- HTML report variant: `render_scenario_report(diff, output_path)` — side-by-side current vs proposed with the diagnostic sections from Phase 1 plus a "What changed" delta table at the top
- New notebook `notebooks/allocation/04_scenario_compare.ipynb` — loads the current statement, demonstrates each edit helper, calls `scenario_compare`, exports the report
- Reuses every existing diagnostic function unchanged. No changes to Phase 1 public APIs.

Out of scope (deferred):

- **Automated optimization** (mean-variance, risk parity, Black-Litterman) — explicit non-goal; the user proposes the scenario manually
- **Regime-conditional analysis** — Phase 3; will consume `scenario_compare` once regimes are defined
- **Multi-statement weight-drift tracking** ("how have my weights actually moved over the last 12 months?") — separate concern (Phase 2B); see `Phase 2B (book history)` notes in the Phase 1 design discussion. May be split into a sibling change `stashaway-book-history`
- **FX-converted scenario totals** — inherits the current USD/SGD mixing caveat from Phase 1 (task 10.9 in `stashaway-allocation-diagnostic`); will benefit from that fix when it lands but is not blocked on it

## Capabilities

### New Capabilities

- `allocation-scenarios`: Express a proposed book as a `Scenario` (immutable wrapper over `list[Portfolio]`), edit it via composable non-mutating helpers, and run the full Phase 1 diagnostic suite on both current and proposed books side-by-side with computed deltas. Produce a self-contained HTML report.

### Modified Capabilities

None. Phase 1 functions (`combined_exposure`, `correlation_matrix`, `redundancy_pairs`, `risk_contribution`, `benchmark_comparison`, `book_performance`, `render_html_report`) are reused unchanged. Phase 1's `Portfolio` model already supports arbitrary `metadata` (D7); no changes needed there either.

## Impact

- **New file**: `src/hailmary/allocation/scenarios.py`
- **New file**: `src/hailmary/allocation/_scenario_report_template.html.j2`
- **New notebook**: `notebooks/allocation/04_scenario_compare.ipynb`
- **New tests**: `tests/allocation/test_scenarios.py` — unit tests for each edit helper, plus an end-to-end "current == proposed → zero deltas" sanity test
- **No changes** to `data/`, `analytics/`, `viz/`, the crypto signal R&D path, or any existing Phase 1 module
- **No new dependencies**
- **Architectural pre-bend for Phase 3**: `ScenarioDiff` carries `(start, end)` through unchanged, so once regimes are defined (Phase 3) a "regime-conditional scenario diff" is one extra parameter, not a refactor
