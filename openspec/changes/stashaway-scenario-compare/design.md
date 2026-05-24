## Context

Phase 1 (`stashaway-allocation-diagnostic`) ships current-snapshot diagnostics on the user's real ~$3.2M SGD Stashaway book. Section 10b of that change's `tasks.md` documents the user-review phase already underway. The natural next move — and one the Phase 1 design pre-bent for (decision D7) — is **what-if rebalancing**: edit a copy of the current book and compare it against the original across the same diagnostic surface.

Constraints shaping this design:

- **User-driven, not optimizer-driven.** The user proposes the change; we diff it. No mean-variance, no risk parity, no Black-Litterman. Those are explicit Phase 2+ non-goals.
- **Manual workflow, batch cadence.** Day-job constraints rule out intraday; the deployment shape is "open a notebook on a weekend, propose 1–3 scenarios, export the HTML, decide."
- **Phase 1 stays untouched.** `combined_exposure`, `correlation_matrix`, `redundancy_pairs`, `risk_contribution`, `benchmark_comparison`, `book_performance` are all called as-is. Any change to those is a Phase 1 bug fix, not a Phase 2 feature.
- **No new dependencies.** Reuses `pdfplumber`, `jinja2`, plotly already in `[allocation]` extra.

## Goals / Non-Goals

**Goals:**

- A `Scenario` data type that wraps `list[Portfolio]` with a human-readable label
- A small set of composable, non-mutating edit helpers (`drop_portfolio`, `set_weights`, `rebalance_into`, `merge_into`) that return a new portfolio list rather than modifying the input
- `scenario_compare(current, proposed)` that runs every Phase 1 diagnostic on both books and returns paired outputs plus delta tables
- A side-by-side HTML report variant showing current vs proposed with a "what changed" delta header
- Notebook 04 demonstrating the workflow on the user's real book

**Non-Goals:**

- Optimization in any form — manual proposal only
- Regime conditioning — Phase 3
- Multi-statement weight-drift tracking — Phase 2B (separate change)
- Persisting scenarios to disk — they live in the notebook session; if the user wants to save one, they copy the edit code into the notebook
- Anything that mutates `Portfolio` objects in place

## Decisions

### D1 — `Scenario` is a thin wrapper, not a new abstraction

```python
@dataclass(frozen=True)
class Scenario:
    label: str
    portfolios: tuple[Portfolio, ...]
    note: str | None = None
```

It's a `tuple` (not `list`) of portfolios so equality + hashing work and accidental mutation is harder. `label` is for the HTML report ("Current book", "Drop Crypto → Singapore Investing"). `note` is freeform.

**Alternatives considered:**

- *Bare `list[Portfolio]` everywhere*: rejected — the diff report needs a label, and threading two unlabelled lists through `scenario_compare(a, b)` gets confusing fast.
- *Subclass of `list`*: rejected — adds nothing; `tuple` + dataclass is enough.

### D2 — Edit helpers return new portfolio lists, never mutate

Every helper has the signature `helper(portfolios: Sequence[Portfolio], ...) -> tuple[Portfolio, ...]`. Holdings are rebuilt via dataclass replace, so the input is never touched.

**Why:** scenario building is iterative — the user may try 5 variations. Mutation makes that fragile. Non-mutation also makes notebook re-runs idempotent.

### D3 — `ScenarioDiff` carries paired outputs, not just deltas

`scenario_compare` returns:

```python
@dataclass
class ScenarioDiff:
    current: Scenario
    proposed: Scenario
    book_performance: tuple[dict, dict]            # (current, proposed)
    exposure: tuple[dict[str, pd.DataFrame], ...]  # (current, proposed)
    correlation: tuple[pd.DataFrame, pd.DataFrame]
    redundancy: tuple[list[tuple], list[tuple]]
    risk: tuple[dict[str, pd.DataFrame], ...]
    benchmarks: tuple[pd.DataFrame, pd.DataFrame]
    deltas: ScenarioDeltas                         # computed
```

We keep both raw outputs so the report can show side-by-side tables. `ScenarioDeltas` is a separate small dataclass holding the *computed* differences (Sharpe Δ per portfolio, redundancy pairs that appeared / disappeared, book-level Sharpe / vol / max-DD shift, exposure shift per bucket).

**Alternative considered:** return only deltas — rejected; the absolute "before" and "after" numbers are decision-relevant on their own.

### D4 — Reuse `render_html_report`'s building blocks, separate template

`render_scenario_report` shares the per-section HTML helpers (`_exposure_to_html`, `_redundancy_to_html`, etc.) but has its own jinja template `_scenario_report_template.html.j2`. The template renders two columns (current | proposed) for each section plus a delta strip at the top.

**Alternative considered:** parametrise `render_html_report` to accept `current` and optional `proposed` — rejected; the conditional branching inside one function and one template would bloat both. A separate template that imports the same partials is cleaner.

### D5 — Edit helpers are intentionally crude; "redo the book from scratch" is also fine

We ship four helpers covering the common manual edits:

- `drop_portfolio(book, name)` — remove a whole portfolio
- `set_weights(book, portfolio_name, {ticker: weight})` — replace one portfolio's weights
- `rebalance_into(book, from_name, to_name)` — move all $ from one portfolio into another
- `merge_into(book, names, into=...)` — combine N portfolios into one

For anything more elaborate, the user constructs a new `Portfolio` directly and passes it. We do not try to anticipate every edit; the helpers are sugar over the data model, not a DSL.

### D6 — Same caveats as Phase 1

The reconstructed return series is current-weights × historical underlying-asset prices — a forward-looking estimate of the proposed book, not a back-test of an actual rebalancing path. The HTML report inherits Phase 1's caveat block verbatim and adds a second caveat clarifying that scenario deltas are estimates under the same projection assumption.

### D7 — No persistence layer for scenarios

Scenarios are values in the notebook session. If the user wants a scenario to survive a kernel restart, they paste the edit code into the notebook. No JSON dump, no DB, no scenario registry — that's optimization-tool ergonomics we don't need at this scale.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| User makes a weight edit that doesn't sum to 1.0 | `set_weights` validates the same way `Portfolio.__post_init__` does; raises with a clear message identifying the portfolio |
| Side-by-side HTML gets unreadable on small screens | Template uses CSS grid with min column width; falls back to stacked layout on narrow viewports |
| `ScenarioDiff` becomes a god-object as we add Phase 1 outputs | Pairs are deliberately tuples of homogeneous types; adding a new diagnostic to Phase 1 means one new field, mechanical change |
| Phase 3 (regime conditioning) wants to reuse this but with windowed inputs | `scenario_compare` already accepts `start`/`end` (D7 from Phase 1 design); no refactor needed |
| User accidentally mutates a `Portfolio` they edited and then runs the diff with stale data | All Phase 1 portfolios are dataclasses with `frozen=True`-equivalent semantics for holdings; `Scenario.portfolios` is a tuple. Mutation is hard by construction |

## Migration Plan

Greenfield additive. No migration needed. No existing public APIs change.

Rollback: delete `src/hailmary/allocation/scenarios.py`, the template, the notebook, and the test file. Phase 1 unaffected.

## Open Questions

- **Where to surface the "what changed" summary in the report**: top of report vs interleaved per-section. Default to **top** for v1 (decision-relevant at a glance); revisit after first real use.
- **Should the delta strip show absolute or % change for Sharpe / vol?** Default to **absolute** (Sharpe Δ = +0.21 is more actionable than "Sharpe up 28%"). Revisit if it feels wrong on first use.
- **Multi-scenario report** (current vs N proposals at once): not in v1; user runs `scenario_compare` N times if they want N comparisons. Add only if pattern emerges.
- **Linking scenarios back to Phase 2B (book history)** — once book history exists, a scenario could be "rewind to Jan-2026 weights and apply to today's prices." Don't design for this yet; revisit when Phase 2B is on the table.
