"""Stashaway book ingestion and allocation diagnostics (Phase 1).

Sub-modules:
    universe    — Stashaway-asset → tradeable-ticker map + metadata
    statements  — PDF statement parser, JSON fallback, parquet cache
    portfolios  — Portfolio data model with role tagging
    returns     — Current-snapshot return reconstruction
    diagnostic  — Combined-book diagnostic engine + HTML report
"""

from hailmary.allocation.etf_explorer import (
    build_etf_explorer,
    parse_etf_universe,
    render_etf_explorer_report,
)
from hailmary.allocation.scenarios import (
    Scenario,
    ScenarioDeltas,
    ScenarioDiff,
    ScenarioEditError,
    drop_portfolio,
    merge_into,
    rebalance_into,
    render_scenario_report,
    scenario_compare,
    set_weights,
)
from hailmary.allocation.diagnostic import (
    PortfolioDroppedError,
    benchmark_comparison,
    book_common_history_start,
    book_performance,
    combined_exposure,
    correlation_matrix,
    holdings_reconciliation,
    load_target_weights,
    portfolio_reconciliation,
    redundancy_pairs,
    render_html_report,
    risk_contribution,
)
from hailmary.allocation.portfolios import Holding, Portfolio, Role, from_parsed
from hailmary.allocation.reconcile import (
    Deposit,
    SleeveReconcile,
    build_reconcile,
    load_app_values,
    load_app_values_all_snapshots,
    load_app_values_at_date,
    load_deposits,
    render_reconcile_report,
)
from hailmary.allocation.returns import InsufficientHistoryError, portfolio_returns
from hailmary.allocation.statements import (
    ParsedHolding,
    ParsedPortfolio,
    StatementParseError,
    load_holdings_from_json,
    parse_statement,
)
from hailmary.allocation.universe import (
    STASHAWAY_UNIVERSE,
    AssetMetadata,
    UnknownAssetError,
    resolve,
)

__all__ = [
    "STASHAWAY_UNIVERSE",
    "AssetMetadata",
    "Deposit",
    "Holding",
    "InsufficientHistoryError",
    "ParsedHolding",
    "ParsedPortfolio",
    "Portfolio",
    "PortfolioDroppedError",
    "Role",
    "Scenario",
    "ScenarioDeltas",
    "ScenarioDiff",
    "ScenarioEditError",
    "StatementParseError",
    "UnknownAssetError",
    "benchmark_comparison",
    "build_etf_explorer",
    "book_common_history_start",
    "book_performance",
    "combined_exposure",
    "correlation_matrix",
    "drop_portfolio",
    "from_parsed",
    "holdings_reconciliation",
    "load_holdings_from_json",
    "load_target_weights",
    "merge_into",
    "parse_etf_universe",
    "parse_statement",
    "render_etf_explorer_report",
    "portfolio_reconciliation",
    "portfolio_returns",
    "rebalance_into",
    "redundancy_pairs",
    "render_html_report",
    "render_reconcile_report",
    "SleeveReconcile",
    "build_reconcile",
    "load_app_values",
    "load_app_values_all_snapshots",
    "load_app_values_at_date",
    "load_deposits",
    "render_scenario_report",
    "resolve",
    "risk_contribution",
    "scenario_compare",
    "set_weights",
]
