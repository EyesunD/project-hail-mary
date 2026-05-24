"""Stashaway book ingestion and allocation diagnostics (Phase 1).

Sub-modules:
    universe    — Stashaway-asset → tradeable-ticker map + metadata
    statements  — PDF statement parser, JSON fallback, parquet cache
    portfolios  — Portfolio data model with role tagging
    returns     — Current-snapshot return reconstruction
    diagnostic  — Combined-book diagnostic engine + HTML report
"""

from hailmary.allocation.diagnostic import (
    PortfolioDroppedError,
    benchmark_comparison,
    book_common_history_start,
    book_performance,
    combined_exposure,
    correlation_matrix,
    portfolio_reconciliation,
    redundancy_pairs,
    render_html_report,
    risk_contribution,
)
from hailmary.allocation.portfolios import Holding, Portfolio, Role, from_parsed
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
    "Holding",
    "InsufficientHistoryError",
    "ParsedHolding",
    "ParsedPortfolio",
    "Portfolio",
    "PortfolioDroppedError",
    "Role",
    "StatementParseError",
    "UnknownAssetError",
    "benchmark_comparison",
    "book_common_history_start",
    "book_performance",
    "combined_exposure",
    "correlation_matrix",
    "from_parsed",
    "load_holdings_from_json",
    "parse_statement",
    "portfolio_reconciliation",
    "portfolio_returns",
    "redundancy_pairs",
    "render_html_report",
    "resolve",
    "risk_contribution",
]
