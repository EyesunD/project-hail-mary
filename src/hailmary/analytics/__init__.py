"""Analytics: performance metrics and signal analytics."""

from hailmary.analytics.metrics import PerformanceMetrics
from hailmary.analytics.signal_analytics import (
    D5_STATS_DOCS,
    FILL_METHOD_DOCS,
    QUALITY_FLAG_DOCS,
    TRADE_STATS_DOCS,
    TRADE_SUMMARY_DOCS,
    SignalAllocationPerformance,
    SignalTradePerformance,
    docs_markdown,
)
from hailmary.analytics.signal_comparison import SignalComparison

__all__ = [
    "D5_STATS_DOCS",
    "FILL_METHOD_DOCS",
    "PerformanceMetrics",
    "QUALITY_FLAG_DOCS",
    "SignalAllocationPerformance",
    "SignalComparison",
    "SignalTradePerformance",
    "TRADE_STATS_DOCS",
    "TRADE_SUMMARY_DOCS",
    "docs_markdown",
]
