"""Diagnostic engine + HTML report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hailmary.allocation.diagnostic import (
    PortfolioDroppedError,
    benchmark_comparison,
    combined_exposure,
    correlation_matrix,
    redundancy_pairs,
    render_html_report,
    risk_contribution,
)
from hailmary.allocation.portfolios import Holding, Portfolio, Role
from hailmary.allocation.universe import STASHAWAY_UNIVERSE


def test_combined_exposure_dimensions(synthetic_book: object) -> None:
    out = combined_exposure(synthetic_book)
    assert set(out.keys()) == {"asset_class", "region", "sector"}
    for df in out.values():
        np.testing.assert_allclose(df["weight"].sum(), 1.0, atol=1e-9)


def test_combined_exposure_excludes_non_holding(
    make_portfolio: object, synthetic_book: object
) -> None:
    # Add a CUSTOM-only (non-HOLDING) portfolio — it should be ignored.
    extra = make_portfolio(
        "Watchlist",
        {"GLD": 1.0},
        roles={Role.CUSTOM},
        total_value=999_999,
    )
    book_with_extra = [*synthetic_book, extra]
    base = combined_exposure(synthetic_book)
    extended = combined_exposure(book_with_extra)
    pd.testing.assert_frame_equal(base["sector"], extended["sector"])


def test_correlation_matrix_symmetric(synthetic_book: object, synthetic_returns: object) -> None:
    corr = correlation_matrix(synthetic_book, returns=synthetic_returns)
    assert not corr.empty
    np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-12)


def test_correlation_window_restriction(synthetic_book: object, synthetic_returns: object) -> None:
    full = correlation_matrix(synthetic_book, returns=synthetic_returns)
    windowed = correlation_matrix(
        synthetic_book,
        returns=synthetic_returns,
        start="2022-06-01",
        end="2023-06-01",
    )
    assert not windowed.empty
    # Different windows should give different correlations on noisy data
    assert not np.allclose(full.values, windowed.values)


def test_correlation_excludes_protected_only_portfolio(
    make_portfolio: object, synthetic_book: object, synthetic_returns: object
) -> None:
    cash = make_portfolio(
        "Cash Pool",
        {"BND": 1.0},
        roles={Role.PROTECTED},
        total_value=50_000,
    )
    book = [*synthetic_book, cash]
    corr = correlation_matrix(book, returns=synthetic_returns)
    assert "Cash Pool" not in corr.index
    assert "Cash Pool" not in corr.columns


def test_benchmark_comparison_excludes_protected_only_portfolio(
    make_portfolio: object, synthetic_book: object, synthetic_returns: object
) -> None:
    cash = make_portfolio(
        "Cash Pool",
        {"BND": 1.0},
        roles={Role.PROTECTED},
        total_value=50_000,
    )
    book = [*synthetic_book, cash]
    df = benchmark_comparison(book, returns=synthetic_returns)
    assert "Cash Pool" not in df.index


def test_redundancy_protected_excluded_as_candidate(
    synthetic_book: object, synthetic_returns: object
) -> None:
    # Force a high-correlation pair: build a portfolio that overlaps Custom Defensive heavily.
    from datetime import date

    from hailmary.allocation.portfolios import Holding, Portfolio
    from hailmary.allocation.universe import STASHAWAY_UNIVERSE

    twin = Portfolio(
        name="Cash+Bonds Twin",
        statement_date=date(2024, 9, 30),
        total_value=100_000,
        currency="USD",
        holdings=[
            Holding("BND", 0.6, 60_000, STASHAWAY_UNIVERSE["BND"]),
            Holding("BNDX", 0.3, 30_000, STASHAWAY_UNIVERSE["BNDX"]),
            Holding("GLD", 0.1, 10_000, STASHAWAY_UNIVERSE["GLD"]),
        ],
        roles={Role.CUSTOM, Role.HOLDING},
    )
    book = [*synthetic_book, twin]
    corr = correlation_matrix(book, returns=synthetic_returns)
    pairs = redundancy_pairs(corr, threshold=0.5, portfolios=book)
    # Custom Defensive is PROTECTED in synthetic_book; pair must list twin as candidate.
    matched = [p for p in pairs if {p[0], p[1]} == {"Custom Defensive", "Cash+Bonds Twin"}]
    if matched:  # only if correlation actually crosses threshold
        assert matched[0][3] != "Custom Defensive"


def test_redundancy_empty_below_threshold(
    synthetic_book: object, synthetic_returns: object
) -> None:
    corr = correlation_matrix(synthetic_book, returns=synthetic_returns)
    pairs = redundancy_pairs(corr, threshold=0.99, portfolios=synthetic_book)
    assert pairs == []


def test_risk_contribution_sums_to_total(synthetic_book: object, synthetic_returns: object) -> None:
    out = risk_contribution(synthetic_book, returns=synthetic_returns)
    by_holding = out["by_holding"]
    by_portfolio = out["by_portfolio"]
    assert not by_holding.empty
    assert not by_portfolio.empty
    np.testing.assert_allclose(by_holding["pct_total"].sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(by_portfolio["pct_total"].sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        by_holding["contribution"].sum(),
        by_portfolio["contribution"].sum(),
        atol=1e-9,
    )


def test_benchmark_comparison_has_deltas(synthetic_book: object, synthetic_returns: object) -> None:
    df = benchmark_comparison(synthetic_book, returns=synthetic_returns)
    assert "sharpe" in df.columns
    assert "max_dd" in df.columns
    assert "annualised_vol" in df.columns
    bench_cols = [c for c in df.columns if c.startswith("sharpe_delta_vs_")]
    assert bench_cols  # at least one benchmark in synthetic_book


def test_benchmark_comparison_warns_when_no_benchmark(
    make_portfolio: object, synthetic_returns: object
) -> None:
    book = [
        make_portfolio("A", {"VTI": 1.0}, roles={Role.CUSTOM, Role.HOLDING}),
        make_portfolio("B", {"BND": 1.0}, roles={Role.CUSTOM, Role.HOLDING}),
    ]
    with pytest.warns(UserWarning, match="No MANAGED_BENCHMARK"):
        df = benchmark_comparison(book, returns=synthetic_returns)
    bench_cols = [c for c in df.columns if c.startswith("sharpe_delta_vs_")]
    assert bench_cols == []


def test_render_html_report_strict_raises_on_dropped_portfolio(
    synthetic_book: object, synthetic_returns: object, tmp_path: Path
) -> None:
    """A portfolio holding a ticker absent from the returns DataFrame must surface
    via PortfolioDroppedError instead of silently disappearing from the report."""
    from dataclasses import replace
    from datetime import date

    ghost = Portfolio(
        name="Unmapped Portfolio",
        statement_date=date(2024, 9, 30),
        total_value=100_000,
        currency="USD",
        holdings=[
            Holding(
                stashaway_id="MYSTERY",
                weight=1.0,
                value=100_000,
                metadata=replace(STASHAWAY_UNIVERSE["VTI"], ticker="MYSTERY"),
            )
        ],
        roles={Role.HOLDING, Role.CUSTOM},
    )
    book = [*synthetic_book, ghost]
    out = tmp_path / "report.html"
    with pytest.raises(PortfolioDroppedError) as exc_info:
        render_html_report(book, out, returns=synthetic_returns, strict=True)
    assert any(name == "Unmapped Portfolio" for name, _ in exc_info.value.dropped)


def test_render_html_report_non_strict_warns_and_continues(
    synthetic_book: object, synthetic_returns: object, tmp_path: Path
) -> None:
    """With strict=False the dropped portfolio emits a warning but the report still renders."""
    from dataclasses import replace
    from datetime import date

    ghost = Portfolio(
        name="Unmapped Portfolio",
        statement_date=date(2024, 9, 30),
        total_value=100_000,
        currency="USD",
        holdings=[
            Holding(
                stashaway_id="MYSTERY",
                weight=1.0,
                value=100_000,
                metadata=replace(STASHAWAY_UNIVERSE["VTI"], ticker="MYSTERY"),
            )
        ],
        roles={Role.HOLDING, Role.CUSTOM},
    )
    book = [*synthetic_book, ghost]
    out = tmp_path / "report.html"
    with pytest.warns(UserWarning, match="Unmapped Portfolio"):
        render_html_report(book, out, returns=synthetic_returns, strict=False)
    assert out.exists()


def test_render_html_report_writes_nontrivial_file(
    synthetic_book: object, synthetic_returns: object, tmp_path: Path
) -> None:
    out = tmp_path / "report.html"
    render_html_report(synthetic_book, out, returns=synthetic_returns)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert len(content) > 5_000  # plotly inline alone runs into the hundreds of KB
    assert "Allocation Diagnostic" in content
    assert "Custom Growth" in content
