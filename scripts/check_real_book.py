"""One-off: parse the real statement, apply role tags, summarise the resolved book."""

from __future__ import annotations

from pathlib import Path

from hailmary.allocation.portfolios import Role, from_parsed
from hailmary.allocation.statements import parse_statement

ROLES: dict[str, set[Role]] = {
    "Simple USD": {Role.PROTECTED},
    "Simple SGD": {Role.PROTECTED},
    "Guitsa": {Role.PROTECTED},
    "General SRS": {Role.HOLDING, Role.PROTECTED},
    "BlackRock": {Role.HOLDING, Role.MANAGED_BENCHMARK},
    "General Investing": {Role.HOLDING, Role.MANAGED_BENCHMARK},
    "Singapore Investing": {Role.HOLDING, Role.MANAGED_BENCHMARK},
    "Income Investing": {Role.HOLDING, Role.MANAGED_BENCHMARK},
    "Energy": {Role.HOLDING, Role.CUSTOM},
    "Utilities": {Role.HOLDING, Role.CUSTOM},
    "High Dividend Yield": {Role.HOLDING, Role.CUSTOM},
    "Ex-US Large-cap": {Role.HOLDING, Role.CUSTOM},
    "SG ETF": {Role.HOLDING, Role.CUSTOM},
    "Nasdaq Covered Call": {Role.HOLDING, Role.CUSTOM},
    "Crypto": {Role.HOLDING, Role.CUSTOM},
}


def main() -> None:
    p = Path("data/statements/2026-04 StashAway Monthly Statement.pdf")
    parsed = parse_statement(p, use_cache=False)
    portfolios = []
    for parsed_pf in parsed:
        roles = ROLES.get(parsed_pf.name)
        if roles is None:
            print(f"NO ROLE for {parsed_pf.name!r}")
            continue
        portfolios.append(from_parsed(parsed_pf, roles=roles))

    print(f"Resolved {len(portfolios)} portfolios")
    print()
    holding = [p for p in portfolios if Role.HOLDING in p.roles]
    print(f"In diagnostic ({len(holding)} HOLDING):")
    for p in holding:
        tags = ",".join(sorted(r.value for r in p.roles))
        print(f"  {p.name:25s}  {p.currency}  {p.total_value:>14,.2f}  [{tags}]")
    print()
    hidden = [p for p in portfolios if Role.HOLDING not in p.roles]
    print(f"Hidden from diagnostic ({len(hidden)}):")
    for p in hidden:
        print(f"  {p.name:25s}  {p.currency}  {p.total_value:>14,.2f}")


if __name__ == "__main__":
    main()
