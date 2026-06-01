"""Build an anonymised JSON fixture from the user's real statement.

Reads the real PDF, parses it, scales all dollar values by a fixed factor,
genericises any user-named portfolios, and writes a JSON file matching the
``load_holdings_from_json`` schema. The fixture is committed to the repo as
``tests/fixtures/statements/example_book.json`` and used as a regression
test that the JSON-loader path keeps working — it does NOT carry any PII or
real dollar amounts.
"""

from __future__ import annotations

import json
from pathlib import Path

from hailmary.allocation.statements import parse_statement

# Scaling factor applied to every dollar amount. Picked to obscure real values
# while keeping ratios and weights intact.
SCALE = 0.0731

# Map user-specific portfolio names to generic ones. Stashaway product names
# (BlackRock, General Investing, etc.) are kept since they're public; only
# the user's own custom-naming gets genericised.
NAME_MAP = {
    "Guitsa": "Cash Pool A",
}


def anonymise(input_path: Path, output_path: Path) -> None:
    portfolios = parse_statement(input_path, use_cache=False)
    payload = {
        "statement_date": portfolios[0].statement_date.isoformat() if portfolios else "2024-09-30",
        "currency": "SGD",
        "portfolios": [
            {
                "name": NAME_MAP.get(p.name, p.name),
                "total_value": round(p.total_value * SCALE, 2),
                "holdings": [
                    {
                        "ticker": h.ticker,
                        "weight": round(h.weight, 6),
                        "value": round(h.value * SCALE, 2),
                    }
                    for h in p.holdings
                ],
            }
            for p in portfolios
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path} — {len(payload['portfolios'])} portfolios")


if __name__ == "__main__":
    anonymise(
        Path("data/statements/2026-04 StashAway Monthly Statement.pdf"),
        Path("tests/fixtures/statements/example_book.json"),
    )
