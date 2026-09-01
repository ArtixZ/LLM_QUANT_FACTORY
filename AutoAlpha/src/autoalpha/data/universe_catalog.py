"""US equity research universes for IBKR downloads.

These are *current* membership lists with no point-in-time history, which makes
them survivorship-biased: today's large caps are, by construction, the names
that survived. They are adequate for wiring and smoke tests and inadequate for
any performance claim. A production universe needs dated index membership.
"""

from __future__ import annotations

from pathlib import Path

SURVIVORSHIP_WARNING = (
    "Built-in universes are current-membership only. Results carry survivorship "
    "bias and must not be promoted to production."
)

# A liquid, sector-spread mega-cap set used for smoke tests and wiring checks.
MEGA_CAP_LIQUID_V1 = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
    "JPM", "V", "MA", "BAC", "WFC", "GS",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE",
    "XOM", "CVX", "COP",
    "WMT", "COST", "HD", "PG", "KO", "PEP", "MCD",
    "CAT", "HON", "GE", "BA", "UNP",
    "LIN", "NEE", "AMT", "PLD",
    "ORCL", "CRM", "ADBE", "AMD", "QCOM", "TXN", "INTC", "CSCO", "IBM", "NFLX", "DIS",
)

DOW_30_V1 = (
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
    "NKE", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT", "AMZN", "NVDA",
)

SMOKE_TEST_V1 = ("AAPL", "MSFT", "JPM", "XOM", "WMT")

UNIVERSES: dict[str, tuple[str, ...]] = {
    "MEGA_CAP_LIQUID_V1": MEGA_CAP_LIQUID_V1,
    "DOW_30_V1": DOW_30_V1,
    "SMOKE_TEST_V1": SMOKE_TEST_V1,
}


def resolve_universe(name_or_path: str) -> tuple[str, tuple[str, ...]]:
    """Resolve a built-in universe name or a newline/comma separated symbol file."""
    key = name_or_path.strip()
    if not key:
        raise ValueError("A universe name or file path is required")
    if key.upper() in UNIVERSES:
        return key.upper(), UNIVERSES[key.upper()]
    path = Path(key).expanduser()
    if not path.is_file():
        known = ", ".join(sorted(UNIVERSES))
        raise ValueError(f"Unknown universe {name_or_path!r}; use a file path or one of: {known}")
    return path.stem, load_symbol_file(path)


def load_symbol_file(path: Path) -> tuple[str, ...]:
    """Read symbols from a file, ignoring blanks and ``#`` comments."""
    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0]
        for token in text.replace(",", " ").split():
            candidate = token.strip().upper()
            if candidate and candidate not in symbols:
                symbols.append(candidate)
    if not symbols:
        raise ValueError(f"No symbols found in {path}")
    return tuple(symbols)
