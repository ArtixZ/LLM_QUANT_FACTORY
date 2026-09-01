from __future__ import annotations

from pathlib import Path

import pytest

from autoalpha.data.universe_catalog import (
    SMOKE_TEST_V1,
    UNIVERSES,
    load_symbol_file,
    resolve_universe,
)


def test_builtin_universes_are_unique_and_non_empty() -> None:
    for name, symbols in UNIVERSES.items():
        assert symbols, f"{name} is empty"
        assert len(set(symbols)) == len(symbols), f"{name} contains duplicates"


def test_resolve_universe_by_name() -> None:
    name, symbols = resolve_universe("smoke_test_v1")
    assert name == "SMOKE_TEST_V1"
    assert symbols == SMOKE_TEST_V1


def test_resolve_universe_from_file(tmp_path: Path) -> None:
    path = tmp_path / "my_universe.txt"
    path.write_text("AAPL, MSFT  # tech\n\n# comment line\nJPM\naapl\n")
    name, symbols = resolve_universe(str(path))
    assert name == "my_universe"
    assert symbols == ("AAPL", "MSFT", "JPM")


def test_resolve_universe_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown universe"):
        resolve_universe("NOT_A_UNIVERSE")


def test_load_symbol_file_rejects_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("# nothing but comments\n")
    with pytest.raises(ValueError, match="No symbols"):
        load_symbol_file(path)
