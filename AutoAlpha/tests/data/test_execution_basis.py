from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autoalpha.data.execution_basis import (
    expression_research_basis_blockers,
    inspect_execution_data_basis,
)


def test_dividend_adjusted_panel_blocks_cash_ledger(tmp_path: Path) -> None:
    (tmp_path / "_metadata.json").write_text(
        json.dumps(
            {
                "price_adjustment": "split_and_dividend_adjusted",
                "volume_unit": "board_lot_100_shares",
                "amount_unit": "thousand_usd",
                "capital_ledger_ready": False,
            }
        ),
        encoding="utf-8",
    )

    basis = inspect_execution_data_basis(tmp_path)

    assert not basis.capital_ledger_ready
    with pytest.raises(RuntimeError, match="unadjusted OHLC"):
        basis.require_capital_ledger()


def test_unadjusted_normalized_panel_allows_cash_ledger(tmp_path: Path) -> None:
    (tmp_path / "_metadata.json").write_text(
        json.dumps(
            {
                "price_adjustment": "unadjusted",
                "volume_unit": "shares",
                "amount_unit": "usd",
                "capital_ledger_ready": True,
            }
        ),
        encoding="utf-8",
    )

    assert inspect_execution_data_basis(tmp_path).capital_ledger_ready


def test_cash_ledger_requires_point_in_time_execution_state(tmp_path: Path) -> None:
    (tmp_path / "_metadata.json").write_text(
        json.dumps(
            {
                "price_adjustment": "unadjusted",
                "volume_unit": "shares",
                "amount_unit": "usd",
                "capital_ledger_ready": True,
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(pa.table({"open": [10.0], "close": [10.1]}), tmp_path / "part.parquet")

    basis = inspect_execution_data_basis(tmp_path)

    assert not basis.capital_ledger_ready
    assert any("point-in-time market state" in blocker for blocker in basis.blockers)


def test_adjusted_price_and_raw_activity_cannot_mix() -> None:
    from autoalpha.data.execution_basis import ExecutionDataBasis

    basis = ExecutionDataBasis(
        price_adjustment="split_and_dividend_adjusted",
        execution_price_adjustment="split_and_dividend_adjusted",
        volume_unit="board_lot_100_shares",
        amount_unit="thousand_usd",
        capital_ledger_ready=False,
        capital_ledger_proxy_ready=False,
        blockers=(),
        proxy_blockers=(),
    )
    mixed = {
        "operator": "divide",
        "parameters": {},
        "arguments": [
            {"operator": "field", "parameters": {"name": "close"}, "arguments": []},
            {"operator": "field", "parameters": {"name": "vol"}, "arguments": []},
        ],
    }
    volume_only = {
        "operator": "rolling_mean",
        "parameters": {"window": 20},
        "arguments": [{"operator": "field", "parameters": {"name": "vol"}, "arguments": []}],
    }

    assert expression_research_basis_blockers(mixed, basis)
    assert not expression_research_basis_blockers(volume_only, basis)


def test_hybrid_panel_allows_only_explicit_non_pit_proxy(tmp_path: Path) -> None:
    (tmp_path / "_metadata.json").write_text(
        json.dumps(
            {
                "price_adjustment": "split_and_dividend_adjusted",
                "execution_price_adjustment": "unadjusted",
                "volume_unit": "shares",
                "amount_unit": "usd",
                "capital_ledger_ready": False,
                "capital_ledger_proxy_ready": True,
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table({"raw_open": [10.0], "raw_close": [10.1], "open": [10.0], "close": [10.1]}),
        tmp_path / "part.parquet",
    )

    basis = inspect_execution_data_basis(tmp_path)

    assert not basis.capital_ledger_ready
    assert basis.capital_ledger_proxy_ready
    basis.require_capital_ledger_proxy()
