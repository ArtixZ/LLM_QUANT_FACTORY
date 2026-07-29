from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from autoalpha.data import tushare_feature_sync
from autoalpha.data.research_fields import (
    available_factor_fields,
    build_research_data_capabilities,
    expression_fields,
    field_definitions,
)
from autoalpha.data.tushare_catalog import DEFAULT_PRODUCT_IDS, PRODUCT_BY_ID
from autoalpha.dsl.expression import field, operation
from autoalpha.service.data_center import build_data_capability_matrix, inspect_data_products


class FakePro:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def trade_cal(self, **_: str) -> pd.DataFrame:
        return pd.DataFrame({"cal_date": ["20240102", "20240103"]})

    def daily_basic(self, *, trade_date: str) -> pd.DataFrame:
        self.requests.append(trade_date)
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [trade_date],
                "turnover_rate": [1.0],
            }
        )


def test_feature_sync_is_resumable(monkeypatch: object, tmp_path: Path) -> None:
    pro = FakePro()
    monkeypatch.setattr(tushare_feature_sync, "_pro_api", lambda _: pro)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        tushare_feature_sync,
        "_latest_complete_trade_date",
        lambda _pro, _end: "20240103",
    )
    first = tushare_feature_sync.run_feature_sync(
        token="secret",
        root=tmp_path,
        dataset_ids=["daily_basic"],
        start_date="20240102",
        end_date="20240103",
        requests_per_minute=1_000_000,
    )
    second = tushare_feature_sync.run_feature_sync(
        token="secret",
        root=tmp_path,
        dataset_ids=["daily_basic"],
        start_date="20240102",
        end_date="20240103",
        requests_per_minute=1_000_000,
    )
    assert first["ok"] is True
    assert second["datasets"][0]["pending_processed"] == 0
    assert pro.requests == ["20240102", "20240103"]
    state = json.loads(
        (tmp_path / "data/state/a_feature_daily_basic.json").read_text(encoding="utf-8")
    )
    assert state["completed_dates"] == ["20240102", "20240103"]


def test_catalog_status_and_dynamic_research_fields(tmp_path: Path) -> None:
    feature_root = tmp_path / "data/downloads/a_share_feature_store/daily_basic"
    feature_root.mkdir(parents=True)
    (tmp_path / "data/state").mkdir(parents=True)
    (tmp_path / "data/state/a_feature_daily_basic.json").write_text(
        json.dumps(
            {
                "completed_dates": ["20240102"],
                "failed_dates": {},
                "target_date": "20240102",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(
        feature_root / "20240102.parquet", index=False
    )
    products = inspect_data_products(
        tmp_path,
        selected_products=list(DEFAULT_PRODUCT_IDS),
        panel_columns={"close", "adj_close", "amount", "vol", "turnover_rate"},
    )
    daily = next(item for item in products["products"] if item["dataset_id"] == "daily_basic")
    income = next(item for item in products["products"] if item["dataset_id"] == "income")
    index_weight = next(
        item for item in products["products"] if item["dataset_id"] == "index_weight"
    )
    assert daily["storage_state"] == "READY"
    assert daily["panel_available_fields"] == ["turnover_rate"]
    assert income["download_selectable"] is True
    assert income["storage_state"] == "NOT_DOWNLOADED"
    assert income["research_state"] == "RAW_DOWNLOAD_ONLY_REQUIRES_PIT_INTEGRATION"
    assert index_weight["download_selectable"] is False
    assert index_weight["research_state"] == "CATALOG_ONLY"
    fields = available_factor_fields(
        {"close", "adj_close", "amount", "vol", "turnover_rate", "pe_ttm"}
    )
    assert fields[-2:] == ["turnover_rate", "pe_ttm"]
    definitions = field_definitions(fields, include_open=False)
    assert {definition.name for definition in definitions} == set(fields)
    assert PRODUCT_BY_ID["daily_basic"].documentation_id == 32


def test_data_capability_matrix_separates_proxy_from_production() -> None:
    matrix = build_data_capability_matrix(
        workspace={
            "price_research_ready": True,
            "institutional_pit_ready": False,
            "warnings": [],
            "blockers": [],
        },
        execution_basis={
            "capital_ledger_proxy_ready": True,
            "capital_ledger_ready": False,
            "proxy_blockers": [],
            "blockers": ["missing ST history"],
        },
    )
    by_module = {item["module_id"]: item for item in matrix["rows"]}

    assert matrix["summary"]["research_ready"] is True
    assert matrix["summary"]["non_pit_proxy_ready"] is True
    assert matrix["summary"]["strict_pit_ready"] is False
    assert by_module["manual_backtest_proxy"]["level"] == "PROXY_BACKTEST_READY"
    assert by_module["paper_trading"]["level"] == "PROXY_PAPER_READY"
    assert by_module["strict_capital_ledger"]["level"] == "PRODUCTION_BLOCKED"
    assert "missing ST history" in by_module["strict_capital_ledger"]["blockers"]


def test_data_capability_matrix_allows_strict_pit_when_market_state_is_ready() -> None:
    matrix = build_data_capability_matrix(
        workspace={
            "price_research_ready": True,
            "institutional_pit_ready": True,
            "warnings": [],
            "blockers": [],
        },
        execution_basis={
            "capital_ledger_proxy_ready": True,
            "capital_ledger_ready": True,
            "proxy_blockers": [],
            "blockers": [],
        },
    )
    production = next(
        item for item in matrix["rows"] if item["module_id"] == "strict_capital_ledger"
    )

    assert matrix["summary"]["production_allowed"] is True
    assert production["allowed"] is True
    assert production["level"] == "STRICT_PIT_READY"


def test_llm_data_contract_exposes_staged_fields_without_unlocking_them(tmp_path: Path) -> None:
    metadata_path = tmp_path / "_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "sources": {
                    "feature_coverage": {
                        "daily_basic": {
                            "first_date": "20260716",
                            "last_date": "20260716",
                            "files": 1,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    workspace = SimpleNamespace(
        metadata_path=str(metadata_path),
        columns=("close", "adj_close", "amount", "vol", "turnover_rate"),
        factor_fields=("close", "adj_close", "amount", "vol"),
    )

    contract = build_research_data_capabilities(workspace)
    turnover = next(item for item in contract["field_catalog"] if item["name"] == "turnover_rate")
    daily_basic = next(
        item for item in contract["data_products"] if item["dataset_id"] == "daily_basic"
    )

    assert turnover["status"] == "STAGED_COVERAGE_INCOMPLETE"
    assert turnover["allowed_in_expression"] is False
    assert contract["eligible_extended_fields"] == []
    assert daily_basic["research_state"] == "STAGED_COVERAGE_INCOMPLETE"


def test_llm_data_contract_automatically_unlocks_bounded_historical_features(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "sources": {
                    "feature_coverage": {
                        "daily_basic": {
                            "first_date": "20200101",
                            "last_date": "20241231",
                            "files": 1200,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    workspace = SimpleNamespace(
        metadata_path=str(metadata_path),
        first_trade_date="2020-01-01",
        last_trade_date="2025-01-03",
        columns=("close", *PRODUCT_BY_ID["daily_basic"].panel_fields),
        factor_fields=("close",),
    )

    contract = build_research_data_capabilities(workspace)

    assert "turnover_rate" in contract["eligible_extended_fields"]
    daily = next(
        item for item in contract["data_products"] if item["dataset_id"] == "daily_basic"
    )
    assert daily["research_state"] == "RESEARCH_ELIGIBLE"
    assert daily["coverage_gate"]["end_lag_days"] == 3


def test_expression_field_lineage_is_extracted_from_typed_tree() -> None:
    expression = operation("divide", field("net_mf_amount"), field("amount"))

    assert expression_fields(expression) == {"net_mf_amount", "amount"}
