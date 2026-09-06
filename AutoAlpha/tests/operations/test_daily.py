from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from autoalpha.ibkr.client import AccountSummary, OpenOrder, Position
from autoalpha.operations.daily import (
    DailyConfig,
    DailyReport,
    DataHealth,
    _submission_blockers,
    build_target_book,
    load_panel,
    momentum_scores,
)


@pytest.fixture
def config(tmp_path: Path) -> DailyConfig:
    return DailyConfig(
        market_data_root=tmp_path,
        lookback_sessions=10,
        skip_sessions=2,
        position_count=2,
        gross_exposure=0.90,
    )


@pytest.fixture
def panel() -> pd.DataFrame:
    """Three symbols over 30 sessions with deliberately ordered momentum."""
    sessions = pd.bdate_range("2026-01-01", periods=30)
    rows = []
    for symbol, drift in (("AAA", 0.010), ("BBB", 0.005), ("CCC", -0.004)):
        price = 100.0
        for session in sessions:
            price *= 1.0 + drift
            rows.append({"symbol": symbol, "trade_date": session, "close": price})
    return pd.DataFrame(rows)


@pytest.fixture
def health() -> DataHealth:
    return DataHealth(
        panel_last_date="2026-08-28", panel_rows=1000, panel_symbols=3, audit_passed=True
    )


def test_config_rejects_a_lookback_inside_the_skip(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lookback_sessions"):
        DailyConfig(market_data_root=tmp_path, lookback_sessions=10, skip_sessions=10)


def test_config_rejects_bad_exposure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gross_exposure"):
        DailyConfig(market_data_root=tmp_path, gross_exposure=1.5)


def test_config_rejects_empty_book(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="position_count"):
        DailyConfig(market_data_root=tmp_path, position_count=0)


def test_momentum_ranks_the_strongest_drift_first(
    panel: pd.DataFrame, config: DailyConfig
) -> None:
    scores = momentum_scores(panel, config)
    assert list(scores.sort_values(ascending=False).index) == ["AAA", "BBB", "CCC"]


def test_momentum_requires_enough_history(panel: pd.DataFrame, tmp_path: Path) -> None:
    config = DailyConfig(market_data_root=tmp_path, lookback_sessions=500, skip_sessions=21)
    with pytest.raises(ValueError, match="need more than"):
        momentum_scores(panel, config)


def test_target_book_equal_weights_the_top_names(
    panel: pd.DataFrame, config: DailyConfig
) -> None:
    targets, detail = build_target_book(panel, config, net_liquidation=100_000.0)
    assert set(targets) == {"AAA", "BBB"}
    assert [d["symbol"] for d in detail] == ["AAA", "BBB"]
    # 90% gross split across 2 names is $45,000 of budget each.
    for item in detail:
        assert item["target_shares"] == int(45_000.0 // item["price"])


def test_target_book_scales_with_account_size(
    panel: pd.DataFrame, config: DailyConfig
) -> None:
    small, _ = build_target_book(panel, config, net_liquidation=10_000.0)
    large, _ = build_target_book(panel, config, net_liquidation=1_000_000.0)
    assert large["AAA"] > small["AAA"]


def test_target_book_skips_symbols_without_a_price(
    panel: pd.DataFrame, config: DailyConfig
) -> None:
    panel.loc[(panel["symbol"] == "AAA") & (panel["trade_date"] == panel["trade_date"].max()),
              "close"] = float("nan")
    targets, detail = build_target_book(panel, config, net_liquidation=100_000.0)
    assert "AAA" not in targets
    assert all(item["symbol"] != "AAA" for item in detail)


def _report(**overrides: object) -> DailyReport:
    base = {
        "as_of": "2026-08-30",
        "account": "DU1",
        "is_paper": True,
        "net_liquidation": 100_000.0,
        "total_cash": 100_000.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "positions": [],
        "health": DataHealth("2026-08-28", 100, 3, True),
        "picks": [{"symbol": "AAA", "score": 0.5, "price": 10.0, "target_shares": 100}],
        "plan": [
            {"symbol": "AAA", "action": "BUY", "quantity": 100, "order_type": "MOO",
             "limit_price": None, "reference_price": 10.0, "notional": 1000.0}
        ],
        "previews": [],
        "modeled_commission": 0.35,
        "plan_notional": 1000.0,
        "submitted": False,
    }
    base.update(overrides)
    return DailyReport(**base)  # type: ignore[arg-type]


def test_clean_run_is_ok() -> None:
    assert _report().severity == "ok"


def test_stale_symbols_downgrade_to_warn() -> None:
    health = DataHealth("2026-08-28", 100, 3, True, stale_symbols={"XOM": "2026-08-07"})
    assert _report(health=health).severity == "warn"


def test_preview_errors_downgrade_to_warn() -> None:
    assert _report(previews=[{"error": "no contract"}]).severity == "warn"


def test_failed_audit_is_an_error() -> None:
    health = DataHealth("2026-08-28", 100, 3, audit_passed=False)
    assert _report(health=health).severity == "error"


def test_sync_failures_are_an_error() -> None:
    health = DataHealth("2026-08-28", 100, 3, True, sync_failures={"XYZ": "no contract"})
    assert _report(health=health).severity == "error"


def test_title_reflects_submission_state() -> None:
    assert "preview only" in _report().title
    assert "submitted" in _report(submitted=True).title


def test_body_states_plainly_that_nothing_was_sent() -> None:
    body = _report().telegram_body()
    assert "NOT submitted - preview only" in body
    assert "SUBMITTED" not in body


def test_body_marks_a_real_submission() -> None:
    assert "SUBMITTED to the broker" in _report(submitted=True).telegram_body()


def test_body_lists_stale_exclusions() -> None:
    health = DataHealth("2026-08-28", 100, 3, True, stale_symbols={"XOM": "2026-08-07"})
    assert "stale, excluded: XOM" in _report(health=health).telegram_body()


def test_body_reports_a_flat_book() -> None:
    assert "Positions: flat" in _report().telegram_body()


def test_body_reports_an_empty_plan() -> None:
    body = _report(plan=[], plan_notional=0.0, modeled_commission=0.0).telegram_body()
    assert "Orders planned: none" in body


def test_body_fits_a_telegram_message() -> None:
    positions = [
        {"symbol": f"S{i:02d}", "quantity": 100, "average_cost": 10.0, "market_price": 11.0,
         "market_value": 1100.0, "unrealized_pnl": 100.0}
        for i in range(30)
    ]
    assert len(_report(positions=positions).telegram_body()) < 4096


def test_load_panel_excludes_stale_symbols(tmp_path: Path, panel: pd.DataFrame) -> None:
    partition = tmp_path / "processed" / "daily_panel" / "trade_year=2026"
    partition.mkdir(parents=True)
    panel.to_parquet(partition / "part.parquet", index=False)
    config = DailyConfig(market_data_root=tmp_path)

    everything = load_panel(config)
    trimmed = load_panel(config, exclude={"CCC"})

    assert set(everything["symbol"].unique()) == {"AAA", "BBB", "CCC"}
    assert set(trimmed["symbol"].unique()) == {"AAA", "BBB"}


def test_load_panel_limits_rows_to_the_configured_universe(
    tmp_path: Path, panel: pd.DataFrame
) -> None:
    partition = tmp_path / "processed" / "daily_panel" / "trade_year=2026"
    partition.mkdir(parents=True)
    panel.to_parquet(partition / "part.parquet", index=False)

    trimmed = load_panel(
        DailyConfig(market_data_root=tmp_path),
        include={"AAA", "BBB"},
    )

    assert set(trimmed["symbol"].unique()) == {"AAA", "BBB"}


def test_load_panel_requires_partitions(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No panel partitions"):
        load_panel(DailyConfig(market_data_root=tmp_path))


def test_report_round_trips_to_a_dict_with_severity() -> None:
    value = _report().to_dict()
    assert value["severity"] == "ok"
    assert value["account"] == "DU1"
    assert value["health"]["panel_last_date"] == "2026-08-28"


def test_default_config_targets_the_shared_market_data_root() -> None:
    config = DailyConfig()
    assert config.panel_path.name == "daily_panel"
    assert config.quality_report_path.name == "data_quality.json"
    assert config.history_start == date(2016, 1, 1)


def test_submission_preflight_accepts_only_a_clean_dedicated_paper_account() -> None:
    blockers = _submission_blockers(
        health=DataHealth("2026-09-03", 100, 3, True),
        account=AccountSummary("DU1", True, 100_000, 100_000, 100_000),
        managed_account="DU1",
        submission_key="quantfactory-20260903",
        run_date=date(2026, 9, 3),
        previews=[{"warning": ""}],
        unmanaged_positions=[],
        open_orders=[],
    )

    assert blockers == []


def test_submission_preflight_fails_closed_on_operational_risk() -> None:
    position = Position("DU1", "SPY", 1, 10, 100, 100, 1_000, 0)
    open_order = OpenOrder("DU1", "AAPL", "BUY", 1, "MKT", "Submitted", 1, 2, "ref")

    blockers = _submission_blockers(
        health=DataHealth(
            "2026-09-02",
            100,
            3,
            audit_passed=False,
            stale_symbols={"AAPL": "2026-09-01"},
        ),
        account=AccountSummary("U1", False, 100_000, 100_000, 100_000),
        managed_account="DU1",
        submission_key=None,
        run_date=date(2026, 9, 3),
        previews=[{"error": "rejected"}, {"warning": "margin warning"}],
        unmanaged_positions=[position],
        open_orders=[open_order],
    )

    assert len(blockers) == 9
