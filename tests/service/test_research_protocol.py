from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autoalpha.config import ResearchConfig
from autoalpha.service.evaluator import _walk_forward_dates
from autoalpha.service.research_protocol import (
    RECENT_FIVE_YEAR_BACKWARD,
    default_task_protocol,
    protocol_blockers,
    recent_five_year_task_protocol,
    task_research_config,
    validation_fold_capacity,
)


def test_recent_task_gets_an_independent_valid_protocol() -> None:
    base = ResearchConfig.from_toml(Path("config/research.toml"))
    protocol = default_task_protocol("2025-06-01", "2026-07-16", base)

    assert protocol_blockers(protocol, data_start="2025-06-01", data_end="2026-07-16") == []
    task_config = task_research_config(base, protocol, task_id="task-recent")
    assert task_config.splits.train.start.isoformat() == "2025-06-01"
    assert task_config.splits.validation.end.isoformat() == protocol["validation_end"]
    assert task_config.splits.test.end.isoformat() == "2026-07-16"
    assert task_config.governance.protocol_version != base.governance.protocol_version


def test_recent_five_year_protocol_allocates_backward_from_latest_date() -> None:
    protocol = recent_five_year_task_protocol("2010-01-04", "2026-07-16")

    assert protocol == {
        "exploration_start": "2019-01-17",
        "exploration_end": "2024-01-16",
        "validation_start": "2024-01-17",
        "validation_end": "2026-01-16",
        "holdout_start": "2026-01-17",
        "holdout_end": "2026-07-16",
        "minimum_folds": 2,
        "design": RECENT_FIVE_YEAR_BACKWARD,
        "anchor_date": "2026-07-16",
        "exploration_years": 5,
        "validation_years": 2,
        "holdout_months": 6,
    }
    assert protocol_blockers(protocol, data_start="2010-01-04", data_end="2026-07-16") == []


def test_recent_five_year_protocol_rejects_stale_anchor_and_short_history() -> None:
    protocol = recent_five_year_task_protocol("2010-01-04", "2026-07-16")
    protocol["holdout_end"] = "2026-07-15"

    blockers = protocol_blockers(protocol, data_start="2010-01-04", data_end="2026-07-16")

    assert any("重新应用模板" in blocker for blocker in blockers)
    with pytest.raises(ValueError, match="至少需要"):
        recent_five_year_task_protocol("2022-01-01", "2026-07-16")


def test_public_evaluation_dates_do_not_expand_to_whole_calendar_year() -> None:
    base = ResearchConfig.from_toml(Path("config/research.toml"))
    protocol = {
        "exploration_start": "2025-01-01",
        "exploration_end": "2025-08-31",
        "validation_start": "2025-09-01",
        "validation_end": "2025-12-15",
        "holdout_start": "2025-12-16",
        "holdout_end": "2026-03-31",
        "minimum_folds": 1,
    }
    config = task_research_config(base, protocol, task_id="task-dates")
    index = pd.date_range("2025-01-01", "2025-12-31", freq="B")

    selected = _walk_forward_dates(index, config)

    assert selected.min().date().isoformat() == "2025-09-01"
    assert selected.max().date().isoformat() == "2025-12-15"


def test_protocol_validation_rejects_overlap_and_tiny_windows() -> None:
    protocol = {
        "exploration_start": "2026-01-01",
        "exploration_end": "2026-02-01",
        "validation_start": "2026-02-01",
        "validation_end": "2026-02-20",
        "holdout_start": "2026-02-21",
        "holdout_end": "2026-03-01",
        "minimum_folds": 1,
    }

    blockers = protocol_blockers(protocol, data_start="2026-01-01", data_end="2026-03-01")

    assert any("互不重叠" in blocker for blocker in blockers)
    assert sum("至少需要" in blocker for blocker in blockers) == 3


def test_fold_capacity_uses_actual_trading_dates_in_partial_years() -> None:
    protocol = {
        "exploration_start": "2010-01-04",
        "exploration_end": "2018-04-10",
        "validation_start": "2018-04-11",
        "validation_end": "2023-03-26",
        "holdout_start": "2023-03-27",
        "holdout_end": "2026-07-16",
        "minimum_folds": 6,
    }
    dates = pd.bdate_range("2018-04-11", "2023-03-26").date.tolist()
    first_2023_dates = [item for item in dates if item.year == 2023][:5]
    dates = [item for item in dates if item not in first_2023_dates]

    capacity = validation_fold_capacity(protocol, dates)

    assert capacity["maximum_folds"] == 5
    assert capacity["evaluable_years"] == [2018, 2019, 2020, 2021, 2022]
    assert capacity["observations_by_year"][2023] == 55
