from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoalpha.config import ResearchConfig
from autoalpha.service.evaluator import _walk_forward_dates
from autoalpha.service.research_protocol import (
    default_task_protocol,
    protocol_blockers,
    task_research_config,
)


def test_recent_task_gets_an_independent_valid_protocol() -> None:
    base = ResearchConfig.from_toml(Path("config/research.toml"))
    protocol = default_task_protocol("2025-06-01", "2026-07-16", base)

    assert protocol_blockers(
        protocol, data_start="2025-06-01", data_end="2026-07-16"
    ) == []
    task_config = task_research_config(base, protocol, task_id="task-recent")
    assert task_config.splits.train.start.isoformat() == "2025-06-01"
    assert task_config.splits.validation.end.isoformat() == protocol["validation_end"]
    assert task_config.splits.test.end.isoformat() == "2026-07-16"
    assert task_config.governance.protocol_version != base.governance.protocol_version


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

    blockers = protocol_blockers(
        protocol, data_start="2026-01-01", data_end="2026-03-01"
    )

    assert any("互不重叠" in blocker for blocker in blockers)
    assert sum("至少需要" in blocker for blocker in blockers) == 3
