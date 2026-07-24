from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from autoalpha.config import DateRange, ResearchConfig, SplitConfig

CONFIG_PATH = Path(__file__).parents[1] / "config" / "research.toml"


def test_research_config_round_trip_and_stable_fingerprint() -> None:
    config = ResearchConfig.from_toml(CONFIG_PATH)

    first = config.fingerprint(data_checksums={"panel": "abc", "calendar": "def"})
    reordered = config.fingerprint(data_checksums={"calendar": "def", "panel": "abc"})

    assert first == reordered
    assert len(first) == 64
    assert config.canonical_dict()["splits"]["test"]["start"] == "2025-01-02"
    assert config.generation == "institutional_v8_ashare_long_only_primary_20260717"
    assert config.strategy_evaluation.engine_protocol == (
        "A_SHARE_LONG_ONLY_WEEKLY_VECTOR_PROXY_V1"
    )
    assert config.governance.protocol_version == (
        "institutional_walkforward_v8_ashare_long_only_primary"
    )
    assert config.adaptive_direction.maximum_attempts_per_campaign == 3
    assert config.walk_forward.first_validation_year == 2015
    assert config.portfolio.holding_period_days == 5
    assert config.evaluation.minimum_incremental_net_ir == 0.10


def test_fingerprint_changes_with_split_or_data() -> None:
    config = ResearchConfig.from_toml(CONFIG_PATH)
    changed_splits = replace(
        config.splits,
        test=DateRange(start=date(2024, 12, 5), end=config.splits.test.end),
    )
    changed = replace(config, splits=changed_splits)

    assert config.fingerprint() != changed.fingerprint()
    assert config.fingerprint(data_checksums={"panel": "a"}) != config.fingerprint(
        data_checksums={"panel": "b"}
    )


def test_split_ranges_must_be_disjoint_and_ordered() -> None:
    with pytest.raises(ValueError, match="ordered and disjoint"):
        SplitConfig(
            train=DateRange(date(2020, 1, 1), date(2021, 1, 1)),
            validation=DateRange(date(2020, 12, 1), date(2022, 1, 1)),
            test=DateRange(date(2023, 1, 1), date(2024, 1, 1)),
        )
