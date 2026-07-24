from datetime import date
from pathlib import Path

from autoalpha.config import ResearchConfig
from autoalpha.service.canonical_evaluation import (
    CANONICAL_LIBRARY_PROTOCOL,
    canonical_library_config,
)


def test_canonical_library_config_excludes_2025_and_later() -> None:
    base = ResearchConfig.from_toml(Path("config/research.toml"))

    config = canonical_library_config(base, data_start=date(2010, 1, 4))

    assert config.governance.protocol_version == CANONICAL_LIBRARY_PROTOCOL
    assert config.splits.train.start == date(2010, 1, 4)
    assert config.splits.train.end == date(2014, 12, 31)
    assert config.splits.validation.start == date(2015, 1, 1)
    assert config.splits.validation.end == date(2024, 12, 31)
    assert config.splits.test.start == date(2025, 1, 1)
    assert config.walk_forward.first_validation_year == 2015
    assert config.walk_forward.last_validation_year == 2024
