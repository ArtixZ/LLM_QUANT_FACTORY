from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from autoalpha.config import ResearchConfig
from autoalpha.dsl.expression import FactorDefinition, field
from autoalpha.service.blind_evaluator import BlindEvaluationBoundary, BlindVerdict


def test_blind_boundary_returns_only_verdict_and_hash(tmp_path: Path) -> None:
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    dates = pd.bdate_range("2024-11-01", "2026-07-14")
    symbols = [f"S{index:03d}" for index in range(40)]
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0003, 0.01, size=(len(dates), len(symbols)))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), dates, symbols)
    fields = {
        "open": prices,
        "close": prices,
        "adj_close": prices,
        "amount": pd.DataFrame(rng.lognormal(16, 0.3, prices.shape), dates, symbols),
        "vol": pd.DataFrame(rng.lognormal(12, 0.3, prices.shape), dates, symbols),
    }
    boundary = BlindEvaluationBoundary(tmp_path, config)
    boundary._load_holdout_fields = lambda: fields  # type: ignore[method-assign]
    factor = FactorDefinition("amount", "test", "test", field("amount"))

    result = boundary.evaluate_holdout([factor], (1.0,))

    assert isinstance(result, BlindVerdict)
    assert set(result.__dict__) == {"passed", "verdict", "evidence_hash"}
    assert len(result.evidence_hash) == 64
