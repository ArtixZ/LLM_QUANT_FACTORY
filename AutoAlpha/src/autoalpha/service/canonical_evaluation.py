from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any

import pandas as pd

from autoalpha.config import DateRange, ResearchConfig, SplitConfig, WalkForwardConfig
from autoalpha.dsl.expression import FactorDefinition
from autoalpha.service.evaluator import PriceVolumeEvaluator, _standalone_long_only_metrics

CANONICAL_LIBRARY_PROTOCOL = "canonical_library_long_only_2015_2024_v1"
CANONICAL_MAIN_START = date(2015, 1, 1)
CANONICAL_MAIN_END = date(2024, 12, 31)
CANONICAL_RECENT_START = date(2020, 1, 1)
CANONICAL_RECENT_END = date(2024, 12, 31)


def canonical_library_config(base: ResearchConfig, *, data_start: date) -> ResearchConfig:
    """Build the immutable public-library protocol without touching 2025+ observations."""
    if data_start >= CANONICAL_MAIN_START:
        raise ValueError("Canonical evaluation requires history before 2015")
    training_end = date(CANONICAL_MAIN_START.year - 1, 12, 31)
    hidden_start = date(CANONICAL_MAIN_END.year + 1, 1, 1)
    return replace(
        base,
        splits=SplitConfig(
            train=DateRange(data_start, training_end),
            validation=DateRange(CANONICAL_MAIN_START, CANONICAL_MAIN_END),
            test=DateRange(hidden_start, base.splits.test.end),
            embargo_days=base.splits.embargo_days,
        ),
        walk_forward=WalkForwardConfig(
            train_years=5,
            validation_years=1,
            first_validation_year=CANONICAL_MAIN_START.year,
            last_validation_year=CANONICAL_MAIN_END.year,
            minimum_folds=10,
        ),
        governance=replace(
            base.governance,
            protocol_version=CANONICAL_LIBRARY_PROTOCOL,
        ),
        generation="canonical_library_2015_2024",
    )


def evaluate_canonical_library_factor(
    evaluator: PriceVolumeEvaluator,
    factor: FactorDefinition,
    *,
    trials: int,
    source_task_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one factor for both canonical boards from one deterministic path."""
    if evaluator.config.governance.protocol_version != CANONICAL_LIBRARY_PROTOCOL:
        raise ValueError("Evaluator does not use the canonical library protocol")
    evaluator.set_trial_count(trials)
    main = dict(evaluator.evaluate(factor).metrics)
    strategy_path = evaluator._portfolio_paths([factor], (1.0,))[1]
    recent = strategy_path.loc[
        (strategy_path.index >= pd.Timestamp(CANONICAL_RECENT_START))
        & (strategy_path.index <= pd.Timestamp(CANONICAL_RECENT_END))
    ].copy()
    recent.attrs.update(strategy_path.attrs)
    recent_metrics = _standalone_long_only_metrics(
        recent,
        _recent_metrics_config(evaluator.config),
        trials=trials,
    )
    main.update(
        {
            f"recent_{key}": value
            for key, value in recent_metrics.items()
            if key.startswith("long_only_")
        }
    )
    main.update(
        {
            "evaluation_protocol": CANONICAL_LIBRARY_PROTOCOL,
            "canonical_library_evaluation": True,
            "canonical_main_start": CANONICAL_MAIN_START.isoformat(),
            "canonical_main_end": CANONICAL_MAIN_END.isoformat(),
            "canonical_recent_start": CANONICAL_RECENT_START.isoformat(),
            "canonical_recent_end": CANONICAL_RECENT_END.isoformat(),
            "hidden_period_accessed": False,
        }
    )
    if source_task_metrics is not None:
        task_metrics = source_task_metrics.get("task_research_metrics")
        if not isinstance(task_metrics, dict):
            task_metrics = source_task_metrics.get("source_task_metrics", source_task_metrics)
        main["task_research_metrics"] = task_metrics
        # Backward-compatible alias for existing factor-library consumers.
        main["source_task_metrics"] = task_metrics
    return main


def _recent_metrics_config(canonical: ResearchConfig) -> ResearchConfig:
    return replace(
        canonical,
        splits=replace(
            canonical.splits,
            train=DateRange(canonical.splits.train.start, date(2019, 12, 31)),
            validation=DateRange(CANONICAL_RECENT_START, CANONICAL_RECENT_END),
        ),
        walk_forward=replace(
            canonical.walk_forward,
            first_validation_year=CANONICAL_RECENT_START.year,
            last_validation_year=CANONICAL_RECENT_END.year,
            minimum_folds=5,
        ),
    )
