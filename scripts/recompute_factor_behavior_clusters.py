#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from autoalpha.service.factor_behavior import BehaviorRunConfig, FactorBehaviorRunner
from autoalpha.service.store import ServiceStore


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Recompute long-only vector behavior fingerprints and factor clusters."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=project / "runtime-full-llm/autoalpha.sqlite3",
    )
    parser.add_argument("--data", type=Path, default=project.parent / "data")
    parser.add_argument("--config", type=Path, default=project / "config/research.toml")
    parser.add_argument(
        "--output", type=Path, default=project / "runtime-full-llm/factor-behavior"
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--cluster-threshold", type=float, default=0.74)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = ServiceStore(args.database.resolve()).factor_pool(limit=5000)
    runner = FactorBehaviorRunner(
        BehaviorRunConfig(
            data_path=args.data.resolve(),
            config_path=args.config.resolve(),
            database_path=args.database.resolve(),
            output_root=args.output.resolve(),
            start_date=args.start,
            end_date=args.end,
            cluster_threshold=args.cluster_threshold,
        ),
        records,
    )
    snapshot = runner.run(resume=not args.no_resume)
    print(
        f"completed={snapshot['evaluated_count']} failed={snapshot['failed_count']} "
        f"clusters={snapshot['cluster_count']} snapshot={runner.config.snapshot_path}"
    )


if __name__ == "__main__":
    main()
