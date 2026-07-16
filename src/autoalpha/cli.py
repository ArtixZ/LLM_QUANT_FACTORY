from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoalpha.config import ResearchConfig
from autoalpha.data.current_panel import inspect_current_panel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoAlpha institutional research platform")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fingerprint = subparsers.add_parser(
        "fingerprint", help="print the immutable fingerprint for a research configuration"
    )
    fingerprint.add_argument(
        "--config", type=Path, default=Path("config/research.toml"), help="research TOML file"
    )
    fingerprint.add_argument(
        "--checksum",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="include a named data checksum; repeat as needed",
    )
    inspect_data = subparsers.add_parser(
        "inspect-data", help="report price-research and institutional PIT data readiness"
    )
    inspect_data.add_argument("path", type=Path, help="partitioned parquet panel directory")
    return parser


def _checksums(values: list[str]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for value in values:
        name, separator, checksum = value.partition("=")
        if not separator or not name or not checksum:
            raise ValueError(f"Invalid checksum declaration: {value!r}; expected NAME=SHA256")
        checksums[name] = checksum
    return checksums


def main() -> None:
    args = _parser().parse_args()
    if args.command == "fingerprint":
        config = ResearchConfig.from_toml(args.config)
        payload = {
            "name": config.name,
            "generation": config.generation,
            "fingerprint": config.fingerprint(data_checksums=_checksums(args.checksum)),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "inspect-data":
        report = inspect_current_panel(args.path)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
