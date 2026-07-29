#!/usr/bin/env python3
"""Validate the repository's public-release hygiene and example snapshot."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "examples" / "public_research_snapshot"

REQUIRED_PATHS = (
    "README.md",
    "README_EN.md",
    "AGENTS.md",
    "LICENSE",
    "NOTICE",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "ROADMAP.md",
    "CITATION.cff",
    "AutoAlpha/.env.example",
    "AutoAlpha/LICENSE",
    "AutoAlpha/NOTICE",
    ".github/workflows/ci.yml",
    "examples/public_research_snapshot/README.md",
    "examples/public_research_snapshot/manifest.json",
    "docs/assets/screenshots/01-autoalpha-research-loop.png",
    "docs/assets/screenshots/02-autocombine.png",
    "docs/assets/screenshots/03-quantcombine.png",
    "docs/assets/screenshots/04-llm-research-team.png",
    "docs/assets/screenshots/05-factor-knowledge-base.png",
    "docs/assets/screenshots/06-factor-screener.png",
    "docs/assets/screenshots/07-manual-backtest.png",
    "docs/assets/community/wechat-llm-quant-factory.png",
    "docs/SOURCE_AVAILABLE_CHECKLIST.md",
)

SNAPSHOT_FILES = (
    "factors.jsonl",
    "combinations.jsonl",
    "strategy_spec.json",
    "audit_events.jsonl",
)

TEXT_SUFFIXES = {
    "",
    ".cff",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

PROHIBITED_SUFFIXES = {
    ".arrow",
    ".db",
    ".duckdb",
    ".feather",
    ".key",
    ".p12",
    ".parquet",
    ".pem",
    ".sqlite",
    ".sqlite3",
}

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[oprsu]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"-{5}BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-{5}"),
)

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")


def candidate_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        ROOT / item.decode()
        for item in result.stdout.split(b"\0")
        if item and (ROOT / item.decode()).is_file()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def check_required_paths(errors: list[str]) -> None:
    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).exists():
            errors.append(f"missing required release file: {relative}")


def check_repository_files(files: list[Path], errors: list[str]) -> None:
    local_home_marker = "/" + "Users/"
    blocked_phrases = (
        "PRIVATE " + "REPOSITORY",
        "PROPRIETARY " + "AND CONFIDENTIAL",
    )

    for path in files:
        relative = path.relative_to(ROOT)
        lower_name = path.name.lower()
        if lower_name == ".env" or (
            lower_name.startswith(".env.") and lower_name != ".env.example"
        ):
            errors.append(f"credential-bearing environment file is publishable: {relative}")

        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            errors.append(f"runtime/data artifact is publishable: {relative}")

        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if local_home_marker in text:
            errors.append(f"local macOS home path found: {relative}")
        for phrase in blocked_phrases:
            if phrase in text:
                errors.append(f"private-license phrase found in: {relative}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible credential found in: {relative}")


def check_markdown_links(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:", "tel:")):
                continue
            target = target.split(maxsplit=1)[0]
            target = unquote(target.split("#", maxsplit=1)[0])
            if target and not (path.parent / target).resolve().exists():
                relative = path.relative_to(ROOT)
                errors.append(f"broken local link in {relative}: {raw_target}")


def check_snapshot(errors: list[str]) -> None:
    manifest_path = SNAPSHOT / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        errors.append(f"public snapshot manifest is invalid JSON: {error}")
        return
    if manifest.get("schema_version") != "autoalpha-public-research-snapshot-v1":
        errors.append("public snapshot has an unexpected schema version")
    for flag in (
        "research_only",
        "contains_market_data",
        "contains_hidden_test_metrics",
        "contains_credentials",
    ):
        expected = flag == "research_only"
        if manifest.get(flag) is not expected:
            errors.append(f"public snapshot safety flag is invalid: {flag}")

    manifest_files = manifest.get("files", {})
    if not isinstance(manifest_files, dict):
        errors.append("public snapshot manifest files must be an object")
        return
    missing_snapshot_files = False
    for name in SNAPSHOT_FILES:
        path = SNAPSHOT / name
        if not path.is_file():
            errors.append(f"public snapshot is missing {name}")
            missing_snapshot_files = True
            continue
        metadata = manifest_files.get(name)
        if not isinstance(metadata, dict):
            errors.append(f"public snapshot manifest is missing metadata for {name}")
            continue
        if metadata.get("bytes") != path.stat().st_size:
            errors.append(f"public snapshot byte count does not match: {name}")
        if metadata.get("sha256") != sha256(path):
            errors.append(f"public snapshot checksum does not match: {name}")

    if missing_snapshot_files:
        return

    counts = manifest.get("record_counts", {})
    expected_counts = {
        "factors": len(load_jsonl(SNAPSHOT / "factors.jsonl")),
        "combinations": len(load_jsonl(SNAPSHOT / "combinations.jsonl")),
        "strategies": 1,
        "audit_events": len(load_jsonl(SNAPSHOT / "audit_events.jsonl")),
    }
    if counts != expected_counts:
        errors.append(
            f"public snapshot record counts do not match: expected {expected_counts}, got {counts}"
        )

    previous_hash = "0" * 64
    for index, row in enumerate(load_jsonl(SNAPSHOT / "audit_events.jsonl"), start=1):
        record_hash = row.pop("record_hash", None)
        if row.get("previous_hash") != previous_hash:
            errors.append(f"public audit chain previous hash mismatch at record {index}")
            break
        expected_hash = hashlib.sha256(canonical_json(row).encode()).hexdigest()
        if record_hash != expected_hash:
            errors.append(f"public audit chain record hash mismatch at record {index}")
            break
        previous_hash = expected_hash


def check_licenses(errors: list[str]) -> None:
    root_license = (ROOT / "LICENSE").read_bytes()
    component_license = (ROOT / "AutoAlpha" / "LICENSE").read_bytes()
    if root_license != component_license:
        errors.append("root and AutoAlpha license texts differ")
    required_markers = (
        b"PolyForm Noncommercial License 1.0.0",
        b"https://polyformproject.org/licenses/noncommercial/1.0.0",
        b"Required Notice: Copyright 2026 Jiang Jingzhe.",
        b"Commercial use requires separate prior written permission",
    )
    for marker in required_markers:
        if marker not in root_license:
            errors.append(f"LICENSE is missing required marker: {marker.decode()}")
    if b"Apache License" in root_license:
        errors.append("LICENSE still contains the former Apache license")


def main() -> int:
    errors: list[str] = []
    files = candidate_files()
    check_required_paths(errors)
    check_repository_files(files, errors)
    check_markdown_links(files, errors)
    check_snapshot(errors)
    check_licenses(errors)

    if errors:
        print("Public-release checks failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        f"Public-release checks passed for {len(files)} publishable files, "
        f"{len(SNAPSHOT_FILES)} snapshot artifacts, 7 product screenshots, "
        "and 1 community QR image."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
