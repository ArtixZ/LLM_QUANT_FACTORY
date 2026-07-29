from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PRIMARY_CONTEXT_MARKERS = (
    "rank",
    "ranking",
    "sort",
    "score",
    "overall",
    "leader",
    "champion",
    "best",
    "primary",
)

PRIMARY_ACTION_MARKERS = (
    "sorted(",
    "max(",
    "min(",
    "key=",
    "ranked",
    "leader",
    "champion",
    "best",
    "_rank",
    "prefilter_score",
    "def score",
    "score(",
)

LEGACY_PRIMARY_METRICS = ("sharpe_ratio", "simple_annual_return")
LONG_ONLY_OR_PORTFOLIO_MARKERS = ("long_only", "portfolio_")
ALLOWED_LEGACY_MARKERS = (
    "diagnostic",
    "legacy fallback",
    "legacy_fallback",
    "batch diagnostic",
    "score_dimensions",
)
CONTEXT_WINDOW_LINES = 3

IGNORED_PATH_PARTS = {
    ".venv",
    "__pycache__",
    "runtime-full-llm",
    ".pytest_cache",
    "scripts",
    "tests",
}

SCANNED_SUFFIXES = {".py", ".js", ".html", ".md"}


@dataclass(frozen=True)
class MetricConventionIssue:
    path: str
    line: int
    severity: str
    evidence: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_long_only_metric_convention(
    root: Path,
    *,
    max_issues: int = 50,
) -> dict[str, Any]:
    """Find likely legacy long-short primary metric usage.

    Long-short IC and spread metrics are still useful diagnostics. This checker
    only flags bare legacy metrics when they appear in ranking/leader/default
    contexts where the platform must prefer long-only evidence.
    """

    issues: list[MetricConventionIssue] = []
    scanned_files = 0
    for path in sorted(root.rglob("*")):
        if len(issues) >= max_issues:
            break
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        if any(part in IGNORED_PATH_PARTS for part in path.parts):
            continue
        scanned_files += 1
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            start = max(0, line_number - CONTEXT_WINDOW_LINES - 1)
            stop = min(len(lines), line_number + CONTEXT_WINDOW_LINES)
            issue = _line_issue(root, path, line_number, line, lines[start:stop])
            if issue is not None:
                issues.append(issue)
                if len(issues) >= max_issues:
                    break
    return {
        "protocol": "LONG_ONLY_METRIC_CONVENTION_CHECK_V1",
        "status": "PASS" if not issues else "WARN",
        "scanned_files": scanned_files,
        "issue_count": len(issues),
        "max_issues": max_issues,
        "primary_metric_policy": "long_only_metrics_are_primary; long_short_metrics_are_diagnostic",
        "issues": [issue.to_dict() for issue in issues],
    }


def _line_issue(
    root: Path,
    path: Path,
    line_number: int,
    line: str,
    context_lines: list[str],
) -> MetricConventionIssue | None:
    lowered = line.casefold()
    if not any(metric in lowered for metric in LEGACY_PRIMARY_METRICS):
        return None
    context = "\n".join(context_lines).casefold()
    stripped = line.strip()
    if (
        (stripped.startswith('"') or stripped.startswith("'"))
        and ":" in stripped
        and ".get(" not in stripped
        and "[" not in stripped
    ):
        return None
    if any(marker in context for marker in ALLOWED_LEGACY_MARKERS):
        return None
    if any(marker in context for marker in LONG_ONLY_OR_PORTFOLIO_MARKERS):
        return None
    if not any(marker in context for marker in PRIMARY_CONTEXT_MARKERS):
        return None
    if not any(marker in context for marker in PRIMARY_ACTION_MARKERS):
        return None
    return MetricConventionIssue(
        path=str(path.relative_to(root)),
        line=line_number,
        severity="WARN",
        evidence=line.strip()[:220],
        reason="Bare legacy performance metric appears in a primary ranking/selection context.",
    )
