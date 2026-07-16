from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

Severity = Literal["ERROR", "WARNING"]


class DataContractError(ValueError):
    """A dataset violates a versioned data contract."""


@dataclass(frozen=True)
class FieldSpec:
    name: str
    logical_type: Literal["string", "float", "integer", "boolean", "timestamp", "date"]
    nullable: bool
    description: str
    unit: str | None = None


@dataclass(frozen=True)
class DataQualityIssue:
    severity: Severity
    code: str
    message: str
    count: int


@dataclass(frozen=True)
class DataQualityReport:
    table: str
    rows: int
    issues: tuple[DataQualityIssue, ...]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "ERROR" for issue in self.issues)

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        details = "; ".join(
            f"{issue.code}={issue.count}: {issue.message}"
            for issue in self.issues
            if issue.severity == "ERROR"
        )
        raise DataContractError(f"{self.table} failed its contract: {details}")


@dataclass(frozen=True)
class TableContract:
    name: str
    version: str
    fields: tuple[FieldSpec, ...]
    primary_key: tuple[str, ...]
    event_time: str
    knowledge_time: str
    entity_key: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("Contract field names must be unique")
        required = (
            set(self.primary_key)
            | set(self.entity_key or ())
            | {
                self.event_time,
                self.knowledge_time,
            }
        )
        if not required <= set(names):
            raise ValueError(
                f"Contract is missing key/time fields: {sorted(required - set(names))}"
            )

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def as_of_key(self) -> tuple[str, ...]:
        return self.entity_key or self.primary_key

    def validate(self, frame: pd.DataFrame) -> DataQualityReport:
        issues: list[DataQualityIssue] = []
        field_names = {field.name for field in self.fields}
        missing = field_names - set(frame.columns)
        if missing:
            issues.append(
                DataQualityIssue(
                    "ERROR",
                    "MISSING_COLUMNS",
                    f"missing {sorted(missing)}",
                    len(missing),
                )
            )
            return DataQualityReport(self.name, len(frame), tuple(issues))

        null_keys = int(frame[list(self.primary_key)].isna().any(axis=1).sum())
        if null_keys:
            issues.append(
                DataQualityIssue("ERROR", "NULL_PRIMARY_KEY", "null key fields", null_keys)
            )
        duplicate_keys = int(frame.duplicated(list(self.primary_key)).sum())
        if duplicate_keys:
            issues.append(
                DataQualityIssue("ERROR", "DUPLICATE_PRIMARY_KEY", "duplicate keys", duplicate_keys)
            )
        for field in self.fields:
            if not field.nullable:
                nulls = int(frame[field.name].isna().sum())
                if nulls:
                    issues.append(
                        DataQualityIssue(
                            "ERROR", "NULL_REQUIRED_FIELD", f"{field.name} contains nulls", nulls
                        )
                    )
            if not _matches_type(frame[field.name], field.logical_type):
                issues.append(
                    DataQualityIssue(
                        "ERROR",
                        "INVALID_LOGICAL_TYPE",
                        f"{field.name} is not {field.logical_type}",
                        len(frame),
                    )
                )
        known = pd.to_datetime(frame[self.knowledge_time], utc=True, errors="coerce")
        invalid_known = int(known.isna().sum())
        if invalid_known:
            issues.append(
                DataQualityIssue(
                    "ERROR",
                    "INVALID_KNOWLEDGE_TIME",
                    "knowledge timestamps are invalid",
                    invalid_known,
                )
            )
        return DataQualityReport(self.name, len(frame), tuple(issues))


def _matches_type(series: pd.Series, logical_type: str) -> bool:
    if logical_type == "string":
        return (
            pd.api.types.is_string_dtype(series.dtype)
            or series.dropna().map(lambda value: isinstance(value, str)).all()
        )
    if logical_type == "float":
        return pd.api.types.is_float_dtype(series.dtype)
    if logical_type == "integer":
        return pd.api.types.is_integer_dtype(series.dtype)
    if logical_type == "boolean":
        return pd.api.types.is_bool_dtype(series.dtype)
    if logical_type in {"timestamp", "date"}:
        values = series.dropna()
        return pd.to_datetime(values, errors="coerce").notna().all()
    return False
