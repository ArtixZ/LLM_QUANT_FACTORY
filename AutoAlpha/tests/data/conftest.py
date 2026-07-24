from __future__ import annotations

import pandas as pd
import pytest

from autoalpha.data.contracts import FieldSpec, TableContract


@pytest.fixture
def membership_contract() -> TableContract:
    return TableContract(
        name="universe_membership",
        version="1.0.0",
        fields=(
            FieldSpec("symbol", "string", False, "Exchange-qualified symbol"),
            FieldSpec("valid_from", "date", False, "Membership start"),
            FieldSpec("valid_to", "date", True, "Membership end"),
            FieldSpec("known_at", "timestamp", False, "When this record became known"),
            FieldSpec("revision", "integer", False, "Source revision"),
            FieldSpec("is_eligible", "boolean", False, "Point-in-time eligibility"),
        ),
        primary_key=("symbol", "valid_from", "revision"),
        event_time="valid_from",
        knowledge_time="known_at",
        entity_key=("symbol", "valid_from"),
    )


@pytest.fixture
def memberships() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(["A", "B", "B"], dtype="string"),
            "valid_from": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "valid_to": [None, None, "2024-02-01"],
            "known_at": [
                "2023-12-29T08:00:00Z",
                "2023-12-29T08:00:00Z",
                "2024-01-15T08:00:00Z",
            ],
            "revision": pd.Series([1, 1, 2], dtype="int64"),
            "is_eligible": pd.Series([True, True, False], dtype="bool"),
        }
    )
