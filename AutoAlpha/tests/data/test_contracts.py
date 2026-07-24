from __future__ import annotations

import pandas as pd
import pytest

from autoalpha.data.contracts import DataContractError, TableContract


def test_contract_detects_duplicate_point_in_time_keys(
    memberships: pd.DataFrame, membership_contract: TableContract
) -> None:
    duplicated = pd.concat([memberships, memberships.iloc[[0]]], ignore_index=True)

    report = membership_contract.validate(duplicated)

    assert report.passed is False
    with pytest.raises(DataContractError, match="DUPLICATE_PRIMARY_KEY"):
        report.raise_for_errors()


def test_contract_fingerprint_changes_with_semantics(membership_contract: TableContract) -> None:
    changed = TableContract(
        name=membership_contract.name,
        version="2.0.0",
        fields=membership_contract.fields,
        primary_key=membership_contract.primary_key,
        event_time=membership_contract.event_time,
        knowledge_time=membership_contract.knowledge_time,
        entity_key=membership_contract.entity_key,
    )

    assert membership_contract.fingerprint != changed.fingerprint
