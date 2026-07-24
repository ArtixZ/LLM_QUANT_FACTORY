from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autoalpha.data.contracts import TableContract
from autoalpha.data.snapshot import SnapshotStore


def test_snapshot_is_immutable_and_verifiable(
    tmp_path: Path,
    memberships: pd.DataFrame,
    membership_contract: TableContract,
) -> None:
    store = SnapshotStore(tmp_path / "snapshots")
    snapshot = store.write(
        "snapshot-001", [(memberships, membership_contract)], source="synthetic-test"
    )

    verified = store.verify(snapshot.snapshot_id)
    assert verified.tables[0].rows == 3
    with pytest.raises(FileExistsError):
        store.write("snapshot-001", [(memberships, membership_contract)], source="duplicate")

    table_path = tmp_path / "snapshots" / "snapshot-001" / "universe_membership.parquet"
    table_path.write_bytes(table_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="modified"):
        store.verify("snapshot-001")
