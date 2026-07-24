from __future__ import annotations

import pandas as pd

from autoalpha.data.contracts import TableContract
from autoalpha.data.pit import PointInTimeFrame
from autoalpha.data.universe import SecurityUniverse


def test_as_of_query_does_not_see_later_revision(
    memberships: pd.DataFrame, membership_contract: TableContract
) -> None:
    table = PointInTimeFrame(memberships, membership_contract)
    early = table.as_of("2024-01-10T00:00:00Z")
    late = table.as_of("2024-01-20T00:00:00Z")

    assert set(early["revision"]) == {1}
    assert set(late["revision"]) == {1, 2}


def test_dynamic_universe_uses_only_information_known_at_the_time(
    memberships: pd.DataFrame, membership_contract: TableContract
) -> None:
    universe = SecurityUniverse(PointInTimeFrame(memberships, membership_contract))

    early = universe.as_of("2024-01-16", known_at="2024-01-10T00:00:00Z")
    revised = universe.as_of("2024-01-16", known_at="2024-01-20T00:00:00Z")

    assert early.tolist() == ["A", "B"]
    assert revised.tolist() == ["A"]
