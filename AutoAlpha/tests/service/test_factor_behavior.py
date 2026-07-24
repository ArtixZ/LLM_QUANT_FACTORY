from __future__ import annotations

import numpy as np

from autoalpha.service.factor_behavior import (
    _align_signal_sketches,
    _hierarchical_labels,
    _redundancy_label,
    _row_correlation,
)


def test_row_correlation_is_finite_for_constant_rows() -> None:
    result = _row_correlation(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]))

    assert np.isfinite(result).all()
    assert np.diag(result).tolist() == [1.0, 1.0]


def test_hierarchical_labels_group_similar_factors() -> None:
    similarity = np.array(
        [
            [1.0, 0.90, 0.10],
            [0.90, 1.0, 0.15],
            [0.10, 0.15, 1.0],
        ]
    )

    labels = _hierarchical_labels(similarity, 0.74)

    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_redundancy_labels_distinguish_duplicates_and_singletons() -> None:
    assert _redundancy_label(0.96, 0.97, 0.93, 2) == "NEAR_DUPLICATE"
    assert _redundancy_label(0.86, 0.90, 0.78, 2) == "SUBSTITUTE"
    assert _redundancy_label(0.20, 0.20, 0.20, 1) == "DISTINCT"


def test_signal_sketches_align_to_union_calendar() -> None:
    first = np.array([[1.0, 2.0], [3.0, 4.0]])
    second = np.array([[5.0, 6.0]])
    dates = [
        np.array(["2024-01-08", "2024-01-15"], dtype="datetime64[D]"),
        np.array(["2024-01-15"], dtype="datetime64[D]"),
    ]

    result = _align_signal_sketches([first, second], dates)

    assert result.shape == (2, 4)
    assert result[1].tolist() == [0.0, 0.0, 5.0, 6.0]
