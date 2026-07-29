from __future__ import annotations

from autoalpha.service.metric_convention import check_long_only_metric_convention


def test_metric_convention_flags_legacy_primary_sharpe_usage(tmp_path) -> None:
    source = tmp_path / "ranking.py"
    source.write_text(
        "leader = max(candidates, key=lambda item: item['sharpe_ratio'])\n",
        encoding="utf-8",
    )

    result = check_long_only_metric_convention(tmp_path)

    assert result["status"] == "WARN"
    assert result["issue_count"] == 1
    assert result["issues"][0]["path"] == "ranking.py"


def test_metric_convention_allows_long_only_primary_usage(tmp_path) -> None:
    source = tmp_path / "ranking.py"
    source.write_text(
        "leader = max(candidates, key=lambda item: item['long_only_sharpe_ratio'])\n",
        encoding="utf-8",
    )

    result = check_long_only_metric_convention(tmp_path)

    assert result["status"] == "PASS"
    assert result["issue_count"] == 0


def test_metric_convention_flags_multiline_legacy_primary_score(tmp_path) -> None:
    source = tmp_path / "batch_app.py"
    source.write_text(
        "\n".join(
            [
                "def _rank_results(results):",
                "    def score(item):",
                "        metrics = item.get('metrics') or {}",
                "        return (",
                "            int(item['status'] == 'SUCCESS'),",
                "            float(metrics.get('large_window_worst_sharpe', -100.0)),",
                "            float(metrics.get('sharpe_ratio', -100.0)),",
                "        )",
                "    return sorted(results, key=score, reverse=True)",
            ]
        ),
        encoding="utf-8",
    )

    result = check_long_only_metric_convention(tmp_path)

    assert result["status"] == "WARN"
    assert result["issue_count"] == 1
    assert result["issues"][0]["line"] == 7


def test_metric_convention_allows_explicit_long_only_then_legacy_fallback(tmp_path) -> None:
    source = tmp_path / "ranking.py"
    source.write_text(
        "\n".join(
            [
                "def score(item):",
                "    metrics = item.get('metrics') or {}",
                "    return (",
                "        float(metrics.get('long_only_sharpe_ratio', -100.0)),",
                "        float(metrics.get('sharpe_ratio', -100.0)),",
                "    )",
                "leader = max(candidates, key=score)",
            ]
        ),
        encoding="utf-8",
    )

    result = check_long_only_metric_convention(tmp_path)

    assert result["status"] == "PASS"
    assert result["issue_count"] == 0


def test_metric_convention_allows_metric_output_definitions(tmp_path) -> None:
    source = tmp_path / "capital.py"
    source.write_text(
        "\n".join(
            [
                "def _metrics(ledger):",
                "    return {",
                "        'sharpe_ratio': ledger.sharpe,",
                "        'simple_annual_return': ledger.annual_return,",
                "    }",
            ]
        ),
        encoding="utf-8",
    )

    result = check_long_only_metric_convention(tmp_path)

    assert result["status"] == "PASS"
    assert result["issue_count"] == 0
