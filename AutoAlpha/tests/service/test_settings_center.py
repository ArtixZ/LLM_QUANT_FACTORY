from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from autoalpha.service.settings_center import (
    GlobalSettingsValues,
    default_settings,
    settings_catalog,
)


def test_global_settings_round_trip_structured_values(tmp_path: Path) -> None:
    defaults = default_settings(tmp_path / "AutoAlpha")
    defaults["data_path"] = str(tmp_path / "data")
    defaults["market_data_root"] = str(tmp_path / "market")
    defaults["data_product_ids"] = ["core_market", "execution_market"]
    defaults["full_llm_enabled"] = False
    defaults["proposal_batch_size"] = 2
    values = GlobalSettingsValues.model_validate(defaults)

    restored = GlobalSettingsValues.from_store(
        values.to_store(), project_root=tmp_path / "AutoAlpha"
    )

    assert restored == values
    assert restored.data_product_ids == ["core_market", "execution_market"]
    assert restored.full_llm_enabled is False
    assert restored.proposal_batch_size == 2
    assert values.to_store()["data_product_ids"] == '["core_market","execution_market"]'


def test_global_settings_reject_infeasible_combine_weights(tmp_path: Path) -> None:
    defaults = default_settings(tmp_path / "AutoAlpha")
    defaults.update(
        {
            "autocombine_default_min_factors": 2,
            "autocombine_default_max_factors": 5,
            "autocombine_default_maximum_weight": 0.40,
        }
    )

    with pytest.raises(ValidationError, match="maximum weights make the factor count infeasible"):
        GlobalSettingsValues.model_validate(defaults)


def test_settings_catalog_marks_secrets_without_exposing_values() -> None:
    catalog = settings_catalog()
    fields = {
        field["key"]: field for group in catalog for field in group["fields"]
    }

    assert fields["api_key"]["kind"] == "secret"
    assert fields["tushare_token"]["source"] == "系统 Keychain"
    assert fields["research_concurrency"]["effect"] == "重启服务"
    product_options = {
        option["value"]: option for option in fields["data_product_ids"]["options"]
    }
    assert product_options["core_market"]["disabled"]
    assert product_options["fundamentals"]["disabled"]
    assert not product_options["execution_market"]["disabled"]
    assert "value" not in fields["api_key"]
    assert "value" not in fields["tushare_token"]
