from __future__ import annotations

import pytest

from autoalpha.ibkr.settings import (
    DEFAULT_PAPER_PORT,
    GatewaySettings,
    TradingModeError,
    is_paper_account,
)


@pytest.mark.parametrize(
    ("account", "expected"),
    [("DU1234567", True), ("DF12345", True), ("U1234567", False), ("F999", False)],
)
def test_paper_account_detection(account: str, expected: bool) -> None:
    assert is_paper_account(account) is expected


def test_defaults_target_the_paper_gateway_read_only() -> None:
    settings = GatewaySettings()
    assert settings.port == DEFAULT_PAPER_PORT
    assert settings.is_paper_port is True
    assert settings.readonly is True


def test_writable_lifts_only_the_readonly_flag() -> None:
    settings = GatewaySettings(client_id=5)
    writable = settings.writable()
    assert writable.readonly is False
    assert writable.client_id == 5
    assert settings.readonly is True


def test_verify_account_rejects_live_account_by_default() -> None:
    with pytest.raises(TradingModeError, match="not a paper account"):
        GatewaySettings().verify_account("U7654321")


def test_verify_account_allows_live_when_guard_is_disabled() -> None:
    GatewaySettings(require_paper_account=False).verify_account("U7654321")


def test_verify_account_accepts_paper_account() -> None:
    GatewaySettings().verify_account("DU1234567")


@pytest.mark.parametrize(
    ("field", "value"),
    [("host", "  "), ("port", 0), ("port", 70000), ("client_id", -1),
     ("connect_timeout_seconds", 0)],
)
def test_invalid_settings_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        GatewaySettings(**{field: value})


def test_from_environment_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IBKR_HOST", "10.0.0.5")
    monkeypatch.setenv("IBKR_PORT", "4001")
    monkeypatch.setenv("IBKR_CLIENT_ID", "42")
    monkeypatch.setenv("IBKR_ACCOUNT", "DU1234567")
    monkeypatch.setenv("IBKR_READONLY", "0")
    settings = GatewaySettings.from_environment()
    assert (settings.host, settings.port, settings.client_id) == ("10.0.0.5", 4001, 42)
    assert settings.account == "DU1234567"
    assert settings.readonly is False


def test_from_environment_defaults_to_readonly_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("IBKR_HOST", "IBKR_PORT", "IBKR_CLIENT_ID", "IBKR_ACCOUNT", "IBKR_READONLY"):
        monkeypatch.delenv(name, raising=False)
    settings = GatewaySettings.from_environment()
    assert settings.port == DEFAULT_PAPER_PORT
    assert settings.readonly is True
    assert settings.account is None


def test_from_environment_rejects_non_integer_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IBKR_PORT", "not-a-port")
    with pytest.raises(ValueError, match="IBKR_PORT"):
        GatewaySettings.from_environment()
