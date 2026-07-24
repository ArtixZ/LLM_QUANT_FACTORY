from __future__ import annotations

from dataclasses import dataclass

from autoalpha.service.worker import SecretVault


@dataclass
class FakeCredentialStore:
    value: str | None = None

    def get(self) -> str | None:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


def test_secret_vault_persists_and_resolves_credential(monkeypatch) -> None:
    monkeypatch.delenv("AUTOALPHA_API_KEY", raising=False)
    credentials = FakeCredentialStore()
    vault = SecretVault(credential_store=credentials)

    vault.set("deepseek-secret")

    assert vault.configured()
    assert vault.get() == "deepseek-secret"
    assert vault.api_key is None


def test_environment_credential_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("AUTOALPHA_API_KEY", "environment-secret")
    credentials = FakeCredentialStore("stored-secret")
    vault = SecretVault(credential_store=credentials)

    assert vault.get() == "environment-secret"
