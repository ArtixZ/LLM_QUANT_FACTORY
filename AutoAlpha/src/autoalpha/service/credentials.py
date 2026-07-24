from __future__ import annotations

from dataclasses import dataclass

import keyring
from keyring.errors import KeyringError


@dataclass(frozen=True)
class SystemCredentialStore:
    """Store service secrets in the operating system credential vault."""

    service_name: str = "com.autoalpha.openai-compatible"
    account_name: str = "research-service"

    def get(self) -> str | None:
        try:
            return keyring.get_password(self.service_name, self.account_name)
        except KeyringError as error:
            raise RuntimeError(f"System credential store is unavailable: {error}") from error

    def set(self, value: str) -> None:
        secret = value.strip()
        if not secret:
            raise ValueError("API key cannot be empty")
        try:
            keyring.set_password(self.service_name, self.account_name, secret)
        except KeyringError as error:
            raise RuntimeError(f"Could not persist API key: {error}") from error
