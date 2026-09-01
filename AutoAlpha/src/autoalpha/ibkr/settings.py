from __future__ import annotations

import os
from dataclasses import dataclass, replace

DEFAULT_PAPER_PORT = 4002
DEFAULT_LIVE_PORT = 4001
PAPER_ACCOUNT_PREFIXES = ("DU", "DF")


class TradingModeError(RuntimeError):
    """An operation was attempted against the wrong trading mode."""


@dataclass(frozen=True)
class GatewaySettings:
    """Connection parameters for an Interactive Brokers gateway or TWS session."""

    host: str = "127.0.0.1"
    port: int = DEFAULT_PAPER_PORT
    client_id: int = 17
    account: str | None = None
    readonly: bool = True
    connect_timeout_seconds: float = 15.0
    require_paper_account: bool = True

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("Gateway host is required")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Gateway port is out of range: {self.port}")
        if self.client_id < 0:
            raise ValueError("Gateway client id must be non-negative")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("Connect timeout must be positive")

    @classmethod
    def from_environment(cls) -> GatewaySettings:
        """Read settings from IBKR_* environment variables, falling back to paper defaults."""
        readonly = _environment_flag("IBKR_READONLY", default=True)
        require_paper = _environment_flag("IBKR_REQUIRE_PAPER_ACCOUNT", default=True)
        account = os.environ.get("IBKR_ACCOUNT", "").strip() or None
        return cls(
            host=os.environ.get("IBKR_HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=_environment_int("IBKR_PORT", DEFAULT_PAPER_PORT),
            client_id=_environment_int("IBKR_CLIENT_ID", 17),
            account=account,
            readonly=readonly,
            connect_timeout_seconds=float(os.environ.get("IBKR_CONNECT_TIMEOUT", "15") or 15),
            require_paper_account=require_paper,
        )

    def for_client_id(self, client_id: int) -> GatewaySettings:
        return replace(self, client_id=client_id)

    def writable(self) -> GatewaySettings:
        """Return the same settings with the read-only guard lifted."""
        return replace(self, readonly=False)

    @property
    def is_paper_port(self) -> bool:
        return self.port == DEFAULT_PAPER_PORT

    def verify_account(self, account: str) -> None:
        """Reject a live account when the session is configured for paper trading only."""
        if not self.require_paper_account:
            return
        if not is_paper_account(account):
            raise TradingModeError(
                f"Account {account} is not a paper account; refusing to continue because "
                "require_paper_account is set. Set IBKR_REQUIRE_PAPER_ACCOUNT=0 to override."
            )


def is_paper_account(account: str) -> bool:
    """IBKR paper accounts carry a DU/DF prefix; live accounts use U/F."""
    return account.strip().upper().startswith(PAPER_ACCOUNT_PREFIXES)


def _environment_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


def _environment_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().casefold()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}
