from __future__ import annotations

import os

import uvicorn

os.environ.setdefault("AUTOALPHA_BATCH_MODE", "ASHARE_REALISTIC_LONG_ONLY")
os.environ.setdefault("AUTOALPHA_BATCH_PORT", "8790")

from autoalpha.service.batch_app import app  # noqa: E402, F401


def main() -> None:
    uvicorn.run(
        "autoalpha.service.realistic_batch_app:app",
        host="127.0.0.1",
        port=int(os.getenv("AUTOALPHA_BATCH_PORT", "8790")),
        reload=False,
    )


if __name__ == "__main__":
    main()
