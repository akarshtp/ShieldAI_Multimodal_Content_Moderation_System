"""Entry point for running ShieldAI as a module: ``python -m shieldai``."""

from __future__ import annotations

import uvicorn

from shieldai.config import get_settings


def main() -> None:
    """Start the ShieldAI API server."""
    settings = get_settings()
    uvicorn.run(
        "shieldai.api.app:create_app",
        factory=True,
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
