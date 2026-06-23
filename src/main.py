"""
Application entrypoint — run with ``python -m src.main``.
"""

import logging
import uvicorn

from src.config.settings import get_settings
from src.utils.logging import setup_logging


def main() -> None:
    settings = get_settings()

    setup_logging(settings.log_level)

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=(settings.app_env == "local"),
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
