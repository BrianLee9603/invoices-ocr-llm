"""
Application entrypoint — run with ``python -m src.main``.
"""

import logging
import uvicorn

from src.config.settings import get_settings


def main() -> None:
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=(settings.app_env == "local"),
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
