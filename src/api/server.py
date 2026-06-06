"""
FastAPI application factory with lifespan-managed shared resources.

Resources initialized at startup:
- MinioBlobStore (object storage)
- RedisMessageQueue (message broker)
- IngestionService (wired with the above)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import get_settings
from src.database.blob_store import MinioBlobStore
from src.database.queue import RedisMessageQueue
from src.services.ingestion.service import IngestionService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Startup: create shared infra clients and wire services.
    Shutdown: close connections gracefully.
    """
    settings = get_settings()
    logger.info("Starting up (env=%s)...", settings.app_env)

    # ── Startup ──────────────────────────────────────────
    blob_store = MinioBlobStore(settings.minio)
    message_queue = RedisMessageQueue(settings.redis)

    ingestion_service = IngestionService(
        blob_store=blob_store,
        queue=message_queue,
    )

    # Attach to app.state so routers can access via request.app.state
    app.state.blob_store = blob_store
    app.state.message_queue = message_queue
    app.state.ingestion_service = ingestion_service

    logger.info("All services initialized.")

    yield

    # ── Shutdown ─────────────────────────────────────────
    await message_queue.close()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Invoice Extraction System",
        description="Intelligent Invoice & Receipt Extraction via OCR + LLM",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────
    from src.api.routers.ingest import router as ingest_router

    app.include_router(ingest_router)

    # ── Health check ─────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.app_env}

    return app


# Module-level app instance used by uvicorn
app = create_app()
