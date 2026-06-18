"""
BaseWorker — Abstract base class for background service workers.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.database.database import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.models import Job
from src.database.blob_store import BlobStore
from src.database.queue import MessageQueue

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """
    Abstract base worker implementing common database operations,
    queue subscription lifecycle, and PEL reaper loops.
    """

    def __init__(
        self,
        blob_store: BlobStore,
        queue: MessageQueue,
        topic: str,
        group_name: str,
        consumer_name_prefix: str,
        worker_id: str = "1",
        reaper_interval: float = 30.0,
        reaper_idle_ms: int = 30000,
    ) -> None:
        self._blob_store = blob_store
        self._queue = queue
        self._topic = topic
        self._group_name = group_name
        self._consumer_name = f"{consumer_name_prefix}{worker_id}"
        self._running_task: Optional[asyncio.Task] = None
        self._reaper_task: Optional[asyncio.Task] = None
        self._reaper_interval = reaper_interval
        self._reaper_idle_ms = reaper_idle_ms

    async def _update_job_status(
        self,
        job_id: uuid.UUID,
        status: str,
        db: Optional[AsyncSession] = None,
        **kwargs
    ) -> None:
        """Update job status and additional fields in the database."""
        if db is not None:
            job = await db.get(Job, job_id)
            if not job:
                logger.error("Job %s not found in database.", job_id)
                return
            job.status = status
            for key, val in kwargs.items():
                setattr(job, key, val)
            await db.commit()
        else:
            async with AsyncSessionLocal() as new_db:
                job = await new_db.get(Job, job_id)
                if not job:
                    logger.error("Job %s not found in database.", job_id)
                    return
                job.status = status
                for key, val in kwargs.items():
                    setattr(job, key, val)
                await new_db.commit()

    @abstractmethod
    async def handle_message(self, message_id: str, payload: Dict[str, Any]) -> None:
        """Process a single message. Must be implemented by concrete classes."""
        ...

    async def _reaper_loop(self) -> None:
        """Background loop that periodically reclaims stale pending messages."""
        logger.info(
            "Starting PEL reaper loop (interval=%.1fs, idle=%dms, topic=%s)...",
            self._reaper_interval, self._reaper_idle_ms, self._topic
        )
        while True:
            try:
                await asyncio.sleep(self._reaper_interval)
                reclaimed = await self._queue.reclaim_pending(
                    topic=self._topic,
                    group_name=self._group_name,
                    consumer_name=self._consumer_name,
                    handler=self.handle_message,
                    idle_ms=self._reaper_idle_ms,
                )
                if reclaimed:
                    logger.info("Reclaimed %d stale messages from %s.", reclaimed, self._topic)
            except asyncio.CancelledError:
                logger.info("PEL reaper loop cancelled for %s.", self._topic)
                break
            except Exception:
                logger.exception("Error in PEL reaper loop for %s.", self._topic)

    async def _handle_exception(
        self,
        exc: Exception,
        job_id: uuid.UUID,
        current_retry: int,
        transient_status: str,
        message_id: str,
    ) -> None:
        """
        Handle exception during message processing.
        Classifies the error, updates status and retry count, and re-raises transient errors.
        """
        logger.exception("Error processing message %s: %s", message_id, exc)

        from src.exceptions import PersistentError

        # Check if this error is persistent/non-retryable
        is_persistent = isinstance(exc, PersistentError) or isinstance(exc, (ValueError, KeyError, TypeError))
        MAX_RETRIES = 3

        if not is_persistent and current_retry < MAX_RETRIES:
            next_retry = current_retry + 1
            logger.warning("[%s] Transient error. Leaving in PEL for retry #%d. Error: %s", job_id, next_retry, exc)
            try:
                await self._update_job_status(
                    job_id,
                    transient_status,
                    retry_count=next_retry,
                    error_message=f"Transient failure (Retry #{next_retry}): {exc}"
                )
            except Exception as db_exc:
                logger.exception("Failed to update status to %s for job %s: %s", transient_status, job_id, db_exc)
            # Re-raise to keep message in PEL (no ACK)
            raise
        else:
            # Permanent failure or max retries exceeded
            logger.error("[%s] Permanent failure or max retries exceeded (retry #%d). Marking job as failed. Error: %s", job_id, current_retry, exc)
            try:
                await self._update_job_status(
                    job_id,
                    "failed",
                    error_message=f"Failed: {exc}"
                )
            except Exception as db_exc:
                logger.exception("Failed to update status to failed in database for job %s: %s", job_id, db_exc)

    async def start(self) -> None:
        """Start the background consumer loop and reaper task."""
        logger.info("Starting worker %s on %s...", self._consumer_name, self._topic)
        self._running_task = asyncio.create_task(
            self._queue.subscribe(
                topic=self._topic,
                group_name=self._group_name,
                consumer_name=self._consumer_name,
                handler=self.handle_message
            )
        )
        self._reaper_task = asyncio.create_task(self._reaper_loop())
        try:
            await asyncio.gather(self._running_task, self._reaper_task)
        except asyncio.CancelledError:
            logger.info("Worker %s task/reaper cancelled.", self._consumer_name)
        except Exception as exc:
            logger.exception("Error in worker %s loop: %s", self._consumer_name, exc)

    async def stop(self) -> None:
        """Stop the background consumer loop and reaper task."""
        logger.info("Stopping worker %s...", self._consumer_name)
        if self._running_task:
            self._running_task.cancel()
        if self._reaper_task:
            self._reaper_task.cancel()
        if self._running_task or self._reaper_task:
            try:
                await asyncio.gather(
                    self._running_task or asyncio.sleep(0),
                    self._reaper_task or asyncio.sleep(0),
                    return_exceptions=True
                )
            except Exception:
                pass
        logger.info("Worker %s stopped.", self._consumer_name)
