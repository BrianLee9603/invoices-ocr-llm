"""
IngestionWorker — Core logic for document validation, upload, and queueing.

Responsible for:
1. Validating input files (supported extensions).
2. Uploading raw invoice/receipt files to MinIO.
3. Creating job entries in PostgreSQL database.
4. Enqueueing job messages to Queue A (Redis Streams).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.blob_store import BlobStore
from src.database.models import Job, Tenant
from src.database.queue import MessageQueue

# -- Logger & Constants --------------------------------------------------------
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "webp"}
BUCKET_NAME = "invoices"
QUEUE_TOPIC = "queue:ingestion"


class IngestionWorker:
    """
    Handles the ingestion stage of the document AI pipeline.

    Coordinates storage uploads, database entries, and message routing.
    """

    def __init__(
        self,
        blob_store: BlobStore,
        queue: MessageQueue,
    ) -> None:
        self._blob = blob_store
        self._queue = queue

    async def ingest_file(
        self,
        db: AsyncSession,
        file_bytes: bytes,
        filename: str,
        tenant_id: uuid.UUID,
        ground_truth: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """
        Ingests a single file into the system.

        1. Validates file extension.
        2. Uploads the file to MinIO.
        3. Saves job metadata in PostgreSQL.
        4. Publishes job to ingestion queue.
        """
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '.{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        job_id = uuid.uuid4()
        minio_path = f"{tenant_id}/{job_id}/input.{ext}"
        await self._blob.put(BUCKET_NAME, minio_path, file_bytes)

        job = Job(
            id=job_id,
            tenant_id=tenant_id,
            status="queued",
            input_file_path=f"{BUCKET_NAME}/{minio_path}",
            ground_truth=ground_truth,
        )
        db.add(job)
        await db.flush()

        await self._queue.publish(
            QUEUE_TOPIC,
            {
                "job_id": str(job_id),
                "tenant_id": str(tenant_id),
                "input_file_path": f"{BUCKET_NAME}/{minio_path}",
                "retry_count": 0,
            },
        )

        logger.info("Job %s queued (tenant=%s, file=%s)", job_id, tenant_id, filename)
        return job_id

    async def ingest_dataset(
        self,
        db: AsyncSession,
        split: str = "test",
        limit: int | None = None,
        tenant_id: uuid.UUID | None = None,
    ) -> list[uuid.UUID]:
        """
        Batch-ingests samples from the Hugging Face dataset.
        """
        from src.services.ingestion.dataset_loader import load_hf_dataset

        if tenant_id is None:
            result = await db.execute(
                select(Tenant).where(Tenant.name == "default"),
            )
            tenant = result.scalars().first()
            if tenant is None:
                raise RuntimeError("No default tenant found. Seed the database first.")
            tenant_id = tenant.id

        job_ids: list[uuid.UUID] = []

        for image_bytes, doc_id, parsed_data in load_hf_dataset(split, limit):
            filename = f"{doc_id}.png"
            job_id = await self.ingest_file(
                db=db,
                file_bytes=image_bytes,
                filename=filename,
                tenant_id=tenant_id,
                ground_truth=parsed_data,
            )
            job_ids.append(job_id)

        await db.commit()
        logger.info(
            "Dataset ingestion complete: %d jobs created (split=%s).",
            len(job_ids),
            split,
        )
        return job_ids
