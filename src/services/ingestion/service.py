"""
IngestionService — business logic for file and dataset ingestion.

Orchestrates:  validate → upload to MinIO → insert DB row → publish to Queue A.
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

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "webp"}
BUCKET_NAME = "invoices"
QUEUE_TOPIC = "queue:ingestion"  # Redis Stream name ("Queue A")


class IngestionService:
    """
    Encapsulates all ingestion logic.

    Dependencies are injected via constructor to keep business logic
    decoupled from infrastructure.
    """

    def __init__(
        self,
        blob_store: BlobStore,
        queue: MessageQueue,
    ) -> None:
        self._blob = blob_store
        self._queue = queue

    # ── Single file ingest ───────────────────────────────

    async def ingest_file(
        self,
        db: AsyncSession,
        file_bytes: bytes,
        filename: str,
        tenant_id: uuid.UUID,
        ground_truth: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """
        Ingest a single file:
        1. Validate extension
        2. Upload to MinIO
        3. Insert Job row in Postgres
        4. Publish message to Queue A
        5. Return job_id
        """
        # 1. Validate
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '.{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        # 2. Upload to MinIO
        job_id = uuid.uuid4()
        minio_path = f"{tenant_id}/{job_id}/input.{ext}"
        await self._blob.put(BUCKET_NAME, minio_path, file_bytes)

        # 3. Insert DB row
        job = Job(
            id=job_id,
            tenant_id=tenant_id,
            status="queued",
            input_file_path=f"{BUCKET_NAME}/{minio_path}",
            ground_truth=ground_truth,
        )
        db.add(job)
        await db.flush()  # Assign server defaults without committing

        # 4. Publish to Queue A
        await self._queue.publish(
            QUEUE_TOPIC,
            {
                "job_id": str(job_id),
                "tenant_id": str(tenant_id),
                "input_file_path": f"{BUCKET_NAME}/{minio_path}",
            },
        )

        logger.info("Job %s queued (tenant=%s, file=%s)", job_id, tenant_id, filename)
        return job_id

    # ── Dataset batch ingest ─────────────────────────────

    async def ingest_dataset(
        self,
        db: AsyncSession,
        split: str = "test",
        limit: int | None = None,
        tenant_id: uuid.UUID | None = None,
    ) -> list[uuid.UUID]:
        """
        Batch-ingest samples from the HuggingFace dataset.

        This runs synchronously within the async context because the
        dataset loader is CPU-bound (PIL image conversion).  For large
        batches the caller should wrap this in asyncio.create_task().
        """
        from src.services.ingestion.dataset_loader import load_hf_dataset

        # Resolve tenant
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
            # Use doc_id as a synthetic filename
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
            len(job_ids), split,
        )
        return job_ids
