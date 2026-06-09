"""
Ingestion API router — endpoints for file upload, dataset ingestion,
and job status checks.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.database import get_db
from src.database.models import Job, Tenant
from src.schemas.job import (
    DatasetIngestRequest,
    DatasetIngestResponse,
    IngestResponse,
    JobStatusResponse,
)
from src.services.ingestion.worker import IngestionWorker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


def _get_service(request: Request) -> IngestionWorker:
    """Retrieve the IngestionWorker from app state (set in lifespan)."""
    return request.app.state.ingestion_service


# ──────────────────────────────────────────────────────────
#  POST /ingest — Single file upload
# ──────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=IngestResponse,
    status_code=202,
    summary="Upload a single invoice/receipt for processing",
)
async def ingest_file(
    file: UploadFile,
    service: IngestionWorker = Depends(_get_service),
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """
    Accept a single file (multipart/form-data) and queue it for OCR + extraction.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Use default tenant for now
    result = await db.execute(select(Tenant).where(Tenant.name == "default"))
    tenant = result.scalars().first()
    if tenant is None:
        raise HTTPException(status_code=500, detail="Default tenant not found.")

    try:
        job_id = await service.ingest_file(
            db=db,
            file_bytes=file_bytes,
            filename=file.filename,
            tenant_id=tenant.id,
        )
        await db.commit()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return IngestResponse(job_id=job_id)


# ──────────────────────────────────────────────────────────
#  POST /ingest/dataset — Batch from HuggingFace
# ──────────────────────────────────────────────────────────

@router.post(
    "/dataset",
    response_model=DatasetIngestResponse,
    status_code=202,
    summary="Trigger batch ingestion from HuggingFace dataset",
)
async def ingest_dataset(
    body: DatasetIngestRequest,
    service: IngestionWorker = Depends(_get_service),
    db: AsyncSession = Depends(get_db),
) -> DatasetIngestResponse:
    """
    Fire-and-forget background task for large dataset ingestion.

    For small batches (limit ≤ 5), runs inline and returns job IDs.
    For larger batches, spawns a background asyncio task.
    """
    INLINE_THRESHOLD = 5
    effective_limit = body.limit

    if effective_limit is not None and effective_limit <= INLINE_THRESHOLD:
        # Small batch — run inline
        job_ids = await service.ingest_dataset(
            db=db, split=body.split, limit=effective_limit,
        )
        return DatasetIngestResponse(
            message=f"Ingested {len(job_ids)} samples from '{body.split}' split.",
            total_jobs=len(job_ids),
            job_ids=job_ids,
        )

    # Large batch — background task
    # We need a fresh DB session for the background task since the
    # request session will be closed when the response is sent.
    from src.database.database import AsyncSessionLocal

    async def _background_ingest() -> None:
        async with AsyncSessionLocal() as bg_db:
            try:
                ids = await service.ingest_dataset(
                    db=bg_db, split=body.split, limit=effective_limit,
                )
                logger.info(
                    "Background dataset ingestion complete: %d jobs.", len(ids),
                )
            except Exception:
                logger.exception("Background dataset ingestion failed.")

    asyncio.create_task(_background_ingest())

    desc = f"limit={effective_limit}" if effective_limit else "all"
    return DatasetIngestResponse(
        message=(
            f"Dataset ingestion started in background "
            f"(split='{body.split}', {desc}). "
            f"Check individual job statuses via GET /jobs/{{job_id}}."
        ),
        total_jobs=0,  # Unknown until background task completes
        job_ids=[],
    )


# ──────────────────────────────────────────────────────────
#  GET /jobs/{job_id} — Job status
# ──────────────────────────────────────────────────────────

@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Check the status of a processing job",
)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    """Return the current status and results of a job."""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalars().first()

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        input_file_path=job.input_file_path,
        confidence_score=job.confidence_score,
        ocr_data=job.ocr_data,
        extraction_data=job.extraction_data,
        evaluation_data=job.evaluation_data,
        ground_truth=job.ground_truth,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
