"""
Pydantic schemas for Ingestion API request/response models.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────
#  Single File Ingest
# ──────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response returned after a single file is accepted for processing."""

    status: str = "accepted"
    message: str = "File queued for processing."
    job_id: uuid.UUID


# ──────────────────────────────────────────────────────────
#  Dataset Batch Ingest
# ──────────────────────────────────────────────────────────

class DatasetIngestRequest(BaseModel):
    """Request body for batch ingestion from HuggingFace dataset."""

    split: str = Field(
        default="test",
        description="Dataset split to ingest: 'train', 'test', or 'valid'.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of samples to ingest. None = entire split.",
    )


class DatasetIngestResponse(BaseModel):
    """Response returned when dataset ingestion is triggered."""

    status: str = "accepted"
    message: str
    total_jobs: int
    job_ids: list[uuid.UUID]


# ──────────────────────────────────────────────────────────
#  Job Status
# ──────────────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    """Full job status and results."""

    job_id: uuid.UUID
    status: str
    input_file_path: str
    confidence_score: float | None = None
    ground_truth: dict | None = None
    created_at: datetime
    updated_at: datetime
