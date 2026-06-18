"""
SQLAlchemy ORM models mapped to the PostgreSQL tables from 001_schema.sql.

These models use the existing tables — schema creation is handled by
docker-entrypoint-initdb.d, not by SQLAlchemy's `create_all()`.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Enum,
    FetchedValue,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from src.database.database import Base

# Mirror the Postgres ENUM — values must match 001_schema.sql exactly
JOB_STATUS_VALUES = (
    "queued",
    "ocr_processing",
    "ocr_done",
    "extracting",
    "extracted",
    "validating",
    "done",
    "failed",
)

JobStatusEnum = Enum(
    *JOB_STATUS_VALUES,
    name="job_status",
    create_type=False,  # Postgres ENUM is created by SQL migration
)


# ──────────────────────────────────────────────────────────
#  Tenant
# ──────────────────────────────────────────────────────────

class Tenant(Base):
    """Multi-tenant owner of invoice processing jobs."""

    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    config: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, server_default="'{}'",
    )
    rate_limit: Mapped[int] = mapped_column(Integer, default=60)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(),
    )

    # Relationships
    jobs: Mapped[list["Job"]] = relationship(back_populates="tenant")

    def __repr__(self) -> str:
        return f"<Tenant {self.name} ({self.id})>"


# ──────────────────────────────────────────────────────────
#  Job
# ──────────────────────────────────────────────────────────

class Job(Base):
    """Single invoice/receipt processing job."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"),
    )
    status: Mapped[str] = mapped_column(
        JobStatusEnum, nullable=False, server_default="queued",
    )
    input_file_path: Mapped[str] = mapped_column(
        String(512), nullable=False,
    )
    ocr_output_path: Mapped[str | None] = mapped_column(String(512))
    extraction_output_path: Mapped[str | None] = mapped_column(String(512))
    confidence_score: Mapped[float | None] = mapped_column(Float)
    ocr_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    extraction_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    evaluation_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    ground_truth: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        server_onupdate=FetchedValue(),
    )

    # Relationships
    tenant: Mapped["Tenant"] = relationship(back_populates="jobs")

    @validates("status")
    def validate_status(self, key: str, value: str) -> str:
        """Enforce strict state machine transitions for Job status."""
        current_status = self.status
        
        # Self-transitions are always valid
        if current_status == value:
            return value

        # Define valid target states from each source state
        valid_transitions = {
            # None represents the initial state when the object is created
            None: {"queued", "ocr_processing", "ocr_done", "extracting", "extracted", "validating", "done", "failed"},
            "queued": {"ocr_processing", "failed"},
            "ocr_processing": {"ocr_done", "queued", "failed"},
            "ocr_done": {"extracting", "failed"},
            "extracting": {"extracted", "queued", "failed"},
            "extracted": {"validating", "done", "failed"},
            "validating": {"done", "failed"},
            # Terminal states (done/failed) cannot transition to any other status
            "done": set(),
            "failed": set(),
        }

        allowed_targets = valid_transitions.get(current_status, set())
        if value not in allowed_targets:
            raise ValueError(
                f"Invalid job status transition: '{current_status}' -> '{value}' "
                f"(Allowed target states: {sorted(list(allowed_targets))})"
            )
        return value

    def __repr__(self) -> str:
        return f"<Job {self.id} [{self.status}]>"
