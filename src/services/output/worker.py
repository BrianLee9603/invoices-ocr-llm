"""
OutputWorker — Background worker for business validation, evaluation, 
final JSON output generation, and database finalization.

Responsible for:
1. Consuming incoming job notifications from Redis queue:extraction (Queue B).
2. Performing business validations (mandatory fields check, VAT math).
3. Deterministically evaluating results against ground truth (for dataset runs).
4. Generating the unified final_output.json and uploading it to MinIO.
5. Updating the PostgreSQL job status to 'done' or 'failed'.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional

from src.database.database import AsyncSessionLocal
from src.database.models import Job
from src.database.blob_store import BlobStore
from src.database.queue import MessageQueue
from src.schemas.document import InvoiceExtraction
from src.services.output.validator import validate_extraction
from src.services.output.evaluator import evaluate_extraction

logger = logging.getLogger(__name__)

QUEUE_EXTRACTION = "queue:extraction"
CONSUMER_GROUP = "output-group"
CONSUMER_NAME_PREFIX = "output-worker-"


class OutputWorker:
    """
    Handles the final stage of the document processing pipeline:
    business validation, ground-truth evaluation, final MinIO output, and database completion.
    """

    def __init__(
        self,
        blob_store: BlobStore,
        queue: MessageQueue,
        worker_id: str = "1",
    ):
        self._blob_store = blob_store
        self._queue = queue
        self._consumer_name = f"{CONSUMER_NAME_PREFIX}{worker_id}"
        self._running_task: Optional[asyncio.Task] = None

    async def _update_job_status(self, job_id: uuid.UUID, status: str, **kwargs) -> None:
        """Update job status and additional fields in the database."""
        async with AsyncSessionLocal() as db:
            job = await db.get(Job, job_id)
            if not job:
                logger.error("Job %s not found in database.", job_id)
                return
            job.status = status
            for key, val in kwargs.items():
                setattr(job, key, val)
            await db.commit()

    async def handle_message(self, message_id: str, payload: Dict[str, Any]) -> None:
        """Processes a single extraction job message."""
        logger.info("Starting processing extraction message %s with payload: %s", message_id, payload)
        
        try:
            job_id_str = payload.get("job_id")
            tenant_id_str = payload.get("tenant_id")
            extraction_data = payload.get("extraction_data")
            ocr_confidence = payload.get("ocr_confidence", 0.0)

            if not job_id_str or not tenant_id_str or extraction_data is None:
                raise ValueError(f"Missing required fields in payload: {payload}")

            job_id = uuid.UUID(job_id_str)
            tenant_id = uuid.UUID(tenant_id_str)

            # Retrieve job from database to check current status (idempotency check)
            async with AsyncSessionLocal() as db:
                job = await db.get(Job, job_id)
                if not job:
                    raise ValueError(f"Job {job_id} not found in database.")
                
                if job.status in ("done", "failed"):
                    logger.warning("[%s] Job is already in final status '%s'. Skipping.", job_id, job.status)
                    return

                # Parse extraction data to InvoiceExtraction schema
                extraction = InvoiceExtraction.model_validate(extraction_data)

                # 1. Run Business Validations
                logger.debug("[%s] Running business validations...", job_id)
                valid = validate_extraction(extraction)
                if not valid:
                    raise ValueError("Business validation failed: mandatory fields are missing or empty.")

                # 2. Evaluate against Ground Truth (if present)
                evaluation_result = {
                    "evaluated": False,
                    "passed": False,
                    "field_accuracies": {
                        "invoice_no": 0.0,
                        "invoice_date": 0.0,
                        "total_net_worth": 0.0
                    }
                }

                if job.ground_truth:
                    logger.debug("[%s] Ground truth found. Evaluating extraction...", job_id)
                    passed, accuracies = evaluate_extraction(extraction, job.ground_truth)
                    evaluation_result = {
                        "evaluated": True,
                        "passed": passed,
                        "field_accuracies": accuracies
                    }

                # 3. Construct Final Unified Output JSON
                final_output = {
                    "job_id": str(job_id),
                    "ocr_confidence": ocr_confidence,
                    "extraction": extraction.model_dump(),
                    "evaluation": evaluation_result
                }

                # 4. Upload final_output.json to MinIO
                final_key = f"{tenant_id}/{job_id}/final_output.json"
                final_json_str = json.dumps(final_output, indent=2)
                logger.debug("[%s] Uploading final_output.json to MinIO...", job_id)
                await self._blob_store.put("invoices", final_key, final_json_str.encode("utf-8"))

                # 5. Update Database Record to completion
                logger.info("[%s] Updating job record to completed status 'done'", job_id)
                job.status = "done"
                job.evaluation_data = evaluation_result
                await db.commit()

        except Exception as exc:
            logger.exception("Error processing extraction message %s: %s", message_id, exc)
            if "job_id" in locals():
                try:
                    await self._update_job_status(
                        job_id,
                        "failed",
                        error_message=str(exc)
                    )
                except Exception as db_exc:
                    logger.exception("Failed to update status to failed in database for job %s: %s", job_id, db_exc)
            raise

    async def start(self) -> None:
        """Start the background consumer loop."""
        logger.info("Starting OutputWorker %s...", self._consumer_name)
        self._running_task = asyncio.create_task(
            self._queue.subscribe(
                topic=QUEUE_EXTRACTION,
                group_name=CONSUMER_GROUP,
                consumer_name=self._consumer_name,
                handler=self.handle_message
            )
        )
        try:
            await self._running_task
        except asyncio.CancelledError:
            logger.info("OutputWorker %s task cancelled.", self._consumer_name)
        except Exception as exc:
            logger.exception("Error in OutputWorker loop: %s", exc)

    async def stop(self) -> None:
        """Stop the background consumer loop."""
        if self._running_task:
            logger.info("Stopping OutputWorker %s...", self._consumer_name)
            self._running_task.cancel()
            try:
                await self._running_task
            except asyncio.CancelledError:
                pass
            logger.info("OutputWorker %s stopped.", self._consumer_name)
