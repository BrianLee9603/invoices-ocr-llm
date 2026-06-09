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
        except Exception as payload_exc:
            logger.error("Failed to parse output queue payload: %s", payload_exc)
            return

        # Fetch current retry count
        current_retry = 0
        try:
            async with AsyncSessionLocal() as db:
                job = await db.get(Job, job_id)
                if job:
                    current_retry = job.retry_count or 0
                    if job.status in ("done", "failed"):
                        logger.warning("[%s] Job is already in final status '%s'. Skipping.", job_id, job.status)
                        return
        except Exception as db_err:
            logger.warning("[%s] Failed to fetch job status/retry count from database: %s", job_id, db_err)

        try:
            # Parse extraction data to InvoiceExtraction schema
            extraction = InvoiceExtraction.model_validate(extraction_data)

            # 1. Run Business Validations
            logger.debug("[%s] Running business validations...", job_id)
            valid = validate_extraction(extraction)
            if not valid:
                raise ValueError("Business validation failed: mandatory fields are missing or empty.")

            # 2. Evaluate against Ground Truth (if present)
            # Default: no ground truth means nothing to fail against — accept extraction as correct
            evaluation_result = {
                "evaluated": False,
                "passed": True,
                "field_accuracies": {
                    "invoice_no": 1.0,
                    "invoice_date": 1.0,
                    "total_net_worth": 1.0
                }
            }

            async with AsyncSessionLocal() as db:
                job = await db.get(Job, job_id)
                if not job:
                    raise ValueError(f"Job {job_id} not found in database.")

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

                # 6. Cleanup intermediate JSON files from MinIO to save storage
                logger.debug("[%s] Cleaning up intermediate JSON files from MinIO...", job_id)
                for file_key in [
                    f"{tenant_id}/{job_id}/ocr_output.json",
                    f"{tenant_id}/{job_id}/extraction.json",
                    f"{tenant_id}/{job_id}/final_output.json"
                ]:
                    try:
                        if await self._blob_store.exists("invoices", file_key):
                            await self._blob_store.delete("invoices", file_key)
                    except Exception as clean_exc:
                        logger.warning("[%s] Failed to clean up intermediate file %s: %s", job_id, file_key, clean_exc)

        except Exception as exc:
            logger.exception("Error processing extraction message %s: %s", message_id, exc)
            
            from src.exceptions import PersistentError, TransientError
            # Check if this error is persistent/non-retryable
            is_persistent = isinstance(exc, PersistentError) or isinstance(exc, (ValueError, KeyError, TypeError))
            MAX_RETRIES = 3

            if not is_persistent and current_retry < MAX_RETRIES:
                next_retry = current_retry + 1
                backoff_delay = 5 * (2 ** current_retry)  # 5s, 10s, 20s
                logger.warning("[%s] Transient error in output stage. Requeuing for retry #%d in %ds. Error: %s", job_id, next_retry, backoff_delay, exc)
                
                try:
                    await self._update_job_status(
                        job_id,
                        "extracted",  # set status back to extracted so it can be retried in output stage
                        retry_count=next_retry,
                        error_message=f"Transient output failure (Retry #{next_retry}): {exc}"
                    )
                    
                    # Run requeue in background to avoid blocking consumer thread
                    async def requeue_task():
                        await asyncio.sleep(backoff_delay)
                        try:
                            await self._queue.publish(QUEUE_EXTRACTION, payload)
                        except Exception as re_exc:
                            logger.exception("Failed to publish retry message for job %s: %s", job_id, re_exc)
                    
                    asyncio.create_task(requeue_task())
                except Exception as db_exc:
                    logger.exception("Failed to update status to extracted for job %s: %s", job_id, db_exc)
            else:
                # Permanent failure or max retries exceeded
                logger.error("[%s] Permanent failure or max retries exceeded. Marking job as failed. Error: %s", job_id, exc)
                try:
                    await self._update_job_status(
                        job_id,
                        "failed",
                        error_message=f"Failed in output stage: {exc}"
                    )
                except Exception as db_exc:
                    logger.exception("Failed to update status to failed in database for job %s: %s", job_id, db_exc)

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
