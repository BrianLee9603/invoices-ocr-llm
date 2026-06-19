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

import json
import logging
import uuid
from typing import Any, Dict

from src.database.database import AsyncSessionLocal
from src.database.models import Job
from src.database.blob_store import BlobStore
from src.database.queue import MessageQueue
from src.schemas.document import InvoiceExtraction
from src.services.output.validator import validate_extraction
from src.services.output.evaluator import evaluate_extraction
from src.services.base_worker import BaseWorker

logger = logging.getLogger(__name__)

QUEUE_EXTRACTION = "queue:extraction"
CONSUMER_GROUP = "output-group"
CONSUMER_NAME_PREFIX = "output-worker-"


class OutputWorker(BaseWorker):
    """
    Handles the final stage of the document processing pipeline:
    business validation, ground-truth evaluation, final MinIO output, and database completion.
    """

    def __init__(
        self,
        blob_store: BlobStore,
        queue: MessageQueue,
        worker_id: str = "1",
        reaper_interval: float = 30.0,
        reaper_idle_ms: int = 30000,
    ):
        super().__init__(
            blob_store=blob_store,
            queue=queue,
            topic=QUEUE_EXTRACTION,
            group_name=CONSUMER_GROUP,
            consumer_name_prefix=CONSUMER_NAME_PREFIX,
            worker_id=worker_id,
            reaper_interval=reaper_interval,
            reaper_idle_ms=reaper_idle_ms,
        )

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

        # Fetch current retry count from payload
        current_retry = int(payload.get("retry_count", 0))

        try:
            # 1. Fetch job details in a short-lived DB session
            async with AsyncSessionLocal() as db:
                job = await db.get(Job, job_id)
                if not job:
                    raise ValueError(f"Job {job_id} not found in database.")

                if job.status in ("done", "failed"):
                    logger.warning("[%s] Job is already in final status '%s'. Skipping.", job_id, job.status)
                    return

                ground_truth = job.ground_truth

            # Parse extraction data to InvoiceExtraction schema
            extraction = InvoiceExtraction.model_validate(extraction_data)

            # 2. Run Business Validations
            logger.debug("[%s] Running business validations...", job_id)
            valid = validate_extraction(extraction)
            if not valid:
                raise ValueError("Business validation failed: mandatory fields are missing or empty.")

            # 3. Evaluate against Ground Truth (if present)
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

            if ground_truth:
                logger.debug("[%s] Ground truth found. Evaluating extraction...", job_id)
                passed, accuracies = evaluate_extraction(extraction, ground_truth)
                evaluation_result = {
                    "evaluated": True,
                    "passed": passed,
                    "field_accuracies": accuracies
                }

            # 4. Construct Final Unified Output JSON
            final_output = {
                "job_id": str(job_id),
                "ocr_confidence": ocr_confidence,
                "extraction": extraction.model_dump(),
                "evaluation": evaluation_result
            }

            # 5. Upload final_output.json to MinIO
            final_key = f"{tenant_id}/{job_id}/final_output.json"
            final_json_str = json.dumps(final_output, indent=2)
            logger.debug("[%s] Uploading final_output.json to MinIO...", job_id)
            await self._blob_store.put("invoices", final_key, final_json_str.encode("utf-8"))

            # 6. Update Database Record to completion
            logger.info("[%s] Updating job record to completed status 'done'", job_id)
            await self._update_job_status(
                job_id,
                "done",
                evaluation_data=evaluation_result
            )

            # 7. Cleanup intermediate JSON files from MinIO to save storage (excluding final_output.json)
            logger.debug("[%s] Cleaning up intermediate JSON files from MinIO...", job_id)
            for file_key in [
                f"{tenant_id}/{job_id}/ocr_output.json",
                f"{tenant_id}/{job_id}/extraction.json"
            ]:
                try:
                    if await self._blob_store.exists("invoices", file_key):
                        await self._blob_store.delete("invoices", file_key)
                except Exception as clean_exc:
                    logger.warning("[%s] Failed to clean up intermediate file %s: %s", job_id, file_key, clean_exc)
        except Exception as exc:
            await self._handle_exception(
                exc=exc,
                job_id=job_id,
                current_retry=current_retry,
                transient_status="extracted",
                message_id=message_id,
            )
