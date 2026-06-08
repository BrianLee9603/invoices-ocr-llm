"""
ProcessingWorker — Background worker for document OCR and structured LLM extraction.

Responsible for:
1. Consuming incoming job notifications from Redis queue:ingestion (Queue A).
2. Downloading documents from MinIO.
3. Performing text OCR layout extraction via PaddleOCR or Docling.
4. Structuring extracted text into JSON using LLMs (Ollama or Gemini).
5. Saving results back to MinIO/PostgreSQL.
6. Publishing results to queue:extraction (Queue B) for validation/storage.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

from src.database.database import AsyncSessionLocal
from src.database.models import Job
from src.database.blob_store import BlobStore
from src.database.queue import MessageQueue
from src.services.processing.ocr import OcrEngine
from src.services.processing.extractor import LlmExtractor
from src.schemas.document import OcrOutput, InvoiceExtraction

# -- Logger & Constants --------------------------------------------------------
logger = logging.getLogger(__name__)

QUEUE_INGESTION = "queue:ingestion"
QUEUE_EXTRACTION = "queue:extraction"
CONSUMER_GROUP = "processing-group"
CONSUMER_NAME_PREFIX = "worker-"


class ProcessingWorker:
    """
    Handles the text parsing & layout extraction stage of the pipeline.

    Subscribes to Redis Stream, manages job state flow, and publishes structured outputs.
    """

    def __init__(
        self,
        blob_store: BlobStore,
        queue: MessageQueue,
        ocr_engine: OcrEngine,
        extractor: LlmExtractor,
        worker_id: str = "1",
    ):
        self._blob_store = blob_store
        self._queue = queue
        self._ocr_engine = ocr_engine
        self._extractor = extractor
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
        """Processes a single ingestion job message."""
        logger.info("Starting processing message %s with payload: %s", message_id, payload)
        
        try:
            job_id_str = payload.get("job_id")
            tenant_id_str = payload.get("tenant_id")
            input_file_path = payload.get("input_file_path")

            if not job_id_str or not tenant_id_str or not input_file_path:
                raise ValueError(f"Missing required fields in payload: {payload}")

            job_id = uuid.UUID(job_id_str)
            tenant_id = uuid.UUID(tenant_id_str)

            # Stage 1: Document Parsing (OCR)
            logger.info("[%s] Stage 1: Document Parsing (OCR)", job_id)
            await self._update_job_status(job_id, "ocr_processing")

            # Download file from MinIO
            parts = input_file_path.split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid input_file_path: {input_file_path}")
            bucket, key = parts[0], parts[1]

            logger.debug("[%s] Downloading file from MinIO: %s/%s", job_id, bucket, key)
            image_bytes = await self._blob_store.get(bucket, key)

            # Run OCR engine
            filename = key.split("/")[-1]
            logger.debug("[%s] Running OCR engine...", job_id)
            ocr_output: OcrOutput = await self._ocr_engine.process(image_bytes, filename)

            # Upload OCR output JSON to MinIO
            ocr_output_key = f"{tenant_id}/{job_id}/ocr_output.json"
            ocr_output_json = ocr_output.model_dump_json()
            logger.debug("[%s] Uploading ocr_output.json to MinIO...", job_id)
            await self._blob_store.put(bucket, ocr_output_key, ocr_output_json.encode("utf-8"))

            # Update DB with OCR results
            ocr_output_path = f"{bucket}/{ocr_output_key}"
            await self._update_job_status(
                job_id,
                "ocr_done",
                ocr_output_path=ocr_output_path,
                confidence_score=ocr_output.average_confidence,
                ocr_data=ocr_output.model_dump()
            )

            # Stage 2: LLM Structured Extraction
            logger.info("[%s] Stage 2: LLM Structured Extraction", job_id)
            await self._update_job_status(job_id, "extracting")

            # Run Ollama extraction
            extraction: InvoiceExtraction = await self._extractor.extract(ocr_output)

            # Upload extraction JSON to MinIO
            extraction_key = f"{tenant_id}/{job_id}/extraction.json"
            extraction_json = extraction.model_dump_json()
            logger.debug("[%s] Uploading extraction.json to MinIO...", job_id)
            await self._blob_store.put(bucket, extraction_key, extraction_json.encode("utf-8"))

            # Update DB with Extraction results
            extraction_output_path = f"{bucket}/{extraction_key}"
            await self._update_job_status(
                job_id,
                "extracted",
                extraction_output_path=extraction_output_path,
                extraction_data=extraction.model_dump()
            )

            # Stage 3: Queue B Publishing
            logger.info("[%s] Stage 3: Queue B Publishing", job_id)
            await self._queue.publish(
                QUEUE_EXTRACTION,
                {
                    "job_id": str(job_id),
                    "tenant_id": str(tenant_id),
                    "ocr_output_path": ocr_output_path,
                    "ocr_confidence": ocr_output.average_confidence,
                    "extraction_data": extraction.model_dump(),
                }
            )

            logger.info("[%s] Processing stage completed successfully.", job_id)


        except Exception as exc:
            logger.exception("Error processing message %s: %s", message_id, exc)
            # Mark job as failed in DB
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
        logger.info("Starting ProcessingWorker %s...", self._consumer_name)
        # Create subscribe task
        self._running_task = asyncio.create_task(
            self._queue.subscribe(
                topic=QUEUE_INGESTION,
                group_name=CONSUMER_GROUP,
                consumer_name=self._consumer_name,
                handler=self.handle_message
              )
        )
        try:
            await self._running_task
        except asyncio.CancelledError:
            logger.info("ProcessingWorker %s task cancelled.", self._consumer_name)
        except Exception as exc:
            logger.exception("Error in ProcessingWorker loop: %s", exc)

    async def stop(self) -> None:
        """Stop the background consumer loop."""
        if self._running_task:
            logger.info("Stopping ProcessingWorker %s...", self._consumer_name)
            self._running_task.cancel()
            try:
                await self._running_task
            except asyncio.CancelledError:
                pass
            logger.info("ProcessingWorker %s stopped.", self._consumer_name)
