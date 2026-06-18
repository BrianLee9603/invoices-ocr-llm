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

import logging
import uuid
from typing import Any, Dict

from src.database.database import AsyncSessionLocal
from src.database.models import Job
from src.database.blob_store import BlobStore
from src.database.queue import MessageQueue
from src.services.processing.ocr.ocr import OcrEngine
from src.services.processing.llm.extractor import LlmExtractor
from src.schemas.document import OcrOutput, InvoiceExtraction
from src.services.base_worker import BaseWorker

# -- Logger & Constants --------------------------------------------------------
logger = logging.getLogger(__name__)

QUEUE_INGESTION = "queue:ingestion"
QUEUE_EXTRACTION = "queue:extraction"
CONSUMER_GROUP = "processing-group"
CONSUMER_NAME_PREFIX = "worker-"


class ProcessingWorker(BaseWorker):
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
        reaper_interval: float = 30.0,
        reaper_idle_ms: int = 30000,
    ):
        super().__init__(
            blob_store=blob_store,
            queue=queue,
            topic=QUEUE_INGESTION,
            group_name=CONSUMER_GROUP,
            consumer_name_prefix=CONSUMER_NAME_PREFIX,
            worker_id=worker_id,
            reaper_interval=reaper_interval,
            reaper_idle_ms=reaper_idle_ms,
        )
        self._ocr_engine = ocr_engine
        self._extractor = extractor


    async def handle_message(self, message_id: str, payload: Dict[str, Any]) -> None:
        """Processes a single ingestion job message."""
        logger.info("Starting processing message %s with payload: %s", message_id, payload)
        
        # Parse payload first
        try:
            job_id_str = payload.get("job_id")
            tenant_id_str = payload.get("tenant_id")
            input_file_path = payload.get("input_file_path")

            if not job_id_str or not tenant_id_str or not input_file_path:
                raise ValueError(f"Missing required fields in payload: {payload}")

            job_id = uuid.UUID(job_id_str)
            tenant_id = uuid.UUID(tenant_id_str)
        except Exception as payload_exc:
            logger.error("Failed to parse queue payload: %s", payload_exc)
            return

        # Fetch current retry count from payload
        current_retry = int(payload.get("retry_count", 0))

        try:
            async with AsyncSessionLocal() as db:
                # Stage 1: Document Parsing (OCR)
                logger.info("[%s] Stage 1: Document Parsing (OCR)", job_id)
                await self._update_job_status(job_id, "ocr_processing", db=db)

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
                    db=db,
                    ocr_output_path=ocr_output_path,
                    confidence_score=ocr_output.average_confidence,
                    ocr_data=ocr_output.model_dump()
                )

                # Stage 2: LLM Structured Extraction
                logger.info("[%s] Stage 2: LLM Structured Extraction", job_id)
                await self._update_job_status(job_id, "extracting", db=db)

                # Run LLM extraction
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
                    db=db,
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
            await self._handle_exception(
                exc=exc,
                job_id=job_id,
                current_retry=current_retry,
                transient_status="queued",
                message_id=message_id,
            )
