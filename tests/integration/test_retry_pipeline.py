import asyncio
import io
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from PIL import Image
from sqlalchemy import select

from src.database.database import AsyncSessionLocal
from src.database.models import Job, Tenant
from src.database.blob_store import MinioBlobStore
from src.database.queue import RedisMessageQueue
from src.schemas.document import OcrOutput, TextBlock, InvoiceExtraction
from src.services.processing.ocr.ocr import OcrEngine
from src.services.processing.llm.extractor import LlmExtractor
from src.exceptions import TransientError

@pytest_asyncio.fixture(autouse=True)
async def cleanup_database_pool():
    from src.database.database import engine
    await engine.dispose()
    yield
    await engine.dispose()

@pytest.fixture
def sample_image_bytes():
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@pytest.mark.asyncio
async def test_processing_worker_retry_flow(sample_image_bytes):
    from src.config.settings import get_settings
    settings = get_settings()
    
    blob_store = MinioBlobStore(settings.minio)
    queue = RedisMessageQueue(settings.redis)
    
    # 1. Create Tenant and Job
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Tenant).limit(1))
        tenant = result.scalars().first()
        if not tenant:
            tenant = Tenant(name="Test Tenant")
            db.add(tenant)
            await db.flush()
        
        tenant_id = tenant.id
        job_id = uuid.uuid4()
        
        job = Job(
            id=job_id,
            tenant_id=tenant_id,
            status="queued",
            input_file_path=f"invoices/{tenant_id}/{job_id}/test_invoice_retry.png",
            retry_count=0
        )
        db.add(job)
        await db.commit()

    # Create unique topic names for this test run to prevent queue pollution
    test_queue_ingest = f"queue:ingestion:test:{uuid.uuid4()}"
    test_queue_extract = f"queue:extraction:test:{uuid.uuid4()}"

    # 2. Upload file
    await blob_store.put("invoices", f"{tenant_id}/{job_id}/test_invoice_retry.png", sample_image_bytes)

    # 3. Publish to unique queue
    await queue.publish(
        test_queue_ingest,
        {
            "job_id": str(job_id),
            "tenant_id": str(tenant_id),
            "input_file_path": f"invoices/{tenant_id}/{job_id}/test_invoice_retry.png"
        }
    )

    # 4. Mock OCR (Success) and LLM Extractor (Throws TransientError)
    mock_ocr = MagicMock(spec=OcrEngine)
    mock_ocr.process = AsyncMock(return_value=OcrOutput(
        file_name="test_invoice_retry.png",
        ocr_engine="mock",
        raw_text="Mocked Text",
        average_confidence=0.99,
        text_blocks=[TextBlock(text="Mocked Text", confidence=0.99)]
    ))

    mock_extractor = MagicMock(spec=LlmExtractor)
    # Mock extract to throw a TransientError
    mock_extractor.extract = AsyncMock(side_effect=TransientError("Ollama is busy, try again later"))

    # Patch QUEUE_INGESTION and QUEUE_EXTRACTION module constants
    with patch("src.services.processing.worker.QUEUE_INGESTION", test_queue_ingest), \
         patch("src.services.processing.worker.QUEUE_EXTRACTION", test_queue_extract):
         
        from src.services.processing.worker import ProcessingWorker
        worker = ProcessingWorker(
            blob_store=blob_store,
            queue=queue,
            ocr_engine=mock_ocr,
            extractor=mock_extractor,
            worker_id=f"test_retry_worker_{uuid.uuid4().hex[:6]}"
        )

        # 5. Start worker and wait
        worker_task = asyncio.create_task(worker.start())
        await asyncio.sleep(2)  # Give time to consume and fail transiently
        await worker.stop()
        await worker_task
    
    # 6. Verify job state is set back to 'queued' and retry_count is 1
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Job).where(Job.id == job_id))
        updated_job = res.scalars().first()
        
        assert updated_job is not None
        assert updated_job.status == "queued"
        assert updated_job.retry_count == 1
        assert "Transient failure" in updated_job.error_message

    # 7. Cleanup Redis queues and MinIO files
    await blob_store.delete("invoices", f"{tenant_id}/{job_id}/test_invoice_retry.png")
    if await blob_store.exists("invoices", f"{tenant_id}/{job_id}/ocr_output.json"):
        await blob_store.delete("invoices", f"{tenant_id}/{job_id}/ocr_output.json")
        
    await queue._redis.delete(test_queue_ingest)
    await queue._redis.delete(test_queue_extract)
    await queue.close()
