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
from src.services.processing.ocr import OcrEngine, PaddleOcrEngine
from src.services.processing.llm.extractor import LlmExtractor, OllamaExtractor
from src.services.processing.worker import ProcessingWorker, QUEUE_INGESTION, QUEUE_EXTRACTION

@pytest_asyncio.fixture(autouse=True)
async def cleanup_database_pool():
    from src.database.database import engine
    await engine.dispose()
    yield
    await engine.dispose()

@pytest.fixture
def sample_image_bytes():
    # Helper to generate a 100x100 white PNG in memory
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@pytest.mark.asyncio
async def test_paddle_ocr_engine(sample_image_bytes):
    engine = PaddleOcrEngine()
    # Mock underlying paddleocr instance to avoid model initialization download in unit/integration test
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [
        [
            [[[10, 10], [100, 10], [100, 30], [10, 30]], ("Invoice No: 123", 0.98)]
        ]
    ]
    with patch.object(engine, "_get_ocr", return_value=mock_ocr):
        output = await engine.process(sample_image_bytes, "test.png")
        
    assert output.ocr_engine == "paddleocr"
    assert "Invoice No: 123" in output.raw_text
    assert len(output.text_blocks) == 1
    assert output.text_blocks[0].text == "Invoice No: 123"
    assert output.text_blocks[0].confidence == 0.98
    assert output.text_blocks[0].bbox == [10, 10, 100, 30]

@pytest.mark.asyncio
async def test_ollama_extractor_mocked():
    extractor = OllamaExtractor(host="http://localhost:11434", model_name="qwen2.5:7b")
    
    mock_chat_response = MagicMock()
    mock_chat_response.message.content = json.dumps({
        "header": {
            "invoice_no": "INV-456",
            "invoice_date": "2026-06-07",
            "seller": "Acme Corp",
            "client": "John Doe",
            "seller_tax_id": "TX-1",
            "client_tax_id": "TX-2",
            "iban": "US123"
        },
        "items": [
            {
                "item_desc": "Service Fee",
                "item_qty": "1",
                "item_net_price": "100",
                "item_net_worth": "100",
                "item_vat": "10%",
                "item_gross_worth": "110"
            }
        ],
        "summary": {
            "total_net_worth": "100",
            "total_vat": "10",
            "total_gross_worth": "110"
        }
    })
    
    mock_client = AsyncMock()
    mock_client.chat.return_value = mock_chat_response
    
    with patch.object(extractor, "_client", mock_client):
        ocr_out = OcrOutput(
            file_name="test.png",
            ocr_engine="paddleocr",
            raw_text="Invoice No: INV-456\nDate: 2026-06-07\nTotal: 110",
            average_confidence=0.98,
            text_blocks=[]
        )
        extraction = await extractor.extract(ocr_out)
        
    assert extraction.header.invoice_no == "INV-456"
    assert extraction.summary.total_gross_worth == "110"
    assert len(extraction.items) == 1
    assert extraction.items[0].item_desc == "Service Fee"

@pytest.mark.asyncio
async def test_processing_worker_flow(sample_image_bytes):
    # Setup real Redis and MinIO config from settings for integration testing
    from src.config.settings import get_settings
    settings = get_settings()
    
    blob_store = MinioBlobStore(settings.minio)
    queue = RedisMessageQueue(settings.redis)
    # Clear ingestion queue of any stale test messages
    await queue._redis.delete(QUEUE_INGESTION)
    
    # 1. Create a Tenant and a Job in DB
    async with AsyncSessionLocal() as db:
        # Check if test tenant exists
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
            input_file_path=f"invoices/{tenant_id}/{job_id}/test_invoice.png"
        )
        db.add(job)
        await db.commit()

    # 2. Upload sample image to MinIO
    await blob_store.put("invoices", f"{tenant_id}/{job_id}/test_invoice.png", sample_image_bytes)

    # 3. Publish to queue:ingestion
    await queue.publish(
        QUEUE_INGESTION,
        {
            "job_id": str(job_id),
            "tenant_id": str(tenant_id),
            "input_file_path": f"invoices/{tenant_id}/{job_id}/test_invoice.png"
        }
    )

    # 4. Initialize mocked OCR and LLM
    mock_ocr = MagicMock(spec=OcrEngine)
    mock_ocr.process = AsyncMock(return_value=OcrOutput(
        file_name="test_invoice.png",
        ocr_engine="mock",
        raw_text="Mocked Invoice Text",
        average_confidence=0.99,
        text_blocks=[TextBlock(text="Mocked Invoice Text", confidence=0.99)]
    ))

    mock_extractor = MagicMock(spec=LlmExtractor)
    mock_extractor.extract = AsyncMock(return_value=InvoiceExtraction(
        header={
            "invoice_no": "MOCK-123",
            "invoice_date": "2026-06-07"
        },
        items=[],
        summary={
            "total_net_worth": "100"
        }
    ))

    worker = ProcessingWorker(
        blob_store=blob_store,
        queue=queue,
        ocr_engine=mock_ocr,
        extractor=mock_extractor,
        worker_id="test_worker"
    )

    # 5. Start worker in the background and wait briefly for it to pick up the job
    worker_task = asyncio.create_task(worker.start())
    await asyncio.sleep(2)  # Give time to consume, process, and ACK
    await worker.stop()
    await worker_task
    
    # 6. Verify Database job state is done and populated
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Job).where(Job.id == job_id))
        updated_job = res.scalars().first()
        
        assert updated_job is not None
        assert updated_job.status == "extracted"

        assert updated_job.confidence_score == 0.99
        assert updated_job.ocr_output_path == f"invoices/{tenant_id}/{job_id}/ocr_output.json"
        assert updated_job.extraction_output_path == f"invoices/{tenant_id}/{job_id}/extraction.json"
        assert updated_job.extraction_data["header"]["invoice_no"] == "MOCK-123"

    # 7. Verify MinIO has output files
    assert await blob_store.exists("invoices", f"{tenant_id}/{job_id}/ocr_output.json")
    assert await blob_store.exists("invoices", f"{tenant_id}/{job_id}/extraction.json")

    # 8. Cleanup MinIO files
    await blob_store.delete("invoices", f"{tenant_id}/{job_id}/test_invoice.png")
    await blob_store.delete("invoices", f"{tenant_id}/{job_id}/ocr_output.json")
    await blob_store.delete("invoices", f"{tenant_id}/{job_id}/extraction.json")
    await queue.close()
