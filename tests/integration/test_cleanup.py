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
from src.schemas.document import InvoiceExtraction
from src.services.output.worker import OutputWorker

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
async def test_output_worker_cleanup_flow(sample_image_bytes):
    from src.config.settings import get_settings
    settings = get_settings()
    
    blob_store = MinioBlobStore(settings.minio)
    queue = RedisMessageQueue(settings.redis)
    
    # 1. Create Tenant and Job in Postgres
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
            status="extracted",
            input_file_path=f"invoices/{tenant_id}/{job_id}/test_invoice_cleanup.png"
        )
        db.add(job)
        await db.commit()

    # Create unique topic name for isolated testing
    test_queue_extract = f"queue:extraction:test:{uuid.uuid4()}"

    # 2. Upload the original input file
    await blob_store.put("invoices", f"{tenant_id}/{job_id}/test_invoice_cleanup.png", sample_image_bytes)

    # 3. Pre-upload mock intermediate files (which should be deleted after done)
    await blob_store.put("invoices", f"{tenant_id}/{job_id}/ocr_output.json", b'{"mock": "ocr"}')
    await blob_store.put("invoices", f"{tenant_id}/{job_id}/extraction.json", b'{"mock": "extraction"}')
    await blob_store.put("invoices", f"{tenant_id}/{job_id}/final_output.json", b'{"mock": "final"}')

    # Verify all files exist before output worker processes
    assert await blob_store.exists("invoices", f"{tenant_id}/{job_id}/test_invoice_cleanup.png")
    assert await blob_store.exists("invoices", f"{tenant_id}/{job_id}/ocr_output.json")
    assert await blob_store.exists("invoices", f"{tenant_id}/{job_id}/extraction.json")
    assert await blob_store.exists("invoices", f"{tenant_id}/{job_id}/final_output.json")

    # 4. Publish message to unique extraction queue
    mock_extraction_data = {
        "header": {
            "invoice_no": "CLEAN-999",
            "invoice_date": "2026-06-08"
        },
        "items": [],
        "summary": {
            "total_net_worth": "150.00"
        }
    }
    
    await queue.publish(
        test_queue_extract,
        {
            "job_id": str(job_id),
            "tenant_id": str(tenant_id),
            "ocr_confidence": 0.95,
            "extraction_data": mock_extraction_data
        }
    )

    # Patch QUEUE_EXTRACTION module constant in output worker
    with patch("src.services.output.worker.QUEUE_EXTRACTION", test_queue_extract):
        worker = OutputWorker(
            blob_store=blob_store,
            queue=queue,
            worker_id=f"test_cleanup_worker_{uuid.uuid4().hex[:6]}"
        )
        
        # Start worker and wait briefly to process and run cleanup
        worker_task = asyncio.create_task(worker.start())
        await asyncio.sleep(2)
        await worker.stop()
        await worker_task

    # 5. Verify database job state is 'done'
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Job).where(Job.id == job_id))
        updated_job = res.scalars().first()
        assert updated_job is not None
        assert updated_job.status == "done"

    # 6. Verify intermediate JSON files are deleted from MinIO
    assert not await blob_store.exists("invoices", f"{tenant_id}/{job_id}/ocr_output.json")
    assert not await blob_store.exists("invoices", f"{tenant_id}/{job_id}/extraction.json")
    assert not await blob_store.exists("invoices", f"{tenant_id}/{job_id}/final_output.json")

    # 7. Verify the original file is NOT deleted (it must be kept!)
    assert await blob_store.exists("invoices", f"{tenant_id}/{job_id}/test_invoice_cleanup.png")

    # 8. Cleanup remaining input file and Redis queue
    await blob_store.delete("invoices", f"{tenant_id}/{job_id}/test_invoice_cleanup.png")
    await queue._redis.delete(test_queue_extract)
    await queue.close()
