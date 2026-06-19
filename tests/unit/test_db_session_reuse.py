import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import uuid

from src.services.processing.worker import ProcessingWorker
from src.services.output.worker import OutputWorker
from src.database.models import Job
from src.schemas.document import OcrOutput, InvoiceExtraction

@pytest.mark.asyncio
async def test_processing_worker_db_session_scoping():
    mock_blob = AsyncMock()
    mock_queue = AsyncMock()
    mock_ocr = AsyncMock()
    mock_extractor = AsyncMock()
    
    # Setup mock returns
    mock_blob.get.return_value = b"image_bytes"
    mock_ocr.process.return_value = OcrOutput(
        file_name="input.pdf",
        ocr_engine="dummy",
        raw_text="Invoice #123",
        average_confidence=0.9,
        text_blocks=[]
    )
    mock_extractor.extract.return_value = InvoiceExtraction(
        header={"invoice_no": "123", "invoice_date": "2026-06-18"},
        items=[],
        summary={"total_net_worth": "100.0"}
    )
    
    worker = ProcessingWorker(
        blob_store=mock_blob,
        queue=mock_queue,
        ocr_engine=mock_ocr,
        extractor=mock_extractor,
        worker_id="test"
    )
    
    job_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    payload = {
        "job_id": str(job_id),
        "tenant_id": str(tenant_id),
        "input_file_path": "invoices/tenant/job/input.pdf",
        "retry_count": 0,
    }
    
    # Mock database session
    job = Job(id=job_id, tenant_id=tenant_id, status="queued")
    mock_session = AsyncMock()
    mock_session.get.return_value = job
    mock_session_cls = MagicMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    
    with patch("src.services.processing.worker.AsyncSessionLocal", mock_session_cls), \
         patch("src.services.base_worker.AsyncSessionLocal", mock_session_cls):
        await worker.handle_message("msg-123", payload)
        
    # Verify the database session was initialized for each status update (4 updates total)
    assert mock_session_cls.call_count == 4
    # Verify we committed each status update
    assert mock_session.commit.call_count == 4
    # Check status progression in the same job instance
    assert job.status == "extracted"


@pytest.mark.asyncio
async def test_output_worker_db_session_scoping():
    mock_blob = AsyncMock()
    mock_queue = AsyncMock()
    
    worker = OutputWorker(
        blob_store=mock_blob,
        queue=mock_queue,
        worker_id="test"
    )
    
    job_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    payload = {
        "job_id": str(job_id),
        "tenant_id": str(tenant_id),
        "ocr_confidence": 0.9,
        "extraction_data": {"header": {"invoice_no": "123", "invoice_date": "2026-06-18"}, "items": [], "summary": {"total_net_worth": "10.0"}},
        "retry_count": 0,
    }
    
    job = Job(id=job_id, tenant_id=tenant_id, status="extracted")
    mock_session = AsyncMock()
    mock_session.get.return_value = job
    mock_session_cls = MagicMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    
    with patch("src.services.output.worker.AsyncSessionLocal", mock_session_cls), \
         patch("src.services.base_worker.AsyncSessionLocal", mock_session_cls):
        await worker.handle_message("msg-123", payload)
        
    # Verify the database session was initialized twice (once for initial fetch, once for final update)
    assert mock_session_cls.call_count == 2
    # Only committed in the update session
    assert mock_session.commit.call_count == 1
    assert job.status == "done"
