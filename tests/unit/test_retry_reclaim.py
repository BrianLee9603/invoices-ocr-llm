import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json
import uuid

from src.database.queue import RedisMessageQueue
from src.services.processing.worker import ProcessingWorker
from src.services.output.worker import OutputWorker
from src.database.models import Job
from src.exceptions import PersistentError, TransientError
from src.config.settings import RedisSettings

# Mock settings for RedisMessageQueue initialization
@pytest.fixture
def mock_redis_settings():
    settings = MagicMock(spec=RedisSettings)
    settings.url = "redis://localhost:6379/0"
    return settings

@pytest.mark.asyncio
async def test_reclaim_pending_success(mock_redis_settings):
    # Mock redis client calls inside queue
    with patch("redis.asyncio.from_url") as mock_from_url:
        mock_redis = AsyncMock()
        mock_from_url.return_value = mock_redis
        
        queue = RedisMessageQueue(mock_redis_settings)
        
        # 1. Mock xpending_range to return one pending message
        mock_redis.xpending_range.return_value = [
            {
                "message_id": "12345-0",
                "consumer": "worker-1",
                "time_since_delivered": 35000,
                "times_delivered": 1,
            }
        ]
        
        # 2. Mock xclaim to return the claimed message details
        payload = {"job_id": "job-abc", "retry_count": 0}
        mock_redis.xclaim.return_value = [
            ("12345-0", {"data": json.dumps(payload)})
        ]
        
        handler_calls = []
        async def mock_handler(msg_id, pay):
            handler_calls.append((msg_id, pay))
            
        # Call reclaim_pending
        count = await queue.reclaim_pending(
            topic="test-topic",
            group_name="test-group",
            consumer_name="test-consumer",
            handler=mock_handler,
            idle_ms=30000,
        )
        
        # Verify handler was called with incremented retry count
        assert count == 1
        assert len(handler_calls) == 1
        assert handler_calls[0][0] == "12345-0"
        assert handler_calls[0][1]["retry_count"] == 1
        
        # Verify redis client called xack
        mock_redis.xack.assert_called_once_with("test-topic", "test-group", "12345-0")


@pytest.mark.asyncio
async def test_reclaim_pending_handler_failure(mock_redis_settings):
    with patch("redis.asyncio.from_url") as mock_from_url:
        mock_redis = AsyncMock()
        mock_from_url.return_value = mock_redis
        
        queue = RedisMessageQueue(mock_redis_settings)
        
        mock_redis.xpending_range.return_value = [
            {
                "message_id": "12345-0",
                "consumer": "worker-1",
                "time_since_delivered": 35000,
                "times_delivered": 1,
            }
        ]
        
        payload = {"job_id": "job-abc", "retry_count": 0}
        mock_redis.xclaim.return_value = [
            ("12345-0", {"data": json.dumps(payload)})
        ]
        
        async def mock_handler(msg_id, pay):
            raise RuntimeError("Transient processing error")
            
        # Call reclaim_pending
        count = await queue.reclaim_pending(
            topic="test-topic",
            group_name="test-group",
            consumer_name="test-consumer",
            handler=mock_handler,
            idle_ms=30000,
        )
        
        # Verify no success counted and no ACK was called
        assert count == 0
        mock_redis.xack.assert_not_called()


@pytest.mark.asyncio
async def test_processing_worker_transient_error_retry():
    mock_blob = AsyncMock()
    mock_queue = AsyncMock()
    mock_ocr = AsyncMock()
    mock_extractor = AsyncMock()
    
    worker = ProcessingWorker(
        blob_store=mock_blob,
        queue=mock_queue,
        ocr_engine=mock_ocr,
        extractor=mock_extractor,
        worker_id="test"
    )
    
    # Force process to fail with a TransientError
    mock_blob.get.side_effect = TransientError("MinIO timed out")
    
    job_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    payload = {
        "job_id": str(job_id),
        "tenant_id": str(tenant_id),
        "input_file_path": "invoices/tenant/job/input.pdf",
        "retry_count": 1,
    }
    
    # Mock database session
    job = Job(id=job_id, tenant_id=tenant_id, status="queued", retry_count=1)
    mock_session = AsyncMock()
    mock_session.get.return_value = job
    mock_session_cls = MagicMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    
    with patch("src.services.base_worker.AsyncSessionLocal", mock_session_cls), \
         patch("src.services.processing.worker.AsyncSessionLocal", mock_session_cls):
        with pytest.raises(TransientError):
            await worker.handle_message("msg-123", payload)
            
    # Verify job status was updated back to queued with retry_count incremented to 2
    assert job.status == "queued"
    assert job.retry_count == 2
    assert "Transient failure" in job.error_message


@pytest.mark.asyncio
async def test_processing_worker_transient_error_max_retries():
    mock_blob = AsyncMock()
    mock_queue = AsyncMock()
    mock_ocr = AsyncMock()
    mock_extractor = AsyncMock()
    
    worker = ProcessingWorker(
        blob_store=mock_blob,
        queue=mock_queue,
        ocr_engine=mock_ocr,
        extractor=mock_extractor,
        worker_id="test"
    )
    
    mock_blob.get.side_effect = TransientError("MinIO timed out")
    
    job_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    payload = {
        "job_id": str(job_id),
        "tenant_id": str(tenant_id),
        "input_file_path": "invoices/tenant/job/input.pdf",
        "retry_count": 3, # Max retries is 3
    }
    
    job = Job(id=job_id, tenant_id=tenant_id, status="queued", retry_count=3)
    mock_session = AsyncMock()
    mock_session.get.return_value = job
    mock_session_cls = MagicMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    
    with patch("src.services.base_worker.AsyncSessionLocal", mock_session_cls), \
         patch("src.services.processing.worker.AsyncSessionLocal", mock_session_cls):
        # Should NOT raise exception (so it gets ACKed)
        await worker.handle_message("msg-123", payload)
            
    # Verify job status was updated to failed
    assert job.status == "failed"
    assert "Failed:" in job.error_message


@pytest.mark.asyncio
async def test_processing_worker_persistent_error():
    mock_blob = AsyncMock()
    mock_queue = AsyncMock()
    mock_ocr = AsyncMock()
    mock_extractor = AsyncMock()
    
    worker = ProcessingWorker(
        blob_store=mock_blob,
        queue=mock_queue,
        ocr_engine=mock_ocr,
        extractor=mock_extractor,
        worker_id="test"
    )
    
    # Force process to fail with a PersistentError
    mock_blob.get.side_effect = PersistentError("Corrupted file format")
    
    job_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    payload = {
        "job_id": str(job_id),
        "tenant_id": str(tenant_id),
        "input_file_path": "invoices/tenant/job/input.pdf",
        "retry_count": 0,
    }
    
    job = Job(id=job_id, tenant_id=tenant_id, status="queued", retry_count=0)
    mock_session = AsyncMock()
    mock_session.get.return_value = job
    mock_session_cls = MagicMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    
    with patch("src.services.base_worker.AsyncSessionLocal", mock_session_cls), \
         patch("src.services.processing.worker.AsyncSessionLocal", mock_session_cls):
        # Should NOT raise exception (fails immediately and ACKs)
        await worker.handle_message("msg-123", payload)
            
    # Verify job status was updated to failed
    assert job.status == "failed"
    assert "Failed: Corrupted file format" in job.error_message


@pytest.mark.asyncio
async def test_output_worker_transient_error_retry():
    mock_blob = AsyncMock()
    mock_queue = AsyncMock()
    
    worker = OutputWorker(
        blob_store=mock_blob,
        queue=mock_queue,
        worker_id="test"
    )
    
    # Mock update status to throw TransientError on status update or blob put to trigger error
    mock_blob.put.side_effect = TransientError("Database deadlock")
    
    job_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    payload = {
        "job_id": str(job_id),
        "tenant_id": str(tenant_id),
        "ocr_confidence": 0.9,
        "extraction_data": {"header": {"invoice_no": "123", "invoice_date": "2026-06-18"}, "items": [], "summary": {"total_net_worth": "10.0"}},
        "retry_count": 0,
    }
    
    job = Job(id=job_id, tenant_id=tenant_id, status="extracted", retry_count=0)
    mock_session = AsyncMock()
    mock_session.get.return_value = job
    mock_session_cls = MagicMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    
    with patch("src.services.base_worker.AsyncSessionLocal", mock_session_cls), \
         patch("src.services.output.worker.AsyncSessionLocal", mock_session_cls):
        with pytest.raises(TransientError):
            await worker.handle_message("msg-123", payload)
            
    # Verify job status remains extracted (set back so output worker retries) with retry_count incremented to 1
    assert job.status == "extracted"
    assert job.retry_count == 1
    assert "Transient failure" in job.error_message


@pytest.mark.asyncio
async def test_output_worker_transient_error_max_retries():
    mock_blob = AsyncMock()
    mock_queue = AsyncMock()
    
    worker = OutputWorker(
        blob_store=mock_blob,
        queue=mock_queue,
        worker_id="test"
    )
    
    mock_blob.put.side_effect = TransientError("Connection lost")
    
    job_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    payload = {
        "job_id": str(job_id),
        "tenant_id": str(tenant_id),
        "ocr_confidence": 0.9,
        "extraction_data": {"header": {"invoice_no": "123", "invoice_date": "2026-06-18"}, "items": [], "summary": {"total_net_worth": "10.0"}},
        "retry_count": 3,
    }
    
    job = Job(id=job_id, tenant_id=tenant_id, status="extracted", retry_count=3)
    mock_session = AsyncMock()
    mock_session.get.return_value = job
    mock_session_cls = MagicMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    
    with patch("src.services.base_worker.AsyncSessionLocal", mock_session_cls), \
         patch("src.services.output.worker.AsyncSessionLocal", mock_session_cls):
        # Should NOT raise exception, and mark as failed
        await worker.handle_message("msg-123", payload)
            
    assert job.status == "failed"
    assert "Failed:" in job.error_message
