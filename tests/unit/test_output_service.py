import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json
import uuid

from src.schemas.document import InvoiceExtraction
from src.services.output.validator import parse_raw_amount, validate_extraction
from src.services.output.evaluator import evaluate_extraction
from src.services.output.worker import OutputWorker
from src.database.models import Job

# ----------------- Validator Tests -----------------

def test_parse_raw_amount():
    assert parse_raw_amount("1,234.56") == 1234.56
    assert parse_raw_amount("1.234,56") == 1234.56
    assert parse_raw_amount("$ 24 161,60") == 24161.60
    assert parse_raw_amount("each 444,60") == 444.60
    assert parse_raw_amount("-123.45") == -123.45
    assert parse_raw_amount("abc") is None
    assert parse_raw_amount("") is None
    assert parse_raw_amount(None) is None
    assert parse_raw_amount("123,45") == 123.45
    assert parse_raw_amount("123,4") == 1234.0

def test_validate_extraction_valid():
    extraction = InvoiceExtraction(
        header={
            "invoice_no": "INV-123",
            "invoice_date": "2026-06-08"
        },
        items=[],
        summary={
            "total_net_worth": "100.00",
            "total_vat": "10.00",
            "total_gross_worth": "110.00"
        }
    )
    assert validate_extraction(extraction) is True

def test_validate_extraction_invalid_missing_fields():
    # missing invoice_no
    extraction = InvoiceExtraction(
        header={
            "invoice_no": "",
            "invoice_date": "2026-06-08"
        },
        items=[],
        summary={
            "total_net_worth": "100.00"
        }
    )
    assert validate_extraction(extraction) is False

def test_validate_extraction_vat_mismatch():
    # VAT mismatch shouldn't fail validation since VAT is optional, but it logs a warning
    extraction = InvoiceExtraction(
        header={
            "invoice_no": "INV-123",
            "invoice_date": "2026-06-08"
        },
        items=[],
        summary={
            "total_net_worth": "100.00",
            "total_vat": "10.00",
            "total_gross_worth": "150.00" # wrong gross
        }
    )
    assert validate_extraction(extraction) is True

def test_validate_extraction_items_sum():
    extraction = InvoiceExtraction(
        header={
            "invoice_no": "INV-123",
            "invoice_date": "2026-06-08"
        },
        items=[
            {"item_desc": "Item 1", "item_net_worth": "40.00"},
            {"item_desc": "Item 2", "item_net_worth": "60.00"}
        ],
        summary={
            "total_net_worth": "100.00"
        }
    )
    assert validate_extraction(extraction) is True


# ----------------- Evaluator Tests -----------------

def test_evaluate_extraction_semantic_match():
    # Test that different formatting but identical content evaluates to passed = True
    extraction = InvoiceExtraction(
        header={
            "invoice_no": "inv - 123", # extra spaces and lowercase
            "invoice_date": "06/08/2026" # MM/DD/YYYY format
        },
        items=[],
        summary={
            "total_net_worth": "100.00"
        }
    )
    ground_truth = {
        "json": {
            "header": {
                "invoice_no": "INV-123", # uppercase and dash
                "invoice_date": "2026-06-08" # YYYY-MM-DD format
            },
            "summary": {
                "total_net_worth": "$ 100.0" # with dollar prefix and single digit decimal
            }
        }
    }
    passed, accuracies = evaluate_extraction(extraction, ground_truth)
    assert passed is True
    assert accuracies == {
        "invoice_no": 1.0,
        "invoice_date": 1.0,
        "total_net_worth": 1.0
    }

def test_evaluate_extraction_actual_mismatch():
    extraction = InvoiceExtraction(
        header={
            "invoice_no": "INV-123",
            "invoice_date": "2026-06-08"
        },
        items=[],
        summary={
            "total_net_worth": "100.00"
        }
    )
    ground_truth = {
        "json": {
            "header": {
                "invoice_no": "INV-999", # totally different invoice number
                "invoice_date": "2026-06-08"
            },
            "summary": {
                "total_net_worth": "100.00"
            }
        }
    }
    passed, accuracies = evaluate_extraction(extraction, ground_truth)
    assert passed is False
    assert accuracies == {
        "invoice_no": 0.0,
        "invoice_date": 1.0,
        "total_net_worth": 1.0
    }

def test_evaluate_extraction_unparsable_gt():
    extraction = InvoiceExtraction(
        header={
            "invoice_no": "INV-123",
            "invoice_date": "2026-06-08"
        },
        items=[],
        summary={
            "total_net_worth": "100.00"
        }
    )
    ground_truth = {"json": "invalid-json-string{"}
    passed, accuracies = evaluate_extraction(extraction, ground_truth)
    assert passed is False
    assert accuracies == {
        "invoice_no": 0.0,
        "invoice_date": 0.0,
        "total_net_worth": 0.0
    }


# ----------------- Worker Tests -----------------

@pytest.mark.asyncio
async def test_worker_handle_message_success():
    mock_blob_store = AsyncMock()
    mock_queue = AsyncMock()
    
    worker = OutputWorker(
        blob_store=mock_blob_store,
        queue=mock_queue,
        worker_id="test-worker"
    )
    
    job_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    
    job = Job(
        id=job_id,
        tenant_id=tenant_id,
        status="extracted",
        ground_truth={
            "json": {
                "header": {
                    "invoice_no": "INV-123",
                    "invoice_date": "2026-06-08"
                },
                "summary": {
                    "total_net_worth": "100.00"
                }
            }
        }
    )
    
    # Mock database session
    mock_session = AsyncMock()
    mock_session.get.return_value = job
    
    mock_session_cls = MagicMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    
    payload = {
        "job_id": str(job_id),
        "tenant_id": str(tenant_id),
        "ocr_confidence": 0.95,
        "extraction_data": {
            "header": {
                "invoice_no": "INV-123",
                "invoice_date": "2026-06-08"
            },
            "items": [],
            "summary": {
                "total_net_worth": "100.00"
            }
        }
    }
    
    with patch("src.services.output.worker.AsyncSessionLocal", mock_session_cls), \
         patch("src.services.base_worker.AsyncSessionLocal", mock_session_cls):
        await worker.handle_message("msg-123", payload)
        
    # Check that blob store upload was called
    mock_blob_store.put.assert_called_once()
    bucket, path, data = mock_blob_store.put.call_args[0]
    assert bucket == "invoices"
    assert path == f"{tenant_id}/{job_id}/final_output.json"
    
    uploaded_json = json.loads(data.decode("utf-8"))
    assert uploaded_json["job_id"] == str(job_id)
    assert uploaded_json["ocr_confidence"] == 0.95
    assert uploaded_json["evaluation"]["passed"] is True
    
    # Verify job status was updated to done
    assert job.status == "done"
    assert job.evaluation_data["passed"] is True

@pytest.mark.asyncio
async def test_worker_handle_message_idempotency():
    mock_blob_store = AsyncMock()
    mock_queue = AsyncMock()
    
    worker = OutputWorker(
        blob_store=mock_blob_store,
        queue=mock_queue,
        worker_id="test-worker"
    )
    
    job_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    
    # Already completed job
    job = Job(
        id=job_id,
        tenant_id=tenant_id,
        status="done"
    )
    
    mock_session = AsyncMock()
    mock_session.get.return_value = job
    mock_session_cls = MagicMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    
    payload = {
        "job_id": str(job_id),
        "tenant_id": str(tenant_id),
        "ocr_confidence": 0.95,
        "extraction_data": {
            "header": {
                "invoice_no": "INV-123",
                "invoice_date": "2026-06-08"
            },
            "items": [],
            "summary": {
                "total_net_worth": "100.00"
            }
        }
    }
    
    with patch("src.services.output.worker.AsyncSessionLocal", mock_session_cls), \
         patch("src.services.base_worker.AsyncSessionLocal", mock_session_cls):
        await worker.handle_message("msg-123", payload)
        
    # Blob store put should NOT be called
    mock_blob_store.put.assert_not_called()
    # Status should remain done
    assert job.status == "done"
