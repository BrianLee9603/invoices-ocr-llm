import pytest
import uuid
from src.database.models import Job

def test_initial_status_creation():
    # Initial status can be set to any valid value at initialization
    job = Job(id=uuid.uuid4(), tenant_id=uuid.uuid4(), status="queued")
    assert job.status == "queued"

def test_valid_transitions():
    job = Job(id=uuid.uuid4(), tenant_id=uuid.uuid4(), status="queued")
    
    # queued -> ocr_processing
    job.status = "ocr_processing"
    assert job.status == "ocr_processing"
    
    # ocr_processing -> ocr_done
    job.status = "ocr_done"
    assert job.status == "ocr_done"
    
    # ocr_done -> extracting
    job.status = "extracting"
    assert job.status == "extracting"
    
    # extracting -> extracted
    job.status = "extracted"
    assert job.status == "extracted"
    
    # extracted -> done
    job.status = "done"
    assert job.status == "done"

def test_retry_transitions():
    # Test ocr_processing -> queued retry
    job1 = Job(id=uuid.uuid4(), tenant_id=uuid.uuid4(), status="queued")
    job1.status = "ocr_processing"
    job1.status = "queued"
    assert job1.status == "queued"

    # Test extracting -> queued retry
    job2 = Job(id=uuid.uuid4(), tenant_id=uuid.uuid4(), status="queued")
    job2.status = "ocr_processing"
    job2.status = "ocr_done"
    job2.status = "extracting"
    job2.status = "queued"
    assert job2.status == "queued"

def test_failure_transitions():
    # From queued
    job = Job(id=uuid.uuid4(), tenant_id=uuid.uuid4(), status="queued")
    job.status = "failed"
    assert job.status == "failed"
    
    # From ocr_processing
    job = Job(id=uuid.uuid4(), tenant_id=uuid.uuid4(), status="queued")
    job.status = "ocr_processing"
    job.status = "failed"
    assert job.status == "failed"
    
    # From extracting
    job = Job(id=uuid.uuid4(), tenant_id=uuid.uuid4(), status="queued")
    job.status = "ocr_processing"
    job.status = "ocr_done"
    job.status = "extracting"
    job.status = "failed"
    assert job.status == "failed"

    # From extracted
    job = Job(id=uuid.uuid4(), tenant_id=uuid.uuid4(), status="queued")
    job.status = "ocr_processing"
    job.status = "ocr_done"
    job.status = "extracting"
    job.status = "extracted"
    job.status = "failed"
    assert job.status == "failed"

def test_self_transitions():
    job = Job(id=uuid.uuid4(), tenant_id=uuid.uuid4(), status="queued")
    job.status = "queued" # no change, should pass
    assert job.status == "queued"

    job.status = "ocr_processing"
    job.status = "ocr_processing" # no change, should pass
    assert job.status == "ocr_processing"

def test_invalid_transitions():
    job = Job(id=uuid.uuid4(), tenant_id=uuid.uuid4(), status="queued")
    
    # queued -> done (invalid)
    with pytest.raises(ValueError) as excinfo:
        job.status = "done"
    assert "Invalid job status transition" in str(excinfo.value)
    
    job.status = "ocr_processing"
    job.status = "ocr_done"
    
    # ocr_done -> queued (invalid, must progress to extracting or fail)
    with pytest.raises(ValueError) as excinfo:
        job.status = "queued"
    assert "Invalid job status transition" in str(excinfo.value)

def test_terminal_states_cannot_transition():
    # From done
    job_done = Job(id=uuid.uuid4(), tenant_id=uuid.uuid4(), status="queued")
    job_done.status = "ocr_processing"
    job_done.status = "ocr_done"
    job_done.status = "extracting"
    job_done.status = "extracted"
    job_done.status = "done"
    
    with pytest.raises(ValueError) as excinfo:
        job_done.status = "queued"
    assert "Invalid job status transition" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        job_done.status = "ocr_processing"
    assert "Invalid job status transition" in str(excinfo.value)

    # From failed
    job_failed = Job(id=uuid.uuid4(), tenant_id=uuid.uuid4(), status="queued")
    job_failed.status = "failed"
    
    with pytest.raises(ValueError) as excinfo:
        job_failed.status = "queued"
    assert "Invalid job status transition" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        job_failed.status = "done"
    assert "Invalid job status transition" in str(excinfo.value)
