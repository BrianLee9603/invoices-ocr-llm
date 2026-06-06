"""
Integration tests for the Ingestion Service API endpoints.

Prerequisites:
    cd infra && docker compose up -d

Run:
    python -m pytest tests/integration/test_ingestion.py -v
"""

from __future__ import annotations

import asyncio
import uuid
import pytest
import pytest_asyncio
import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.server import app
from src.database.database import AsyncSessionLocal
from src.database.models import Job, Tenant


@pytest_asyncio.fixture(autouse=True)
async def cleanup_database_pool():
    """Dispose the database engine pool to prevent connection sharing across event loops."""
    from src.database.database import engine
    await engine.dispose()
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def client():
    """Create an async HTTPX client that manages lifespan events."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


# ──────────────────────────────────────────────────────────
#  Test API Lifecycle & Endpoints
# ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    """Verify that the health check endpoint works."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_ingest_file_validation_error(client):
    """Verify upload rejects invalid file extensions."""
    files = {"file": ("test.txt", b"some text content", "text/plain")}
    response = await client.post("/ingest", files=files)
    assert response.status_code == 422
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_ingest_file_success(client):
    """Verify upload of a valid image succeeds and queues the job."""
    # Create a tiny 1x1 png image in memory
    img_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    files = {"file": ("invoice_test.png", img_bytes, "image/png")}
    
    response = await client.post("/ingest", files=files)
    assert response.status_code == 202
    res_data = response.json()
    assert res_data["status"] == "accepted"
    assert "job_id" in res_data
    
    job_id_str = res_data["job_id"]
    job_id = uuid.UUID(job_id_str)
    
    # 1. Verify Job exists in database and is 'queued'
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalars().first()
        assert job is not None
        assert job.status == "queued"
        assert "invoices/" in job.input_file_path
        assert job.ground_truth is None
    
    # 2. Verify we can fetch the status via the GET endpoint
    status_response = await client.get(f"/ingest/jobs/{job_id_str}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["job_id"] == job_id_str
    assert status_data["status"] == "queued"
    
    # 3. Verify file exists in MinIO (Blob Store)
    blob_store = app.state.blob_store
    bucket, path = job.input_file_path.split("/", 1)
    assert await blob_store.exists(bucket, path)
    
    # Clean up minio test file
    await blob_store.delete(bucket, path)


@pytest.mark.asyncio
async def test_dataset_ingest_inline(client):
    """Verify dataset ingestion with limit <= 5 (runs inline)."""
    # Ingest limit=2 samples from "test" split
    payload = {"split": "test", "limit": 2}
    response = await client.post("/ingest/dataset", json=payload)
    
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert "Ingested 2 samples" in data["message"]
    assert data["total_jobs"] == 2
    assert len(data["job_ids"]) == 2
    
    # Check that jobs were written with ground truth data
    async with AsyncSessionLocal() as db:
        for job_id_str in data["job_ids"]:
            job_uuid = uuid.UUID(job_id_str)
            result = await db.execute(select(Job).where(Job.id == job_uuid))
            job = result.scalars().first()
            assert job is not None
            assert job.status == "queued"
            # Should have loaded ground truth from HF dataset
            assert job.ground_truth is not None
            
            # Cleanup MinIO
            blob_store = app.state.blob_store
            bucket, path = job.input_file_path.split("/", 1)
            await blob_store.delete(bucket, path)


@pytest.mark.asyncio
async def test_dataset_ingest_background(client):
    """Verify dataset ingestion with limit > 5 (runs in background)."""
    payload = {"split": "test", "limit": 6}
    response = await client.post("/ingest/dataset", json=payload)
    
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert "started in background" in data["message"]
    assert data["total_jobs"] == 0
    assert data["job_ids"] == []
    
    # Wait a few seconds for background task to complete processing
    await asyncio.sleep(5)
    
    # Verify that jobs were successfully created
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Job).order_by(Job.created_at.desc()).limit(6))
        jobs = result.scalars().all()
        assert len(jobs) >= 6
        for job in jobs[:6]:
            assert job.status == "queued"
            assert job.ground_truth is not None
            
            # Cleanup MinIO
            blob_store = app.state.blob_store
            bucket, path = job.input_file_path.split("/", 1)
            await blob_store.delete(bucket, path)
