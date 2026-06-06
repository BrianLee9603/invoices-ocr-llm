"""
Integration smoke tests for the local infrastructure layer.

Prerequisites:
    cd infra && docker compose up -d

Run:
    python -m pytest tests/integration/test_infra.py -v
"""

import asyncio
import uuid

import pytest
import pytest_asyncio

from src.config.settings import get_settings
from src.database.blob_store import MinioBlobStore
from src.database.queue import RedisMessageQueue


# ──────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def blob_store(settings):
    return MinioBlobStore(settings.minio)


@pytest_asyncio.fixture(scope="module")
async def queue(settings):
    q = RedisMessageQueue(settings.redis)
    yield q
    await q.close()


# ──────────────────────────────────────────────────────────
#  PostgreSQL
# ──────────────────────────────────────────────────────────

class TestPostgres:
    """Verify PostgreSQL schema was initialized correctly."""

    @pytest.mark.asyncio
    async def test_tables_exist(self, settings):
        """Check that tenants and jobs tables exist."""
        import asyncpg

        conn = await asyncpg.connect(
            host=settings.database.host,
            port=settings.database.port,
            user=settings.database.user,
            password=settings.database.password,
            database=settings.database.db,
        )
        try:
            tables = await conn.fetch(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
                """
            )
            table_names = [row["table_name"] for row in tables]
            assert "tenants" in table_names, f"Missing 'tenants' table. Found: {table_names}"
            assert "jobs" in table_names, f"Missing 'jobs' table. Found: {table_names}"
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_default_tenant_exists(self, settings):
        """Check that the seed default tenant was created."""
        import asyncpg

        conn = await asyncpg.connect(
            host=settings.database.host,
            port=settings.database.port,
            user=settings.database.user,
            password=settings.database.password,
            database=settings.database.db,
        )
        try:
            row = await conn.fetchrow(
                "SELECT name FROM tenants WHERE name = 'default';"
            )
            assert row is not None, "Default tenant not found"
            assert row["name"] == "default"
        finally:
            await conn.close()


# ──────────────────────────────────────────────────────────
#  MinIO (BlobStore)
# ──────────────────────────────────────────────────────────

class TestMinIO:
    """Verify MinIO blob store operations."""

    TEST_BUCKET = "test-infra-smoke"
    TEST_KEY = f"smoke-test/{uuid.uuid4().hex}.txt"
    TEST_DATA = b"Hello from the integration smoke test!"

    @pytest.mark.asyncio
    async def test_put_get_exists_delete(self, blob_store):
        """Full lifecycle: put → exists → get → delete → not exists."""
        # PUT
        await blob_store.put(self.TEST_BUCKET, self.TEST_KEY, self.TEST_DATA)

        # EXISTS
        assert await blob_store.exists(self.TEST_BUCKET, self.TEST_KEY)

        # GET
        data = await blob_store.get(self.TEST_BUCKET, self.TEST_KEY)
        assert data == self.TEST_DATA

        # DELETE
        await blob_store.delete(self.TEST_BUCKET, self.TEST_KEY)

        # NOT EXISTS
        assert not await blob_store.exists(self.TEST_BUCKET, self.TEST_KEY)


# ──────────────────────────────────────────────────────────
#  Redis (MessageQueue)
# ──────────────────────────────────────────────────────────

class TestRedis:
    """Verify Redis Streams message queue operations."""

    TEST_TOPIC = f"test:smoke:{uuid.uuid4().hex[:8]}"

    @pytest.mark.asyncio
    async def test_publish_and_consume(self, queue):
        """Publish a message, then consume and ack it."""
        test_payload = {"job_id": str(uuid.uuid4()), "action": "smoke_test"}

        # Publish
        message_id = await queue.publish(self.TEST_TOPIC, test_payload)
        assert message_id is not None

        # Consume via a one-shot handler
        received: list[dict] = []

        async def handler(msg_id: str, payload: dict):
            received.append(payload)

        # Create consumer group and read
        group = "test-group"
        consumer = "test-consumer"

        # We'll subscribe in a task and cancel after receiving
        async def subscribe_with_timeout():
            try:
                await asyncio.wait_for(
                    queue.subscribe(self.TEST_TOPIC, group, consumer, handler),
                    timeout=3.0,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        await subscribe_with_timeout()

        assert len(received) == 1
        assert received[0]["job_id"] == test_payload["job_id"]
