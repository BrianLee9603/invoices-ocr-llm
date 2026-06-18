import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from botocore.exceptions import ClientError

from src.config.settings import MinioSettings
from src.database.blob_store import MinioBlobStore


@pytest.fixture
def mock_minio_settings():
    settings = MagicMock(spec=MinioSettings)
    settings.endpoint = "http://localhost:9000"
    settings.access_key = "minioadmin"
    settings.secret_key = "minioadminpassword"
    settings.secure = False
    return settings


@pytest.mark.asyncio
async def test_minio_blob_store_put_bucket_exists(mock_minio_settings):
    store = MinioBlobStore(mock_minio_settings)
    mock_client = AsyncMock()

    # Mock async context manager for client
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_client

    with patch.object(store, "_get_client", return_value=mock_ctx):
        # bucket head request returns successfully (bucket exists)
        mock_client.head_bucket.return_value = {}

        await store.put("my-bucket", "file.txt", b"hello world")

        mock_client.head_bucket.assert_awaited_once_with(Bucket="my-bucket")
        mock_client.create_bucket.assert_not_awaited()
        mock_client.put_object.assert_awaited_once()


@pytest.mark.asyncio
async def test_minio_blob_store_put_bucket_does_not_exist(mock_minio_settings):
    store = MinioBlobStore(mock_minio_settings)
    mock_client = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_client

    with patch.object(store, "_get_client", return_value=mock_ctx):
        # bucket head request raises ClientError (bucket doesn't exist)
        mock_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
        )
        mock_client.create_bucket.return_value = {}

        await store.put("my-bucket", "file.txt", b"hello world")

        mock_client.head_bucket.assert_awaited_once_with(Bucket="my-bucket")
        mock_client.create_bucket.assert_awaited_once_with(Bucket="my-bucket")
        mock_client.put_object.assert_awaited_once()


@pytest.mark.asyncio
async def test_minio_blob_store_get_success(mock_minio_settings):
    store = MinioBlobStore(mock_minio_settings)
    mock_client = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_client

    # Mock body stream as an async context manager
    mock_stream = AsyncMock()
    mock_stream.__aenter__.return_value = mock_stream
    mock_stream.read.return_value = b"retrieved content"

    mock_client.get_object.return_value = {"Body": mock_stream}

    with patch.object(store, "_get_client", return_value=mock_ctx):
        data = await store.get("my-bucket", "file.txt")
        assert data == b"retrieved content"
        mock_client.get_object.assert_awaited_once_with(Bucket="my-bucket", Key="file.txt")


@pytest.mark.asyncio
async def test_minio_blob_store_exists(mock_minio_settings):
    store = MinioBlobStore(mock_minio_settings)
    mock_client = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_client

    with patch.object(store, "_get_client", return_value=mock_ctx):
        mock_client.head_object.return_value = {}
        assert await store.exists("my-bucket", "file.txt") is True

        mock_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        assert await store.exists("my-bucket", "file.txt") is False


@pytest.mark.asyncio
async def test_minio_blob_store_delete(mock_minio_settings):
    store = MinioBlobStore(mock_minio_settings)
    mock_client = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_client

    with patch.object(store, "_get_client", return_value=mock_ctx):
        await store.delete("my-bucket", "file.txt")
        mock_client.delete_object.assert_awaited_once_with(Bucket="my-bucket", Key="file.txt")
