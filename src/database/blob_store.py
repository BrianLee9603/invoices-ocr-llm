"""
BlobStore — Abstract interface and MinIO implementation for object storage.

Adapter pattern: business logic depends only on the `BlobStore` ABC.
Swap implementations via configuration without changing service code.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod

import boto3
from botocore.exceptions import ClientError

from src.config.settings import MinioSettings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Abstract Base Class
# ──────────────────────────────────────────────────────────

class BlobStore(ABC):
    """Abstract interface for blob/object storage operations."""

    @abstractmethod
    async def put(self, bucket: str, path: str, data: bytes) -> None:
        """Upload data to a specific bucket path."""
        ...

    @abstractmethod
    async def get(self, bucket: str, path: str) -> bytes:
        """Download data from a specific bucket path."""
        ...

    @abstractmethod
    async def exists(self, bucket: str, path: str) -> bool:
        """Check if an object exists at the path."""
        ...

    @abstractmethod
    async def delete(self, bucket: str, path: str) -> None:
        """Remove an object from the path."""
        ...


# ──────────────────────────────────────────────────────────
#  MinIO Implementation (S3-compatible)
# ──────────────────────────────────────────────────────────

class MinioBlobStore(BlobStore):
    """
    BlobStore implementation backed by MinIO (S3-compatible API).

    Uses boto3 synchronous client under the hood — boto3 does not have
    native asyncio support, so I/O-bound calls are acceptable for the
    expected local-dev workload. For high-throughput production use,
    consider wrapping in ``asyncio.to_thread()``.
    """

    def __init__(self, settings: MinioSettings) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint,
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=settings.secret_key,
            use_ssl=settings.secure,
            # MinIO does not use AWS regions
            region_name="us-east-1",
        )

    # -- helpers ----------------------------------------------------------

    def _ensure_bucket(self, bucket: str) -> None:
        """Create the bucket if it does not already exist."""
        try:
            self._client.head_bucket(Bucket=bucket)
        except ClientError:
            logger.info("Bucket '%s' not found — creating.", bucket)
            self._client.create_bucket(Bucket=bucket)

    # -- interface --------------------------------------------------------

    async def put(self, bucket: str, path: str, data: bytes) -> None:
        self._ensure_bucket(bucket)
        self._client.put_object(
            Bucket=bucket,
            Key=path,
            Body=io.BytesIO(data),
            ContentLength=len(data),
        )
        logger.debug("PUT %s/%s (%d bytes)", bucket, path, len(data))

    async def get(self, bucket: str, path: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=bucket, Key=path)
            data = response["Body"].read()
            logger.debug("GET %s/%s (%d bytes)", bucket, path, len(data))
            return data
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(
                    f"Object not found: {bucket}/{path}"
                ) from exc
            raise

    async def exists(self, bucket: str, path: str) -> bool:
        try:
            self._client.head_object(Bucket=bucket, Key=path)
            return True
        except ClientError:
            return False

    async def delete(self, bucket: str, path: str) -> None:
        self._client.delete_object(Bucket=bucket, Key=path)
        logger.debug("DELETE %s/%s", bucket, path)
