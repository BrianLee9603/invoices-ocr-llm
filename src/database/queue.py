"""
MessageQueue — Abstract interface and Redis Streams implementation.

Adapter pattern: business logic depends only on the `MessageQueue` ABC.
Redis Streams with Consumer Groups provide reliable, persistent messaging
with at-least-once delivery semantics.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import redis.asyncio as aioredis

from src.config.settings import RedisSettings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Abstract Base Class
# ──────────────────────────────────────────────────────────

class MessageQueue(ABC):
    """Abstract interface for asynchronous message queue operations."""

    @abstractmethod
    async def publish(self, topic: str, message: dict) -> str:
        """
        Publish a message to a topic (Redis Stream).

        Returns the message ID assigned by the broker.
        """
        ...

    @abstractmethod
    async def subscribe(
        self,
        topic: str,
        group_name: str,
        consumer_name: str,
        handler: Callable[[str, dict], Any],
    ) -> None:
        """
        Subscribe to a topic within a consumer group.

        Blocks and continuously reads new messages, invoking ``handler``
        for each one.  The handler receives ``(message_id, payload_dict)``.
        """
        ...

    @abstractmethod
    async def ack(self, topic: str, group_name: str, message_id: str) -> None:
        """Acknowledge successful processing of a message."""
        ...

    @abstractmethod
    async def reclaim_pending(
        self,
        topic: str,
        group_name: str,
        consumer_name: str,
        handler: Callable[[str, dict], Any],
        idle_ms: int = 30000,
        batch_count: int = 10,
    ) -> int:
        """
        Scan the PEL for messages idle > idle_ms and re-process them.
        Returns the number of messages successfully reclaimed and processed.
        """
        ...


# ──────────────────────────────────────────────────────────
#  Redis Streams Implementation
# ──────────────────────────────────────────────────────────

class RedisMessageQueue(MessageQueue):
    """
    MessageQueue implementation using Redis Streams + Consumer Groups.

    Key semantics:
    - ``publish``  → ``XADD`` to a stream
    - ``subscribe`` → ``XREADGROUP`` in a blocking loop
    - ``ack``       → ``XACK``

    If a consumer crashes before calling ``ack``, the message remains
    in the Pending Entries List (PEL) and can be reclaimed via
    ``XPENDING`` / ``XCLAIM``.
    """

    def __init__(
        self,
        settings: RedisSettings,
        block_ms: int = 5000,
        batch_count: int = 1,
    ) -> None:
        self._redis = aioredis.from_url(
            settings.url,
            decode_responses=True,
        )
        self._block_ms = block_ms
        self._batch_count = batch_count

    # -- interface --------------------------------------------------------

    async def publish(self, topic: str, message: dict) -> str:
        """Publish a JSON-serialized message to a Redis Stream."""
        payload = {"data": json.dumps(message)}
        message_id: str = await self._redis.xadd(topic, payload, maxlen=10000, approximate=True)
        logger.debug("XADD %s → %s", topic, message_id)
        return message_id

    async def subscribe(
        self,
        topic: str,
        group_name: str,
        consumer_name: str,
        handler: Callable[[str, dict], Any],
    ) -> None:
        """
        Blocking consumer loop — reads from a Redis Stream consumer group.

        Creates the consumer group if it does not exist (``MKSTREAM``).
        """
        # Ensure the consumer group exists
        try:
            await self._redis.xgroup_create(
                topic, group_name, id="0", mkstream=True,
            )
            logger.info(
                "Created consumer group '%s' on stream '%s'.",
                group_name, topic,
            )
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            # Group already exists — that's fine

        logger.info(
            "Consumer '%s' subscribing to '%s' (group='%s').",
            consumer_name, topic, group_name,
        )

        # 1. Process pending messages for this consumer on startup (non-blocking)
        logger.info(
            "Consumer '%s' checking for pending messages in '%s' (group='%s').",
            consumer_name, topic, group_name,
        )
        try:
            recovered_ids = set()
            while True:
                pending_entries = await self._redis.xreadgroup(
                    groupname=group_name,
                    consumername=consumer_name,
                    streams={topic: "0"},
                    count=self._batch_count,
                )
                if not pending_entries:
                    break

                processed_any = False
                for _stream, messages in pending_entries:
                    for message_id, fields in messages:
                        if message_id in recovered_ids:
                            # Avoid infinite loop if message fails to process and isn't ACKed
                            continue
                        recovered_ids.add(message_id)
                        processed_any = True
                        try:
                            payload = json.loads(fields["data"])
                            logger.info("Recovering pending message %s", message_id)
                            await handler(message_id, payload)
                            await self.ack(topic, group_name, message_id)
                        except Exception:
                            logger.exception(
                                "Error processing pending message %s from '%s'.",
                                message_id, topic,
                            )
                
                # If we couldn't process any new pending message (all were already attempted), break
                if not processed_any:
                    break
        except Exception as p_exc:
            logger.exception("Error checking/processing pending messages: %s", p_exc)

        while True:
            try:
                # XREADGROUP: read new messages (">") for this consumer
                entries = await self._redis.xreadgroup(
                    groupname=group_name,
                    consumername=consumer_name,
                    streams={topic: ">"},
                    count=self._batch_count,
                    block=self._block_ms,
                )

                if not entries:
                    continue

                for _stream, messages in entries:
                    for message_id, fields in messages:
                        try:
                            payload = json.loads(fields["data"])
                            await handler(message_id, payload)
                            await self.ack(topic, group_name, message_id)
                        except Exception:
                            logger.exception(
                                "Error processing message %s from '%s'.",
                                message_id, topic,
                            )
                            # Message stays in PEL for later reclaim

            except asyncio.CancelledError:
                logger.info("Consumer '%s' shutting down.", consumer_name)
                break
            except (aioredis.TimeoutError, TimeoutError):
                # Standard timeout when no messages arrive during block_ms
                logger.debug("Timeout waiting for messages on '%s'.", topic)
                continue
            except Exception:
                logger.exception("Unexpected error in consumer loop.")
                await asyncio.sleep(1)  # back off before retry

    async def ack(self, topic: str, group_name: str, message_id: str) -> None:
        """Acknowledge a message so it is removed from the PEL."""
        await self._redis.xack(topic, group_name, message_id)
        logger.debug("XACK %s/%s → %s", topic, group_name, message_id)

    async def reclaim_pending(
        self,
        topic: str,
        group_name: str,
        consumer_name: str,
        handler: Callable[[str, dict], Any],
        idle_ms: int = 30000,
        batch_count: int = 10,
    ) -> int:
        """
        Scan the PEL for messages idle > idle_ms and re-process them.
        Returns the number of messages successfully reclaimed and processed.
        """
        try:
            pending_entries = await self._redis.xpending_range(
                name=topic,
                groupname=group_name,
                min="-",
                max="+",
                count=batch_count,
                idle=idle_ms,
            )
            if not pending_entries:
                return 0

            message_ids = [entry["message_id"] for entry in pending_entries]
            
            # Claim ownership of these messages
            claimed_messages = await self._redis.xclaim(
                name=topic,
                groupname=group_name,
                consumername=consumer_name,
                min_idle_time=idle_ms,
                message_ids=message_ids,
            )

            if not claimed_messages:
                return 0

            processed_count = 0
            for message_id, fields in claimed_messages:
                if not message_id or not fields:
                    continue
                try:
                    payload = json.loads(fields["data"])
                    payload["retry_count"] = int(payload.get("retry_count", 0)) + 1
                    await handler(message_id, payload)
                    await self.ack(topic, group_name, message_id)
                    processed_count += 1
                except Exception:
                    logger.exception(
                        "Error processing reclaimed message %s from '%s'.",
                        message_id, topic,
                    )
            return processed_count
        except Exception as exc:
            logger.exception("Error in reclaim_pending for '%s': %s", topic, exc)
            return 0

    # -- lifecycle --------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        await self._redis.aclose()
