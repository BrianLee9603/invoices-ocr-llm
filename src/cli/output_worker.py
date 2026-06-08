import asyncio
import logging
import sys

from src.config.settings import get_settings
from src.database.blob_store import MinioBlobStore
from src.database.queue import RedisMessageQueue
from src.services.output.worker import OutputWorker

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cli.output_worker")

async def main():
    logger.info("Initializing Output Worker dependencies...")
    
    blob_store = MinioBlobStore(settings.minio)
    message_queue = RedisMessageQueue(settings.redis)
    
    worker = OutputWorker(
        blob_store=blob_store,
        queue=message_queue,
        worker_id="1"
    )
    
    logger.info("Starting Output Worker. Press Ctrl+C to stop.")
    try:
        await worker.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Received stop signal. Shutting down...")
    finally:
        await worker.stop()
        await message_queue.close()
        logger.info("Worker connections closed. Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker terminated.")
