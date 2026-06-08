import asyncio
import logging
import sys

from src.config.settings import get_settings
from src.database.blob_store import MinioBlobStore
from src.database.queue import RedisMessageQueue
from src.services.processing.ocr import create_ocr_engine
from src.services.processing.extractor import create_extractor
from src.services.processing.worker import ProcessingWorker

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cli.worker")

async def main():
    logger.info("Initializing Processing Worker dependencies...")
    
    blob_store = MinioBlobStore(settings.minio)
    message_queue = RedisMessageQueue(settings.redis)
    
    ocr_engine_name = settings.processing.ocr_engine
    logger.info("Using OCR Engine: %s", ocr_engine_name)
    ocr_engine = create_ocr_engine(ocr_engine_name)
    
    logger.info("Using LLM Provider: %s", settings.llm_provider)
    extractor = create_extractor(settings)

    worker = ProcessingWorker(
        blob_store=blob_store,
        queue=message_queue,
        ocr_engine=ocr_engine,
        extractor=extractor,
        worker_id="1"
    )
    
    logger.info("Starting Processing Worker. Press Ctrl+C to stop.")
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
