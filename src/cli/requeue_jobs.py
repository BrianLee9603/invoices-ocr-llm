import asyncio
import logging
import sys
from sqlalchemy import select

from src.config.settings import get_settings
from src.database.database import AsyncSessionLocal
from src.database.models import Job
from src.database.queue import RedisMessageQueue
from src.services.processing.worker import QUEUE_INGESTION

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cli.requeue")

async def main():
    logger.info("Initializing dependencies for Re-queue utility...")
    message_queue = RedisMessageQueue(settings.redis)
    
    logger.info("Connecting to database to check for 'queued' jobs...")
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Job).where(Job.status == "queued").order_by(Job.created_at.asc())
        )
        queued_jobs = result.scalars().all()
        
        if not queued_jobs:
            logger.info("No jobs found in 'queued' state. Nothing to re-queue.")
            await message_queue.close()
            return
            
        logger.info("Found %d jobs in 'queued' state. Re-queueing to Redis...", len(queued_jobs))
        
        success_count = 0
        for job in queued_jobs:
            try:
                payload = {
                    "job_id": str(job.id),
                    "tenant_id": str(job.tenant_id),
                    "input_file_path": job.input_file_path,
                }
                msg_id = await message_queue.publish(QUEUE_INGESTION, payload)
                logger.info("Published job %s to %s (msg_id: %s)", job.id, QUEUE_INGESTION, msg_id)
                success_count += 1
            except Exception as exc:
                logger.error("Failed to publish job %s: %s", job.id, exc)
                
        logger.info("Re-queue process complete. Successfully re-queued %d/%d jobs.", success_count, len(queued_jobs))
        
    await message_queue.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Re-queue utility terminated.")
