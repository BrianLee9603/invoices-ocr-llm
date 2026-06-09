import asyncio
from src.database.database import AsyncSessionLocal
from sqlalchemy import text

async def main():
    async with AsyncSessionLocal() as session:
        # Get count of done jobs
        result_done = await session.execute(text("SELECT COUNT(*) FROM jobs WHERE status = 'done'"))
        done_count = result_done.scalar()
        print(f"Total Completed Jobs (status='done'): {done_count}")

        # Get count of failed jobs
        result_failed = await session.execute(text("SELECT COUNT(*) FROM jobs WHERE status = 'failed'"))
        failed_count = result_failed.scalar()
        print(f"Total Failed Jobs (status='failed'): {failed_count}")

        # Get status distribution of all jobs
        result_all = await session.execute(text("SELECT status, COUNT(*) FROM jobs GROUP BY status"))
        print("\nJob status distribution:")
        for row in result_all.fetchall():
            print(f"- {row[0]}: {row[1]}")

        # Let's inspect some of the recently completed jobs
        result_sample = await session.execute(text("SELECT id, status, updated_at FROM jobs WHERE status = 'done' ORDER BY updated_at DESC LIMIT 5"))
        print("\nSample completed jobs:")
        for row in result_sample.fetchall():
            print(f"- ID: {row[0]} | Updated: {row[2]}")

if __name__ == "__main__":
    asyncio.run(main())
