"""
Async database engine and session management using SQLAlchemy 2.0+ asyncio.

Usage:
    from src.database.database import get_db, Base

    # In FastAPI dependency injection:
    async def my_endpoint(db: AsyncSession = Depends(get_db)):
        ...

    # Standalone usage:
    async with get_db() as session:
        ...
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.config.settings import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database.dsn,
    echo=(settings.log_level == "DEBUG"),
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session, ensuring proper cleanup."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
