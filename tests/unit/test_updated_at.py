from sqlalchemy import inspect
from src.database.models import Job

def test_updated_at_has_server_onupdate():
    """Verify that the updated_at column has server_onupdate=FetchedValue() set."""
    mapper = inspect(Job)
    updated_at_col = mapper.columns["updated_at"]
    assert updated_at_col.server_onupdate is not None
