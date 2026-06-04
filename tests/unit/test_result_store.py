"""Unit tests for the async ResultStore backed by SQLite."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from shieldai.storage.result_store import ResultStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def _mock_settings(tmp_path: Path):
    """Patch get_settings so ResultStore writes to a temp directory."""
    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.storage.database_path = str(tmp_path / "test.db")

    with patch("shieldai.storage.result_store.get_settings", return_value=mock_settings):
        yield tmp_path


@pytest.fixture
async def store(_mock_settings: Path) -> ResultStore:
    """Provide an initialised ResultStore bound to a temp database."""
    s = ResultStore()
    await s.initialize()
    yield s
    await s.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_creates_database(_mock_settings: Path) -> None:
    """Initialisation should create the SQLite database file on disk."""
    s = ResultStore()
    await s.initialize()

    db_path = _mock_settings / "test.db"
    assert db_path.exists()

    await s.close()


@pytest.mark.asyncio
async def test_save_and_get_result(store: ResultStore) -> None:
    """A saved result should be retrievable by its task ID."""
    payload = json.dumps({"verdict": "approved", "scores": []})
    await store.save_result(
        task_id="task-001",
        status="completed",
        result_json=payload,
        input_type="text",
    )

    row = await store.get_result("task-001")
    assert row is not None
    assert row["id"] == "task-001"
    assert row["status"] == "completed"
    assert row["result_json"] == payload
    assert row["input_type"] == "text"


@pytest.mark.asyncio
async def test_get_nonexistent_result(store: ResultStore) -> None:
    """Looking up a non-existent task ID should return None."""
    result = await store.get_result("does-not-exist")
    assert result is None


@pytest.mark.asyncio
async def test_update_status(store: ResultStore) -> None:
    """Updating the status of an existing task should persist the change."""
    await store.save_result(
        task_id="task-002",
        status="pending",
        result_json="{}",
        input_type="image",
    )

    await store.update_status("task-002", status="completed")

    row = await store.get_result("task-002")
    assert row is not None
    assert row["status"] == "completed"


@pytest.mark.asyncio
async def test_cleanup_old_results(store: ResultStore) -> None:
    """Results older than the TTL should be removed by cleanup."""
    # Insert a result with a very old created_at timestamp
    assert store._connection is not None  # noqa: SLF001
    old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    await store._connection.execute(  # noqa: SLF001
        """
        INSERT INTO moderation_results (id, status, input_type, result_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("old-task", "completed", "text", "{}", old_time),
    )
    await store._connection.commit()  # noqa: SLF001

    # Verify it exists
    row = await store.get_result("old-task")
    assert row is not None

    # Cleanup with a 24-hour TTL — the 48-hour-old result should be deleted
    await store.cleanup_old_results(ttl_hours=24)

    row = await store.get_result("old-task")
    assert row is None
