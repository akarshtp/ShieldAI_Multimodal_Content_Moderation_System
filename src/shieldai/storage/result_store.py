"""Persistent result storage backed by SQLite via aiosqlite.

Provides an async-friendly ``ResultStore`` that persists moderation results
and supports TTL-based cleanup for automatic data retention management.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

from shieldai.config import get_settings
from shieldai.logging_config import get_logger

logger = get_logger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS moderation_results (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    input_type  TEXT NOT NULL,
    result_json TEXT,
    created_at  TEXT NOT NULL,
    completed_at TEXT
);
"""


class ResultStore:
    """Async SQLite store for moderation results.

    Usage::

        store = ResultStore()
        await store.initialize()
        await store.save_result(task_id="abc", status="completed",
                                result_json='{"scores": [...]}', input_type="text")
        result = await store.get_result("abc")
        await store.close()

    The database path is read from the application settings
    (``settings.storage.database_path``).
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._db_path: Path = Path(settings.storage.database_path)
        self._connection: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the database directory, open a connection, and ensure the
        schema exists.

        This method is idempotent — calling it multiple times is safe.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(str(self._db_path))
        # Enable WAL mode for better concurrent read performance.
        await self._connection.execute("PRAGMA journal_mode=WAL;")
        await self._connection.execute(_CREATE_TABLE_SQL)
        await self._connection.commit()

        logger.info("result_store_initialized", database_path=str(self._db_path))

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            logger.info("result_store_closed")

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    async def save_result(
        self,
        task_id: str,
        status: str,
        result_json: str,
        input_type: str,
    ) -> None:
        """Insert or replace a moderation result.

        Args:
            task_id: Unique task identifier (primary key).
            status: Current task status (e.g. ``"completed"``).
            result_json: JSON-serialised moderation result payload.
            input_type: The modality of the input (``"text"`` or ``"image"``).
        """
        self._ensure_connected()
        assert self._connection is not None

        now_iso = datetime.now(timezone.utc).isoformat()

        await self._connection.execute(
            """
            INSERT INTO moderation_results (id, status, input_type, result_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status      = excluded.status,
                input_type  = excluded.input_type,
                result_json = excluded.result_json
            """,
            (task_id, status, input_type, result_json, now_iso),
        )
        await self._connection.commit()

        logger.debug(
            "result_saved",
            task_id=task_id,
            status=status,
            input_type=input_type,
        )

    async def get_result(self, task_id: str) -> dict[str, str | None] | None:
        """Fetch a moderation result by task ID.

        Args:
            task_id: The task identifier to look up.

        Returns:
            A dictionary with the row columns, or ``None`` if not found.
        """
        self._ensure_connected()
        assert self._connection is not None

        cursor = await self._connection.execute(
            "SELECT id, status, input_type, result_json, created_at, completed_at "
            "FROM moderation_results WHERE id = ?",
            (task_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        return {
            "id": row[0],
            "status": row[1],
            "input_type": row[2],
            "result_json": row[3],
            "created_at": row[4],
            "completed_at": row[5],
        }

    async def update_status(
        self,
        task_id: str,
        status: str,
        result_json: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        """Update the status (and optionally the result payload) of a task.

        Args:
            task_id: The task identifier to update.
            status: New status value.
            result_json: Updated JSON result payload, if available.
            completed_at: ISO-8601 completion timestamp, if applicable.
        """
        self._ensure_connected()
        assert self._connection is not None

        await self._connection.execute(
            """
            UPDATE moderation_results
            SET status       = ?,
                result_json  = COALESCE(?, result_json),
                completed_at = COALESCE(?, completed_at)
            WHERE id = ?
            """,
            (status, result_json, completed_at, task_id),
        )
        await self._connection.commit()

        logger.debug(
            "result_status_updated",
            task_id=task_id,
            status=status,
        )

    async def cleanup_old_results(self, ttl_hours: int) -> None:
        """Delete results older than the specified TTL.

        Args:
            ttl_hours: Number of hours after which results are considered
                stale and eligible for deletion.
        """
        self._ensure_connected()
        assert self._connection is not None

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=ttl_hours)).isoformat()

        cursor = await self._connection.execute(
            "DELETE FROM moderation_results WHERE created_at < ?",
            (cutoff,),
        )
        await self._connection.commit()

        deleted = cursor.rowcount
        logger.info(
            "old_results_cleaned_up",
            ttl_hours=ttl_hours,
            cutoff=cutoff,
            deleted_count=deleted,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Raise if the store has not been initialised."""
        if self._connection is None:
            raise RuntimeError("ResultStore is not initialised. Call initialize() first.")
