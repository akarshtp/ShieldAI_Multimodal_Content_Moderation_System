"""Async task queue for batch content moderation.

Provides an ``AsyncTaskQueue`` backed by :mod:`asyncio.Queue` that manages
concurrent processing of moderation requests with configurable parallelism.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from shieldai.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from shieldai.models import ModerationResult

logger = get_logger(__name__)


class TaskStatus(str, enum.Enum):
    """Lifecycle status of a queued moderation task."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskInfo:
    """Metadata and results for a single queued task.

    Attributes:
        task_id: Unique identifier for the task.
        status: Current lifecycle status.
        items: List of ``BatchItem`` objects to be processed.
        results: Accumulated moderation results (one per item).
        created_at: Timestamp when the task was submitted.
        completed_at: Timestamp when the task finished (success or failure).
        error: Error message if the task failed.
    """

    task_id: str
    status: TaskStatus
    items: list[Any]
    results: list[ModerationResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error: str | None = None


class AsyncTaskQueue:
    """Asyncio-based task queue for parallel content moderation.

    Usage::

        queue = AsyncTaskQueue(max_workers=4, max_queue_size=1000)
        await queue.start(process_fn=my_moderation_fn)
        task_info = await queue.submit("task-1", items)
        ...
        await queue.stop()

    Args:
        max_workers: Number of concurrent asyncio worker tasks.
        max_queue_size: Upper bound on the internal queue length.  When the
            queue is full, ``submit`` will block until space is available.
    """

    def __init__(self, max_workers: int, max_queue_size: int) -> None:
        self._max_workers = max_workers
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self._tasks: dict[str, TaskInfo] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._process_fn: Callable[..., Any] | None = None
        self._running = False

        logger.info(
            "task_queue_initialized",
            max_workers=max_workers,
            max_queue_size=max_queue_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, task_id: str, items: list[Any]) -> TaskInfo:
        """Submit a new task to the queue.

        Args:
            task_id: Unique identifier for the task.
            items: List of ``BatchItem`` objects to moderate.

        Returns:
            The newly created ``TaskInfo`` with status ``PENDING``.

        Raises:
            RuntimeError: If the queue has not been started yet.
        """
        if not self._running:
            raise RuntimeError("Queue is not running. Call start() first.")

        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            items=items,
        )
        self._tasks[task_id] = task_info

        await self._queue.put(task_id)

        logger.info(
            "task_submitted",
            task_id=task_id,
            item_count=len(items),
            queue_size=self._queue.qsize(),
        )
        return task_info

    async def get_task(self, task_id: str) -> TaskInfo | None:
        """Look up a task by its ID.

        Args:
            task_id: The task identifier to search for.

        Returns:
            The ``TaskInfo`` if found, otherwise ``None``.
        """
        return self._tasks.get(task_id)

    async def start(self, process_fn: Callable[..., Any]) -> None:
        """Start background worker tasks that consume from the queue.

        Args:
            process_fn: An async callable with the signature
                ``async (BatchItem) -> ModerationResult``.  Each worker calls
                this function for every item in a task.
        """
        if self._running:
            logger.warning("task_queue_already_running")
            return

        self._process_fn = process_fn
        self._running = True

        for idx in range(self._max_workers):
            worker = asyncio.create_task(
                self._worker_loop(worker_id=idx),
                name=f"shieldai-worker-{idx}",
            )
            self._workers.append(worker)

        logger.info("task_queue_started", worker_count=self._max_workers)

    async def stop(self) -> None:
        """Gracefully shut down all workers.

        Outstanding items in the queue will **not** be processed after
        shutdown is initiated.
        """
        if not self._running:
            return

        self._running = False

        # Signal each worker to exit by injecting sentinel values.
        for _ in self._workers:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait("")

        # Wait for workers to finish their current work and exit.
        results = await asyncio.gather(*self._workers, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "worker_exit_error",
                    worker_id=idx,
                    error=str(result),
                )

        self._workers.clear()
        logger.info("task_queue_stopped")

    # ------------------------------------------------------------------
    # Internal worker loop
    # ------------------------------------------------------------------

    async def _worker_loop(self, worker_id: int) -> None:
        """Process tasks from the queue until ``stop()`` is called.

        Args:
            worker_id: Numeric identifier for logging.
        """
        logger.debug("worker_started", worker_id=worker_id)

        while self._running:
            try:
                task_id = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                # Re-check the running flag periodically.
                continue

            # Sentinel value signals shutdown.
            if not task_id:
                self._queue.task_done()
                break

            task_info = self._tasks.get(task_id)
            if task_info is None:
                logger.warning("task_not_found", task_id=task_id, worker_id=worker_id)
                self._queue.task_done()
                continue

            await self._process_task(task_info, worker_id)
            self._queue.task_done()

        logger.debug("worker_stopped", worker_id=worker_id)

    async def _process_task(self, task_info: TaskInfo, worker_id: int) -> None:
        """Process all items within a single task.

        Each item is processed independently — a failure in one item does
        not abort the remaining items.

        Args:
            task_info: The task metadata to update in place.
            worker_id: Numeric identifier for logging.
        """
        assert self._process_fn is not None

        task_info.status = TaskStatus.PROCESSING
        logger.info(
            "task_processing_started",
            task_id=task_info.task_id,
            worker_id=worker_id,
            item_count=len(task_info.items),
        )

        errors: list[str] = []

        for idx, item in enumerate(task_info.items):
            try:
                result = await self._process_fn(item)
                task_info.results.append(result)
                logger.debug(
                    "item_processed",
                    task_id=task_info.task_id,
                    item_index=idx,
                    verdict=result.verdict.value if hasattr(result, "verdict") else None,
                )
            except Exception as exc:
                error_msg = f"Item {idx} failed: {exc}"
                errors.append(error_msg)
                logger.error(
                    "item_processing_failed",
                    task_id=task_info.task_id,
                    item_index=idx,
                    error=str(exc),
                    exc_info=True,
                )

        task_info.completed_at = datetime.now(timezone.utc)

        if errors:
            task_info.status = TaskStatus.FAILED
            task_info.error = "; ".join(errors)
            logger.warning(
                "task_completed_with_errors",
                task_id=task_info.task_id,
                worker_id=worker_id,
                total_items=len(task_info.items),
                successful=len(task_info.results),
                failed=len(errors),
            )
        else:
            task_info.status = TaskStatus.COMPLETED
            logger.info(
                "task_completed",
                task_id=task_info.task_id,
                worker_id=worker_id,
                total_items=len(task_info.items),
                successful=len(task_info.results),
            )
