"""Task result retrieval endpoint for batch moderation jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from shieldai.api.schemas import ErrorResponse, TaskResultResponse
from shieldai.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["results"])


@router.get(
    "/results/{task_id}",
    response_model=TaskResultResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Task not found",
        },
    },
    summary="Get batch task results",
    description="Retrieve the status and results of a previously submitted batch moderation task.",
)
async def get_task_result(task_id: str, request: Request) -> TaskResultResponse:
    """Look up a batch task by its ID and return current status / results.

    Args:
        task_id: The unique task identifier returned by ``POST /moderate/batch``.
        request: The incoming FastAPI request (used to access app state).

    Raises:
        HTTPException 404: If the ``task_id`` does not exist in the result store.
        HTTPException 503: If the result store is not available.
    """
    request_id: str = getattr(request.state, "request_id", "unknown")
    result_store = getattr(request.app.state, "result_store", None)

    if result_store is None:
        logger.error("result_store_unavailable", request_id=request_id)
        raise HTTPException(
            status_code=503,
            detail="Result store is not available.",
        )

    logger.info("result_lookup", request_id=request_id, task_id=task_id)

    task = await result_store.get(task_id)

    if task is None:
        logger.warning(
            "task_not_found",
            request_id=request_id,
            task_id=task_id,
        )
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found.",
        )

    logger.info(
        "result_retrieved",
        request_id=request_id,
        task_id=task_id,
        status=task.status,
    )

    return TaskResultResponse(
        task_id=task.task_id,
        status=task.status,
        results=task.results,
        created_at=task.created_at,
        completed_at=task.completed_at,
    )
