"""Content moderation endpoints for text, image, and batch requests."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from shieldai.api.schemas import (
    BatchModerationRequest,
    CategoryScoreResponse,
    ImageModerationRequest,
    ModerationResponse,
    TaskResponse,
    TextModerationRequest,
)
from shieldai.logging_config import get_logger

if TYPE_CHECKING:
    from shieldai.models import ModerationResult

logger = get_logger(__name__)

router = APIRouter(prefix="/moderate", tags=["moderation"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_response(
    result: ModerationResult,
    request_id: str,
) -> ModerationResponse:
    """Convert an internal ``ModerationResult`` dataclass to an API response schema.

    Args:
        result: The moderation result produced by a pipeline.
        request_id: The unique request identifier to embed in the response.

    Returns:
        A fully-populated ``ModerationResponse``.
    """
    categories = [
        CategoryScoreResponse(
            category=score.category.value,
            confidence=round(score.confidence, 4),
        )
        for score in result.scores
    ]

    highest = result.highest_risk_category
    highest_response: CategoryScoreResponse | None = None
    if highest is not None:
        highest_response = CategoryScoreResponse(
            category=highest.category.value,
            confidence=round(highest.confidence, 4),
        )

    return ModerationResponse(
        request_id=request_id,
        verdict=result.verdict.value,
        categories=categories,
        highest_risk_category=highest_response,
        processing_time_ms=round(result.processing_time_ms, 2),
        model_name=result.model_name,
        input_type=result.input_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/text",
    response_model=ModerationResponse,
    summary="Moderate text content",
    description="Classify a single text string and return category scores with a verdict.",
)
async def moderate_text(
    body: TextModerationRequest,
    request: Request,
) -> ModerationResponse:
    """Run text moderation on the supplied content.

    Raises:
        HTTPException 422: If the request body fails validation.
        HTTPException 503: If the text pipeline is not available.
    """
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))
    text_pipeline = getattr(request.app.state, "text_pipeline", None)

    if text_pipeline is None or not text_pipeline.is_loaded():
        logger.error("text_pipeline_unavailable", request_id=request_id)
        raise HTTPException(
            status_code=503,
            detail="Text moderation pipeline is not available.",
        )

    logger.info(
        "moderate_text_start",
        request_id=request_id,
        text_length=len(body.text),
    )

    try:
        result: ModerationResult = text_pipeline.moderate(body.text)
    except Exception as exc:
        logger.exception("moderate_text_error", request_id=request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Text moderation failed: {exc}",
        ) from exc

    logger.info(
        "moderate_text_complete",
        request_id=request_id,
        verdict=result.verdict.value,
        processing_time_ms=round(result.processing_time_ms, 2),
    )

    return _result_to_response(result, request_id)


@router.post(
    "/image",
    response_model=ModerationResponse,
    summary="Moderate image content",
    description="Classify a base64-encoded image and return category scores with a verdict.",
)
async def moderate_image(
    body: ImageModerationRequest,
    request: Request,
) -> ModerationResponse:
    """Run image moderation on the supplied base64 data.

    Raises:
        HTTPException 503: If the image pipeline is not available.
    """
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))
    image_pipeline = getattr(request.app.state, "image_pipeline", None)

    if image_pipeline is None or not image_pipeline.is_loaded():
        logger.error("image_pipeline_unavailable", request_id=request_id)
        raise HTTPException(
            status_code=503,
            detail="Image moderation pipeline is not available.",
        )

    logger.info(
        "moderate_image_start",
        request_id=request_id,
        payload_size=len(body.image_base64),
    )

    try:
        result: ModerationResult = image_pipeline.moderate(body.image_base64)
    except Exception as exc:
        logger.exception("moderate_image_error", request_id=request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Image moderation failed: {exc}",
        ) from exc

    logger.info(
        "moderate_image_complete",
        request_id=request_id,
        verdict=result.verdict.value,
        processing_time_ms=round(result.processing_time_ms, 2),
    )

    return _result_to_response(result, request_id)


@router.post(
    "/batch",
    response_model=TaskResponse,
    summary="Submit a batch moderation job",
    description="Queue multiple items for asynchronous moderation and receive a task ID.",
)
async def moderate_batch(
    body: BatchModerationRequest,
    request: Request,
) -> TaskResponse:
    """Submit a batch of items for async moderation.

    The items are placed on the internal task queue. Use ``GET /results/{task_id}``
    to poll for completion.

    Raises:
        HTTPException 503: If the task queue is not available.
    """
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))
    task_queue = getattr(request.app.state, "task_queue", None)

    if task_queue is None:
        logger.error("task_queue_unavailable", request_id=request_id)
        raise HTTPException(
            status_code=503,
            detail="Batch processing queue is not available.",
        )

    task_id = str(uuid.uuid4())

    logger.info(
        "batch_submitted",
        request_id=request_id,
        task_id=task_id,
        item_count=len(body.items),
        webhook_url=body.webhook_url,
    )

    try:
        await task_queue.submit(
            task_id=task_id,
            items=body.items,
        )
    except Exception as exc:
        logger.exception("batch_submit_error", request_id=request_id, task_id=task_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enqueue batch task: {exc}",
        ) from exc

    return TaskResponse(
        task_id=task_id,
        status="pending",
        message=f"Batch of {len(body.items)} item(s) queued for processing.",
    )
