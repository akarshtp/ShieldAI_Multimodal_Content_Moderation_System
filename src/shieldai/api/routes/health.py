"""Health and readiness probe endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response

from shieldai.api.schemas import HealthResponse
from shieldai.config import get_settings
from shieldai.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns application health including model-load status and uptime.",
)
async def health_check(request: Request) -> HealthResponse:
    """Return current health status of the service.

    Reports overall status as:
    * **healthy** — all models loaded successfully.
    * **degraded** — at least one model loaded, but not all.
    * **unhealthy** — no models loaded.
    """
    settings = get_settings()

    # Collect model-load states from app-level pipelines.
    models: dict[str, bool] = {}
    text_pipeline = getattr(request.app.state, "text_pipeline", None)
    image_pipeline = getattr(request.app.state, "image_pipeline", None)

    if text_pipeline is not None:
        models[text_pipeline.model_name] = text_pipeline.is_loaded()
    if image_pipeline is not None:
        models[image_pipeline.model_name] = image_pipeline.is_loaded()

    loaded_count = sum(models.values())
    total_count = len(models)

    if total_count == 0 or loaded_count == 0:
        status = "unhealthy"
    elif loaded_count < total_count:
        status = "degraded"
    else:
        status = "healthy"

    # Calculate uptime from the timestamp stored at startup.
    start_time: float = getattr(request.app.state, "start_time", time.time())
    uptime_seconds = round(time.time() - start_time, 2)

    logger.debug("health_check", status=status, models=models)

    return HealthResponse(
        status=status,
        version=settings.version,
        uptime_seconds=uptime_seconds,
        models=models,
        environment=settings.environment,
    )


@router.get(
    "/ready",
    summary="Readiness probe",
    description="Returns 200 when all models are loaded; 503 otherwise.",
)
async def readiness_probe(request: Request, response: Response) -> dict[str, str]:
    """Return 200 if all models are loaded and ready, 503 otherwise.

    Designed for Kubernetes / load-balancer readiness probes.
    """
    text_pipeline = getattr(request.app.state, "text_pipeline", None)
    image_pipeline = getattr(request.app.state, "image_pipeline", None)

    all_ready = True

    if text_pipeline is not None and not text_pipeline.is_loaded():
        all_ready = False
    if image_pipeline is not None and not image_pipeline.is_loaded():
        all_ready = False
    # If no pipelines are registered at all, consider the service not ready.
    if text_pipeline is None and image_pipeline is None:
        all_ready = False

    if not all_ready:
        logger.warning("readiness_probe_failed")
        response.status_code = 503
        return {"status": "not_ready"}

    logger.debug("readiness_probe_passed")
    return {"status": "ready"}
