"""FastAPI application factory with lifespan management.

Creates and configures the FastAPI application, loads ML models on startup,
and registers all routes and middleware.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shieldai import __app_name__, __version__
from shieldai.config import get_settings
from shieldai.logging_config import get_logger, setup_logging
from shieldai.models.image_classifier import ImageClassifier
from shieldai.models.text_classifier import TextClassifier
from shieldai.pipeline.image_pipeline import ImagePipeline
from shieldai.pipeline.text_pipeline import TextPipeline
from shieldai.queue.task_queue import AsyncTaskQueue
from shieldai.storage.result_store import ResultStore

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    On startup:
        - Load ML models into memory.
        - Initialise the result store and task queue.

    On shutdown:
        - Stop the task queue workers.
        - Close the database connection.
    """
    settings = get_settings()
    setup_logging(
        log_level=settings.log_level,
        json_output=settings.environment != "development",
    )

    logger.info(
        "starting_application",
        app_name=settings.app_name,
        version=settings.version,
        environment=settings.environment,
    )

    # Record startup time for uptime tracking
    app.state.start_time = time.time()

    # ── Load ML models ────────────────────────────────────────────────────
    logger.info("loading_text_model", model=settings.model.text_model_name)
    text_classifier = TextClassifier()
    try:
        text_classifier.load_model()
        logger.info("text_model_loaded")
    except Exception:
        logger.exception("text_model_load_failed")

    logger.info("loading_image_model", model=settings.model.image_model_name)
    image_classifier = ImageClassifier()
    try:
        image_classifier.load_model()
        logger.info("image_model_loaded")
    except Exception:
        logger.exception("image_model_load_failed")

    # ── Build pipelines ───────────────────────────────────────────────────
    app.state.text_pipeline = TextPipeline(classifier=text_classifier)
    app.state.image_pipeline = ImagePipeline(classifier=image_classifier)

    # ── Initialise storage ────────────────────────────────────────────────
    result_store = ResultStore()
    await result_store.initialize()
    app.state.result_store = result_store

    # ── Start task queue ──────────────────────────────────────────────────
    task_queue = AsyncTaskQueue(
        max_workers=settings.queue.max_workers,
        max_queue_size=settings.queue.max_queue_size,
    )

    async def _process_item(item: dict) -> dict:
        """Process a single batch item through the appropriate pipeline."""
        if item["type"] == "text":
            result = app.state.text_pipeline.moderate(item["content"])
        else:
            result = app.state.image_pipeline.moderate(item["content"])
        return {
            "verdict": result.verdict.value,
            "scores": [
                {"category": s.category.value, "confidence": s.confidence}
                for s in result.scores
            ],
            "processing_time_ms": result.processing_time_ms,
            "model_name": result.model_name,
            "input_type": result.input_type,
        }

    await task_queue.start(process_fn=_process_item)
    app.state.task_queue = task_queue

    logger.info("application_ready")

    yield  # ── Application runs here ──

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("shutting_down")
    await task_queue.stop()
    await result_store.close()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured ``FastAPI`` instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=__app_name__,
        description=(
            "Production-grade Multimodal Content Moderation API. "
            "Classifies text and images for toxicity, hate speech, spam, "
            "and NSFW content using transformer models."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/api/v1/openapi.json",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Custom middleware ─────────────────────────────────────────────────
    from shieldai.api.middleware import RequestIDMiddleware, RequestLoggingMiddleware

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # ── Routes ────────────────────────────────────────────────────────────
    from shieldai.api.routes.health import router as health_router
    from shieldai.api.routes.moderation import router as moderation_router
    from shieldai.api.routes.results import router as results_router

    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(moderation_router, prefix="/api/v1", tags=["Moderation"])
    app.include_router(results_router, prefix="/api/v1", tags=["Results"])

    return app
