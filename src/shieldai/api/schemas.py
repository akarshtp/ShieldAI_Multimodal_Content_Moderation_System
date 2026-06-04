"""Pydantic v2 request and response schemas for the moderation API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class TextModerationRequest(BaseModel):
    """Request body for single text moderation."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Text content to moderate",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "This is a sample text to moderate."},
            ],
        },
    }


class ImageModerationRequest(BaseModel):
    """Request body for single image moderation."""

    image_base64: str = Field(
        ...,
        description="Base64-encoded image data (JPEG, PNG, or WebP)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAA..."},
            ],
        },
    }


class BatchItem(BaseModel):
    """A single item within a batch moderation request."""

    type: Literal["text", "image"] = Field(
        ...,
        description="The content type — either 'text' or 'image'",
    )
    content: str = Field(
        ...,
        description="Text content or base64-encoded image data",
    )


class BatchModerationRequest(BaseModel):
    """Request body for batch (async) moderation."""

    items: list[BatchItem] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of items to moderate",
    )
    webhook_url: str | None = Field(
        default=None,
        description="Optional URL to POST results to when processing completes",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [
                        {"type": "text", "content": "Hello world"},
                        {"type": "image", "content": "iVBORw0KGgo..."},
                    ],
                    "webhook_url": "https://example.com/webhook",
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class CategoryScoreResponse(BaseModel):
    """Confidence score for a single content category."""

    category: str = Field(..., description="Content category name")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence for this category (0.0–1.0)",
    )


class ModerationResponse(BaseModel):
    """Response returned for a single moderation request."""

    request_id: str = Field(..., description="Unique identifier for this request")
    verdict: str = Field(
        ...,
        description="Moderation verdict: approved, rejected, or needs_review",
    )
    categories: list[CategoryScoreResponse] = Field(
        ...,
        description="Per-category confidence scores",
    )
    highest_risk_category: CategoryScoreResponse | None = Field(
        default=None,
        description="Category with the highest non-safe confidence score",
    )
    processing_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds",
    )
    model_name: str = Field(..., description="Name of the model used")
    input_type: str = Field(
        ...,
        description="Type of input processed: 'text' or 'image'",
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp when the result was produced",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "verdict": "approved",
                    "categories": [
                        {"category": "safe", "confidence": 0.95},
                        {"category": "toxic", "confidence": 0.03},
                    ],
                    "highest_risk_category": {"category": "toxic", "confidence": 0.03},
                    "processing_time_ms": 42.5,
                    "model_name": "unitary/toxic-bert",
                    "input_type": "text",
                    "timestamp": "2026-06-04T10:30:00Z",
                },
            ],
        },
    }


class TaskResponse(BaseModel):
    """Acknowledgement returned when a batch task is submitted."""

    task_id: str = Field(..., description="Unique identifier for the queued task")
    status: str = Field(
        ...,
        description="Task status: pending, processing, completed, or failed",
    )
    message: str = Field(..., description="Human-readable status message")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task_id": "batch-abc123",
                    "status": "pending",
                    "message": "Batch of 5 items queued for processing.",
                },
            ],
        },
    }


class TaskResultResponse(BaseModel):
    """Full result of a previously submitted batch task."""

    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(
        ...,
        description="Task status: pending, processing, completed, or failed",
    )
    results: list[ModerationResponse] | None = Field(
        default=None,
        description="List of moderation results (present when status is 'completed')",
    )
    created_at: str = Field(
        ...,
        description="ISO 8601 timestamp when the task was created",
    )
    completed_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when the task finished (null if still running)",
    )


class HealthResponse(BaseModel):
    """Response for the health-check endpoint."""

    status: str = Field(
        ...,
        description="Overall health status: healthy, degraded, or unhealthy",
    )
    version: str = Field(..., description="Application version string")
    uptime_seconds: float = Field(
        ...,
        description="Seconds since the application started",
    )
    models: dict[str, bool] = Field(
        ...,
        description="Map of model name → whether the model is loaded",
    )
    environment: str = Field(
        ...,
        description="Current deployment environment (development/staging/production)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "1.0.0",
                    "uptime_seconds": 3600.0,
                    "models": {
                        "unitary/toxic-bert": True,
                        "openai/clip-vit-base-patch32": True,
                    },
                    "environment": "production",
                },
            ],
        },
    }


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str = Field(..., description="Short error code or title")
    detail: str = Field(..., description="Human-readable error description")
    request_id: str | None = Field(
        default=None,
        description="Request ID for tracing (if available)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "validation_error",
                    "detail": "Text field is required and must be between 1 and 10 000 characters.",
                    "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                },
            ],
        },
    }
