"""Unit tests for the Pydantic request / response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shieldai.api.schemas import (
    BatchItem,
    BatchModerationRequest,
    HealthResponse,
    ImageModerationRequest,
    ModerationResponse,
    TextModerationRequest,
)

# ---------------------------------------------------------------------------
# Request schema tests
# ---------------------------------------------------------------------------


class TestTextModerationRequest:
    """Tests for TextModerationRequest validation."""

    def test_text_request_valid(self) -> None:
        """A valid text request should parse without errors."""
        req = TextModerationRequest(text="Hello, this is a test.")
        assert req.text == "Hello, this is a test."

    def test_text_request_empty_text(self) -> None:
        """An empty text field should raise a ValidationError."""
        with pytest.raises(ValidationError):
            TextModerationRequest(text="")


class TestImageModerationRequest:
    """Tests for ImageModerationRequest validation."""

    def test_image_request_valid(self) -> None:
        """A valid image request should parse without errors."""
        req = ImageModerationRequest(image_base64="iVBORw0KGgoAAAANSUhEUg==")
        assert req.image_base64 == "iVBORw0KGgoAAAANSUhEUg=="


class TestBatchModerationRequest:
    """Tests for BatchModerationRequest validation."""

    def test_batch_request_valid(self) -> None:
        """A batch request with text and image items should parse."""
        req = BatchModerationRequest(
            items=[
                BatchItem(type="text", content="Hello world"),
                BatchItem(type="image", content="iVBORw0KGgoAAAANSUhEUg=="),
            ]
        )
        assert len(req.items) == 2
        assert req.items[0].type == "text"
        assert req.items[1].type == "image"

    def test_batch_request_too_many_items(self) -> None:
        """A batch with more than 100 items should raise a ValidationError."""
        items = [BatchItem(type="text", content=f"item-{i}") for i in range(101)]
        with pytest.raises(ValidationError):
            BatchModerationRequest(items=items)


# ---------------------------------------------------------------------------
# Response schema tests
# ---------------------------------------------------------------------------


class TestModerationResponse:
    """Tests for ModerationResponse serialisation."""

    def test_moderation_response_serialization(self) -> None:
        """A ModerationResponse should serialise to JSON and back correctly."""
        resp = ModerationResponse(
            request_id="abc-123",
            verdict="approved",
            categories=[
                {"category": "safe", "confidence": 0.95},
                {"category": "toxic", "confidence": 0.03},
            ],
            highest_risk_category={"category": "toxic", "confidence": 0.03},
            processing_time_ms=42.5,
            model_name="test-model",
            input_type="text",
            timestamp="2026-06-04T10:30:00Z",
        )

        data = resp.model_dump()
        assert data["request_id"] == "abc-123"
        assert data["verdict"] == "approved"
        assert len(data["categories"]) == 2
        assert data["processing_time_ms"] == 42.5

        # Round-trip through JSON
        json_str = resp.model_dump_json()
        restored = ModerationResponse.model_validate_json(json_str)
        assert restored.request_id == resp.request_id
        assert restored.verdict == resp.verdict


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_health_response(self) -> None:
        """HealthResponse should accept all required fields."""
        resp = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.0,
            models={"test-model": True},
            environment="development",
        )
        assert resp.status == "healthy"
        assert resp.version == "1.0.0"
        assert resp.uptime_seconds == 3600.0
        assert resp.models == {"test-model": True}
        assert resp.environment == "development"
