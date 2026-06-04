"""Integration tests for the moderation API endpoints."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_b64_image(width: int = 100, height: int = 100, fmt: str = "PNG") -> str:
    """Create a small test image and return its base64-encoded string."""
    img = Image.new("RGB", (width, height), color="blue")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Text moderation tests
# ---------------------------------------------------------------------------


class TestModerateText:
    """Tests for POST /api/v1/moderate/text."""

    def test_moderate_text_success(self, client: TestClient) -> None:
        """A valid text payload should return 200 with a ModerationResponse shape."""
        resp = client.post(
            "/api/v1/moderate/text",
            json={"text": "This is perfectly normal text for moderation."},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert "request_id" in data
        assert "verdict" in data
        assert data["verdict"] in {"approved", "rejected", "needs_review"}
        assert "categories" in data
        assert isinstance(data["categories"], list)
        assert "processing_time_ms" in data
        assert "model_name" in data
        assert "input_type" in data
        assert data["input_type"] == "text"

    def test_moderate_text_empty(self, client: TestClient) -> None:
        """An empty text field should return a 422 validation error."""
        resp = client.post("/api/v1/moderate/text", json={"text": ""})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Image moderation tests
# ---------------------------------------------------------------------------


class TestModerateImage:
    """Tests for POST /api/v1/moderate/image."""

    def test_moderate_image_success(self, client: TestClient) -> None:
        """A valid base64 image should return 200 with a ModerationResponse shape."""
        b64 = _make_b64_image()
        resp = client.post(
            "/api/v1/moderate/image",
            json={"image_base64": b64},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert "request_id" in data
        assert "verdict" in data
        assert data["verdict"] in {"approved", "rejected", "needs_review"}
        assert "categories" in data
        assert data["input_type"] == "image"


# ---------------------------------------------------------------------------
# Batch moderation tests
# ---------------------------------------------------------------------------


class TestModerateBatch:
    """Tests for POST /api/v1/moderate/batch."""

    def test_moderate_batch(self, client: TestClient) -> None:
        """A batch request should return 200 with a task_id."""
        b64 = _make_b64_image(50, 50)
        resp = client.post(
            "/api/v1/moderate/batch",
            json={
                "items": [
                    {"type": "text", "content": "Hello world"},
                    {"type": "image", "content": b64},
                ],
            },
        )
        assert resp.status_code == 200

        data = resp.json()
        assert "task_id" in data
        assert "status" in data
        assert data["status"] == "pending"
        assert "message" in data
