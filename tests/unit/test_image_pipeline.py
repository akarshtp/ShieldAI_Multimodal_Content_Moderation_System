"""Unit tests for the ImagePipeline decoding, validation, preprocessing, and moderation."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from shieldai.models import ModerationResult, ModerationVerdict
from shieldai.pipeline.image_pipeline import ImagePipeline

if TYPE_CHECKING:
    from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_image(
    width: int = 100,
    height: int = 100,
    mode: str = "RGB",
    fmt: str = "PNG",
) -> bytes:
    """Create a small in-memory image and return its raw bytes."""
    img = Image.new(mode, (width, height), color="red")
    buf = io.BytesIO()
    if mode == "RGBA" and fmt == "JPEG":
        # JPEG doesn't support RGBA — convert first
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return buf.getvalue()


def _encode_image_base64(
    width: int = 100,
    height: int = 100,
    mode: str = "RGB",
    fmt: str = "PNG",
) -> str:
    """Return a base64-encoded string of a test image."""
    raw = _make_test_image(width, height, mode, fmt)
    return base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Base64 decoding tests
# ---------------------------------------------------------------------------


class TestDecodeBase64:
    """Tests for ImagePipeline.decode_base64()."""

    def test_decode_base64_valid(self) -> None:
        """A valid base64-encoded PNG should decode to a PIL Image."""
        b64 = _encode_image_base64(50, 50)
        img = ImagePipeline.decode_base64(b64)
        assert isinstance(img, Image.Image)
        assert img.size == (50, 50)

    def test_decode_base64_with_data_uri(self) -> None:
        """A data-URI prefixed string should be handled correctly."""
        b64 = _encode_image_base64(30, 30, fmt="JPEG")
        data_uri = f"data:image/jpeg;base64,{b64}"
        img = ImagePipeline.decode_base64(data_uri)
        assert isinstance(img, Image.Image)
        assert img.size == (30, 30)

    def test_decode_base64_invalid(self) -> None:
        """Invalid base64 data should raise ValueError."""
        with pytest.raises(ValueError, match=r"[Ii]nvalid"):
            ImagePipeline.decode_base64("not-valid-base64!!!")


# ---------------------------------------------------------------------------
# Image validation tests
# ---------------------------------------------------------------------------


class TestValidateImage:
    """Tests for ImagePipeline.validate_image()."""

    def test_validate_valid_image(self) -> None:
        """A 100x100 RGB image should pass validation."""
        raw = _make_test_image(100, 100)
        img = Image.open(io.BytesIO(raw))
        is_valid, msg = ImagePipeline.validate_image(img)
        assert is_valid is True
        assert msg == ""

    def test_validate_too_small(self) -> None:
        """A 5x5 image should fail the minimum-dimension check."""
        raw = _make_test_image(5, 5)
        img = Image.open(io.BytesIO(raw))
        is_valid, msg = ImagePipeline.validate_image(img)
        assert is_valid is False
        assert "small" in msg.lower()

    def test_validate_too_large(self) -> None:
        """A 5000x5000 image should fail the maximum-dimension check."""
        raw = _make_test_image(5000, 5000)
        img = Image.open(io.BytesIO(raw))
        is_valid, msg = ImagePipeline.validate_image(img)
        assert is_valid is False
        assert "large" in msg.lower()


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestPreprocess:
    """Tests for ImagePipeline.preprocess()."""

    def test_preprocess_rgba_to_rgb(self) -> None:
        """An RGBA image should be converted to RGB mode."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        processed = ImagePipeline.preprocess(img)
        assert processed.mode == "RGB"

    def test_preprocess_resize_large(self) -> None:
        """An image larger than 1024 on its longest side should be resized."""
        img = Image.new("RGB", (2048, 1024))
        processed = ImagePipeline.preprocess(img)

        # Longest side should now be 1024
        assert max(processed.size) == 1024
        # Aspect ratio should be preserved (2:1 -> 1024x512)
        assert processed.size == (1024, 512)


# ---------------------------------------------------------------------------
# Full moderation flow test
# ---------------------------------------------------------------------------


class TestModerate:
    """Tests for ImagePipeline.moderate()."""

    def test_moderate_returns_result(
        self, image_pipeline: ImagePipeline, mock_image_classifier: MagicMock
    ) -> None:
        """The full pipeline should return a ModerationResult with APPROVED verdict."""
        b64 = _encode_image_base64(100, 100)
        result = image_pipeline.moderate(b64)
        assert isinstance(result, ModerationResult)
        assert result.verdict == ModerationVerdict.APPROVED
