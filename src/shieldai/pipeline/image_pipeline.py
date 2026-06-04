"""Image moderation pipeline — loads, validates, preprocesses, and classifies images."""

from __future__ import annotations

import base64
import io
import time
from typing import TYPE_CHECKING

from PIL import Image, UnidentifiedImageError

from shieldai.logging_config import get_logger
from shieldai.models import ModerationResult, ModerationVerdict

if TYPE_CHECKING:
    from shieldai.models.image_classifier import ImageClassifier

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_MIN_DIMENSION = 10
_MAX_DIMENSION = 4096
_MAX_RESIZE_SIDE = 1024

_ALLOWED_FORMATS: frozenset[str] = frozenset(
    {
        "JPEG",
        "PNG",
        "GIF",
        "BMP",
        "WEBP",
        "TIFF",
    }
)


class ImagePipeline:
    """Orchestrates the full image-moderation flow.

    The pipeline performs four sequential steps:
    1. **Load** — decode a base64 string or raw bytes into a PIL Image.
    2. **Validate** — ensure the format is supported and dimensions are within
       acceptable bounds.
    3. **Preprocess** — convert RGBA to RGB, resize oversized images while
       preserving aspect ratio.
    4. **Classify** — delegate to the underlying ``ImageClassifier`` for
       inference.

    Args:
        classifier: A loaded ``ImageClassifier`` instance used for prediction.
    """

    def __init__(self, classifier: ImageClassifier) -> None:
        self._classifier = classifier
        logger.info(
            "image_pipeline_initialized",
            classifier=classifier.model_name,
        )

    @property
    def model_name(self) -> str:
        """Return the name of the underlying classifier model."""
        return self._classifier.model_name

    def is_loaded(self) -> bool:
        """Return ``True`` if the underlying classifier model is loaded."""
        return self._classifier.is_loaded()

    # ── Loading helpers ──────────────────────────────────────────────────

    @staticmethod
    def decode_base64(data: str) -> Image.Image:
        """Decode a base64-encoded string into a PIL Image.

        The string may optionally include a ``data:image/…;base64,`` prefix
        (data-URI scheme); the prefix is stripped automatically.

        Args:
            data: A base64-encoded image string.

        Returns:
            A PIL ``Image`` instance.

        Raises:
            ValueError: If the string cannot be decoded or is not a valid
                image.
        """
        # Strip optional data-URI prefix
        if "," in data and data.startswith("data:"):
            data = data.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(data, validate=True)
        except Exception as exc:
            raise ValueError("Invalid base64 encoding") from exc

        try:
            return Image.open(io.BytesIO(raw_bytes))
        except (UnidentifiedImageError, Exception) as exc:
            raise ValueError(f"Cannot open image from decoded bytes: {exc}") from exc

    def load_image(self, source: str | bytes) -> Image.Image:
        """Load an image from a base64 string or raw bytes.

        Args:
            source: Either a base64-encoded string or raw ``bytes`` of an
                image file.

        Returns:
            A PIL ``Image`` instance.

        Raises:
            ValueError: If *source* is neither ``str`` nor ``bytes``, or the
                image data is invalid.
        """
        if isinstance(source, str):
            logger.debug("loading_image_from_base64", length=len(source))
            return self.decode_base64(source)

        if isinstance(source, bytes):
            logger.debug("loading_image_from_bytes", size=len(source))
            try:
                return Image.open(io.BytesIO(source))
            except (UnidentifiedImageError, Exception) as exc:
                raise ValueError(f"Cannot open image from bytes: {exc}") from exc

        raise TypeError(f"Expected str or bytes for image source, got {type(source).__name__}")

    # ── Validation ───────────────────────────────────────────────────────

    @staticmethod
    def validate_image(image: Image.Image) -> tuple[bool, str]:
        """Validate an image for format and dimensions.

        Checks performed:
        * The image format must be in ``_ALLOWED_FORMATS``.
        * Both width and height must be ≥ ``_MIN_DIMENSION``.
        * Both width and height must be ≤ ``_MAX_DIMENSION``.
        * The image data must not be corrupted (``image.verify()``).

        Note:
            ``image.verify()`` invalidates the image object.  The caller
            should reopen the image data after a successful verification if
            further pixel-level operations are required.

        Args:
            image: A PIL ``Image`` to validate.

        Returns:
            A ``(is_valid, error_message)`` tuple.  When valid, the error
            message is an empty string.
        """
        # Check format
        fmt = image.format
        if fmt and fmt.upper() not in _ALLOWED_FORMATS:
            msg = f"Unsupported image format: {fmt}"
            logger.warning("image_validation_failed", reason="unsupported_format", format=fmt)
            return False, msg

        width, height = image.size

        # Minimum dimensions
        if width < _MIN_DIMENSION or height < _MIN_DIMENSION:
            msg = (
                f"Image too small ({width}x{height}); minimum is {_MIN_DIMENSION}x{_MIN_DIMENSION}"
            )
            logger.warning(
                "image_validation_failed",
                reason="too_small",
                width=width,
                height=height,
            )
            return False, msg

        # Maximum dimensions
        if width > _MAX_DIMENSION or height > _MAX_DIMENSION:
            msg = (
                f"Image too large ({width}x{height}); maximum is {_MAX_DIMENSION}x{_MAX_DIMENSION}"
            )
            logger.warning(
                "image_validation_failed",
                reason="too_large",
                width=width,
                height=height,
            )
            return False, msg

        # Corruption check — verify() can raise various exceptions
        try:
            image.verify()
        except Exception as exc:
            msg = f"Image data appears corrupted: {exc}"
            logger.warning("image_validation_failed", reason="corrupted", error=str(exc))
            return False, msg

        return True, ""

    # ── Preprocessing ────────────────────────────────────────────────────

    @staticmethod
    def preprocess(image: Image.Image) -> Image.Image:
        """Prepare an image for the classifier.

        Processing steps:
        1. Convert RGBA (or palette with transparency) to RGB.
        2. If the longest side exceeds ``_MAX_RESIZE_SIDE``, resize the image
           while preserving the aspect ratio.

        Args:
            image: The PIL ``Image`` to preprocess.

        Returns:
            A preprocessed PIL ``Image`` in RGB mode.
        """
        # Convert to RGB if needed
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
            logger.debug("image_converted_to_rgb", original_mode=image.mode)

        # Resize if the longest side is too large
        width, height = image.size
        longest = max(width, height)

        if longest > _MAX_RESIZE_SIDE:
            scale = _MAX_RESIZE_SIDE / longest
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.debug(
                "image_resized",
                original_size=(width, height),
                new_size=(new_width, new_height),
            )

        return image

    # ── Full moderation flow ─────────────────────────────────────────────

    def moderate(self, image_data: str | bytes) -> ModerationResult:
        """Run the full moderation pipeline on *image_data*.

        Steps:
        1. Load the image from base64 or raw bytes.
        2. Validate format, dimensions, and integrity.
        3. Reload and preprocess the image (verify invalidates the object).
        4. Classify via the underlying ``ImageClassifier``.

        If any step fails the error is logged and a ``ModerationResult`` with
        a ``NEEDS_REVIEW`` verdict is returned so that human moderators can
        inspect the content.

        Args:
            image_data: Base64-encoded image string or raw image bytes.

        Returns:
            A ``ModerationResult`` containing per-category scores and a
            verdict.
        """
        start = time.perf_counter()

        try:
            # Step 1 — load
            image = self.load_image(image_data)

            # Step 2 — validate (verify() invalidates the object afterwards)
            is_valid, error_message = self.validate_image(image)
            if not is_valid:
                elapsed_ms = (time.perf_counter() - start) * 1_000
                logger.info(
                    "image_moderation_rejected",
                    reason=error_message,
                    processing_time_ms=round(elapsed_ms, 2),
                )
                return ModerationResult(
                    scores=[],
                    verdict=ModerationVerdict.NEEDS_REVIEW,
                    input_type="image",
                    processing_time_ms=round(elapsed_ms, 2),
                    model_name=self._classifier.model_name,
                    metadata={"error": error_message},
                )

            # Step 3 — reload (verify() consumed the image) and preprocess
            image = self.load_image(image_data)
            image = self.preprocess(image)

            # Step 4 — classify
            logger.debug("image_classification_start", size=image.size)
            result = self._classifier.predict(image)
            elapsed_ms = (time.perf_counter() - start) * 1_000

            logger.info(
                "image_moderation_complete",
                verdict=result.verdict.value,
                processing_time_ms=round(elapsed_ms, 2),
            )
            return result

        except (ValueError, TypeError) as exc:
            elapsed_ms = (time.perf_counter() - start) * 1_000
            logger.warning(
                "image_moderation_input_error",
                error=str(exc),
                processing_time_ms=round(elapsed_ms, 2),
            )
            return ModerationResult(
                scores=[],
                verdict=ModerationVerdict.NEEDS_REVIEW,
                input_type="image",
                processing_time_ms=round(elapsed_ms, 2),
                model_name=self._classifier.model_name,
                metadata={"error": str(exc)},
            )

        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1_000
            logger.exception(
                "image_moderation_error",
                processing_time_ms=round(elapsed_ms, 2),
            )
            return ModerationResult(
                scores=[],
                verdict=ModerationVerdict.NEEDS_REVIEW,
                input_type="image",
                processing_time_ms=round(elapsed_ms, 2),
                model_name=self._classifier.model_name,
                metadata={"error": "Internal processing error"},
            )
