"""Image safety classifier using OpenAI CLIP (``openai/clip-vit-base-patch32``).

Performs zero-shot classification by computing cosine similarity between the
input image and a set of descriptive text prompts, then maps the resulting
probabilities to the unified :class:`~shieldai.models.ContentCategory` enum.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from shieldai.config import get_settings
from shieldai.logging_config import get_logger
from shieldai.models import (
    BaseClassifier,
    CategoryScore,
    ContentCategory,
    ModerationResult,
    ModerationVerdict,
)

logger = get_logger(__name__)

# Zero-shot text prompts and their corresponding categories.  The softmax
# over cosine similarities produces one probability per prompt.
_PROMPTS: list[tuple[str, ContentCategory]] = [
    ("a safe, family-friendly image", ContentCategory.SAFE),
    ("a violent or graphic image", ContentCategory.VIOLENCE),
    ("an explicit or sexual image", ContentCategory.NSFW),
    ("an image containing hate symbols or offensive content", ContentCategory.HATE_SPEECH),
]


class ImageClassifier(BaseClassifier):
    """Zero-shot image safety classifier built on CLIP.

    The classifier compares the input image against a fixed set of text
    prompts describing safe and unsafe categories, producing probability
    scores via softmax-normalised cosine similarity.

    Example::

        classifier = ImageClassifier()
        classifier.load_model()
        from PIL import Image
        img = Image.open("photo.jpg")
        result = classifier.predict(img)
        print(result.verdict, result.scores)
    """

    def __init__(self) -> None:
        self._processor: Any = None
        self._model: Any = None
        self._device: str = get_settings().model.device
        # Pre-extract prompt texts for the processor.
        self._prompt_texts: list[str] = [p[0] for p in _PROMPTS]

    # ------------------------------------------------------------------
    # BaseClassifier interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Download (if necessary) and load the CLIP model and processor.

        Uses the model cache directory and device configured in
        :func:`shieldai.config.get_settings`.
        """
        settings = get_settings()
        model_id = settings.model.image_model_name
        cache_dir = settings.model.model_cache_dir

        logger.info(
            "loading_image_model",
            model_id=model_id,
            device=self._device,
            cache_dir=str(cache_dir),
        )

        try:
            self._processor = CLIPProcessor.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
            self._model = CLIPModel.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
            self._model.to(self._device)  # type: ignore[union-attr]
            self._model.eval()  # type: ignore[union-attr]
            logger.info("image_model_loaded", model_id=model_id)
        except Exception:
            logger.exception("image_model_load_failed", model_id=model_id)
            raise

    def predict(self, input_data: Any) -> ModerationResult:
        """Classify an image and return a :class:`ModerationResult`.

        Args:
            input_data: A :class:`PIL.Image.Image` to classify.

        Returns:
            A :class:`ModerationResult` containing per-category confidence
            scores and the overall moderation verdict.

        Raises:
            RuntimeError: If the model has not been loaded yet.
            TypeError: If *input_data* is not a PIL Image.
        """
        if not self.is_loaded():
            raise RuntimeError("ImageClassifier model is not loaded. Call load_model() first.")

        if not isinstance(input_data, Image.Image):
            raise TypeError(
                f"ImageClassifier expects a PIL.Image.Image, got {type(input_data).__name__}"
            )

        image: Image.Image = self._prepare_image(input_data)
        settings = get_settings()
        start = time.perf_counter()

        try:
            scores = self._run_inference(image)
            verdict = self._determine_verdict(scores, settings)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            result = ModerationResult(
                scores=scores,
                verdict=verdict,
                input_type="image",
                processing_time_ms=round(elapsed_ms, 2),
                model_name=self.model_name,
                metadata={
                    "image_width": image.width,
                    "image_height": image.height,
                    "image_mode": image.mode,
                },
            )

            logger.info(
                "image_prediction_complete",
                verdict=verdict.value,
                processing_time_ms=result.processing_time_ms,
                image_size=f"{image.width}x{image.height}",
            )
            return result

        except Exception:
            logger.exception(
                "image_prediction_failed",
                image_size=f"{image.width}x{image.height}",
            )
            raise

    def is_loaded(self) -> bool:
        """Return ``True`` if both processor and model are ready."""
        return self._processor is not None and self._model is not None

    @property
    def model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return get_settings().model.image_model_name

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        """Normalise the image to RGB mode.

        RGBA, LA, P, and other non-RGB modes are converted so the CLIP
        processor always receives a consistent 3-channel input.

        Args:
            image: The input PIL image.

        Returns:
            An RGB :class:`PIL.Image.Image`.
        """
        if image.mode == "RGBA":
            # Composite onto a white background to avoid black artefacts.
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # alpha channel
            return background
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def _run_inference(self, image: Image.Image) -> list[CategoryScore]:
        """Compute zero-shot similarity scores for *image*.

        The CLIP model embeds the image and each text prompt independently.
        Cosine similarities are normalised via softmax to produce a valid
        probability distribution over the prompt categories.
        """
        assert self._processor is not None
        assert self._model is not None

        inputs = self._processor(
            text=self._prompt_texts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # outputs.logits_per_image: shape (1, num_prompts)
        logits_per_image = outputs.logits_per_image.squeeze(0)
        probabilities = torch.softmax(logits_per_image, dim=0).cpu().tolist()

        return [
            CategoryScore(
                category=_PROMPTS[i][1],
                confidence=round(prob, 6),
            )
            for i, prob in enumerate(probabilities)
        ]

    @staticmethod
    def _determine_verdict(
        scores: list[CategoryScore],
        settings: Any,
    ) -> ModerationVerdict:
        """Derive a verdict from *scores* using configured thresholds.

        Decision logic (evaluated in order):

        1. **REJECTED** - any unsafe category score ≥ its rejection threshold.
        2. **NEEDS_REVIEW** - any unsafe category score ≥ the review threshold.
        3. **APPROVED** - otherwise.
        """
        thresholds = settings.thresholds
        category_threshold_map: dict[ContentCategory, float] = {
            ContentCategory.VIOLENCE: thresholds.toxic,
            ContentCategory.NSFW: thresholds.nsfw,
            ContentCategory.HATE_SPEECH: thresholds.hate_speech,
        }

        for score in scores:
            if score.category == ContentCategory.SAFE:
                continue
            reject_threshold = category_threshold_map.get(score.category, thresholds.toxic)
            if score.confidence >= reject_threshold:
                return ModerationVerdict.REJECTED

        for score in scores:
            if score.category == ContentCategory.SAFE:
                continue
            if score.confidence >= thresholds.needs_review:
                return ModerationVerdict.NEEDS_REVIEW

        return ModerationVerdict.APPROVED
