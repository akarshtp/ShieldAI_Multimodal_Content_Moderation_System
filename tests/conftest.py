"""Shared pytest fixtures for the ShieldAI test suite.

All fixtures use mocks so that tests run WITHOUT downloading ML models.
"""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

# Add the 'src' directory to the python path to ensure imports work in all environments
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from shieldai.models import (
    CategoryScore,
    ContentCategory,
    ModerationResult,
    ModerationVerdict,
)

# ---------------------------------------------------------------------------
# Core result fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_safe_result() -> ModerationResult:
    """A ModerationResult that represents safe content."""
    return ModerationResult(
        scores=[
            CategoryScore(category=ContentCategory.SAFE, confidence=0.95),
            CategoryScore(category=ContentCategory.TOXIC, confidence=0.05),
            CategoryScore(category=ContentCategory.HATE_SPEECH, confidence=0.02),
            CategoryScore(category=ContentCategory.SPAM, confidence=0.01),
            CategoryScore(category=ContentCategory.NSFW, confidence=0.01),
        ],
        verdict=ModerationVerdict.APPROVED,
        input_type="text",
        processing_time_ms=15.5,
        model_name="test-model",
    )


@pytest.fixture
def mock_toxic_result() -> ModerationResult:
    """A ModerationResult that represents toxic content."""
    return ModerationResult(
        scores=[
            CategoryScore(category=ContentCategory.SAFE, confidence=0.1),
            CategoryScore(category=ContentCategory.TOXIC, confidence=0.92),
            CategoryScore(category=ContentCategory.HATE_SPEECH, confidence=0.75),
            CategoryScore(category=ContentCategory.SPAM, confidence=0.05),
            CategoryScore(category=ContentCategory.NSFW, confidence=0.03),
        ],
        verdict=ModerationVerdict.REJECTED,
        input_type="text",
        processing_time_ms=18.2,
        model_name="test-model",
    )


# ---------------------------------------------------------------------------
# Mocked classifier fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_text_classifier(mock_safe_result: ModerationResult) -> MagicMock:
    """A mocked TextClassifier that returns safe results."""
    classifier = MagicMock()
    classifier.predict.return_value = mock_safe_result
    classifier.is_loaded.return_value = True
    classifier.model_name = "mock-text-classifier"
    return classifier


@pytest.fixture
def mock_image_classifier(mock_safe_result: ModerationResult) -> MagicMock:
    """A mocked ImageClassifier that returns safe results."""
    safe_image = ModerationResult(
        scores=[
            CategoryScore(category=ContentCategory.SAFE, confidence=0.90),
            CategoryScore(category=ContentCategory.VIOLENCE, confidence=0.05),
            CategoryScore(category=ContentCategory.NSFW, confidence=0.03),
            CategoryScore(category=ContentCategory.HATE_SPEECH, confidence=0.02),
        ],
        verdict=ModerationVerdict.APPROVED,
        input_type="image",
        processing_time_ms=45.0,
        model_name="mock-image-classifier",
    )
    classifier = MagicMock()
    classifier.predict.return_value = safe_image
    classifier.is_loaded.return_value = True
    classifier.model_name = "mock-image-classifier"
    return classifier


# ---------------------------------------------------------------------------
# Pipeline fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def text_pipeline(mock_text_classifier: MagicMock) -> object:
    """A TextPipeline wired to the mocked text classifier."""
    from shieldai.pipeline.text_pipeline import TextPipeline

    return TextPipeline(classifier=mock_text_classifier)


@pytest.fixture
def image_pipeline(mock_image_classifier: MagicMock) -> object:
    """An ImagePipeline wired to the mocked image classifier."""
    from shieldai.pipeline.image_pipeline import ImagePipeline

    return ImagePipeline(classifier=mock_image_classifier)


# ---------------------------------------------------------------------------
# Sample data fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_texts() -> list[dict[str, str]]:
    """Load sample texts from the fixtures directory."""
    import json
    from pathlib import Path

    fixture_path = Path(__file__).parent / "fixtures" / "sample_texts.json"
    with open(fixture_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# FastAPI test-app fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app(mock_text_classifier: MagicMock, mock_image_classifier: MagicMock):
    """Create a FastAPI test app with mocked models.

    The pipelines now expose ``model_name`` and ``is_loaded()`` directly
    (proxied from their underlying classifiers), so no MagicMock wrapping
    is needed.
    """
    from fastapi import FastAPI

    from shieldai.api.routes.health import router as health_router
    from shieldai.api.routes.moderation import router as moderation_router
    from shieldai.api.routes.results import router as results_router
    from shieldai.pipeline.image_pipeline import ImagePipeline
    from shieldai.pipeline.text_pipeline import TextPipeline

    test_app = FastAPI()
    test_app.include_router(health_router, prefix="/api/v1")
    test_app.include_router(moderation_router, prefix="/api/v1")
    test_app.include_router(results_router, prefix="/api/v1")

    test_app.state.start_time = time.time()
    test_app.state.text_pipeline = TextPipeline(classifier=mock_text_classifier)
    test_app.state.image_pipeline = ImagePipeline(classifier=mock_image_classifier)
    test_app.state.result_store = MagicMock()
    test_app.state.task_queue = AsyncMock()

    return test_app


@pytest.fixture
def client(app):
    """A synchronous ``TestClient`` bound to the test FastAPI app."""
    from fastapi.testclient import TestClient

    return TestClient(app)
