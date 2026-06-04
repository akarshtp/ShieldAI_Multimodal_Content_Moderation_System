#!/usr/bin/env python3
"""Download and cache pre-trained models for ShieldAI.

Run this script before first use to download the required transformer models:

    python scripts/download_models.py

Models are cached in ``~/.cache/shieldai/models`` by default.
Override with ``SHIELDAI_MODEL_MODEL_CACHE_DIR`` environment variable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path so we can import shieldai
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    """Download text and image models from HuggingFace Hub."""
    from shieldai.config import get_settings

    settings = get_settings()
    cache_dir = settings.model.model_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ShieldAI — Model Download Script")
    print("=" * 60)
    print(f"\nCache directory: {cache_dir}\n")

    # ── Download text model ───────────────────────────────────────────────
    print(f"[1/2] Downloading text model: {settings.model.text_model_name}")
    print("      This may take a few minutes on first run...")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        AutoTokenizer.from_pretrained(
            settings.model.text_model_name,
            cache_dir=str(cache_dir),
        )
        AutoModelForSequenceClassification.from_pretrained(
            settings.model.text_model_name,
            cache_dir=str(cache_dir),
        )
        print("      ✓ Text model downloaded successfully\n")
    except Exception as e:
        print(f"      ✗ Failed to download text model: {e}\n")
        sys.exit(1)

    # ── Download image model ──────────────────────────────────────────────
    print(f"[2/2] Downloading image model: {settings.model.image_model_name}")
    print("      This may take a few minutes on first run...")
    try:
        from transformers import CLIPModel, CLIPProcessor

        CLIPProcessor.from_pretrained(
            settings.model.image_model_name,
            cache_dir=str(cache_dir),
        )
        CLIPModel.from_pretrained(
            settings.model.image_model_name,
            cache_dir=str(cache_dir),
        )
        print("      ✓ Image model downloaded successfully\n")
    except Exception as e:
        print(f"      ✗ Failed to download image model: {e}\n")
        sys.exit(1)

    print("=" * 60)
    print("All models downloaded successfully!")
    print(f"Cache location: {cache_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
