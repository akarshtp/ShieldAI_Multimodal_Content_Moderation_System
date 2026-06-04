#!/usr/bin/env python3
"""Performance benchmarking script for ShieldAI.

Measures inference latency and throughput for both text and image
moderation pipelines.

Usage:
    python scripts/benchmark.py [--iterations 100] [--warmup 5]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def create_test_image() -> bytes:
    """Create a simple test image for benchmarking."""
    from PIL import Image
    import io

    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def benchmark_text(
    pipeline: object,
    texts: list[str],
    iterations: int,
    warmup: int,
) -> dict[str, float]:
    """Benchmark text moderation latency.

    Args:
        pipeline: The text pipeline instance.
        texts: Sample texts to moderate.
        iterations: Number of measurement iterations.
        warmup: Number of warmup iterations (excluded from stats).

    Returns:
        Dictionary with p50, p95, p99, mean, and throughput metrics.
    """
    latencies: list[float] = []

    for i in range(warmup + iterations):
        text = texts[i % len(texts)]
        start = time.perf_counter()
        pipeline.moderate(text)  # type: ignore[attr-defined]
        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            latencies.append(elapsed_ms)

    latencies.sort()
    return {
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "mean_ms": statistics.mean(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "throughput_rps": 1000.0 / statistics.mean(latencies),
        "iterations": len(latencies),
    }


def benchmark_image(
    pipeline: object,
    image_bytes: bytes,
    iterations: int,
    warmup: int,
) -> dict[str, float]:
    """Benchmark image moderation latency.

    Args:
        pipeline: The image pipeline instance.
        image_bytes: Raw image bytes to moderate.
        iterations: Number of measurement iterations.
        warmup: Number of warmup iterations.

    Returns:
        Dictionary with latency and throughput metrics.
    """
    import base64

    image_b64 = base64.b64encode(image_bytes).decode()
    latencies: list[float] = []

    for i in range(warmup + iterations):
        start = time.perf_counter()
        pipeline.moderate(image_b64)  # type: ignore[attr-defined]
        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            latencies.append(elapsed_ms)

    latencies.sort()
    return {
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "mean_ms": statistics.mean(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "throughput_rps": 1000.0 / statistics.mean(latencies),
        "iterations": len(latencies),
    }


def format_results(name: str, results: dict[str, float]) -> str:
    """Format benchmark results as a readable table."""
    lines = [
        f"\n{'─' * 50}",
        f"  {name}",
        f"{'─' * 50}",
        f"  Iterations:  {results['iterations']:.0f}",
        f"  Mean:        {results['mean_ms']:.2f} ms",
        f"  Std Dev:     {results['stdev_ms']:.2f} ms",
        f"  P50:         {results['p50_ms']:.2f} ms",
        f"  P95:         {results['p95_ms']:.2f} ms",
        f"  P99:         {results['p99_ms']:.2f} ms",
        f"  Throughput:  {results['throughput_rps']:.1f} req/s",
        f"{'─' * 50}",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run benchmarks for text and image moderation."""
    parser = argparse.ArgumentParser(description="ShieldAI Benchmark")
    parser.add_argument("--iterations", type=int, default=50, help="Measurement iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--text-only", action="store_true", help="Benchmark text only")
    parser.add_argument("--image-only", action="store_true", help="Benchmark image only")
    args = parser.parse_args()

    from shieldai.config import get_settings
    from shieldai.models.text_classifier import TextClassifier
    from shieldai.models.image_classifier import ImageClassifier
    from shieldai.pipeline.text_pipeline import TextPipeline
    from shieldai.pipeline.image_pipeline import ImagePipeline

    settings = get_settings()

    print("=" * 50)
    print("  ShieldAI Performance Benchmark")
    print("=" * 50)
    print(f"  Device:      {settings.model.device}")
    print(f"  Iterations:  {args.iterations}")
    print(f"  Warmup:      {args.warmup}")

    sample_texts = [
        "This product is absolutely amazing! I love it.",
        "The worst purchase I've ever made. Complete waste of money.",
        "Neutral review. The product works as described.",
        "I hate everything about this. The company is terrible and should be ashamed.",
        "Great quality, fast shipping, and excellent customer service!",
    ]

    if not args.image_only:
        print("\n  Loading text model...")
        text_classifier = TextClassifier(
            model_name_or_path=settings.model.text_model_name,
            device=settings.model.device,
            max_length=settings.model.max_text_length,
            cache_dir=settings.model.model_cache_dir,
        )
        text_classifier.load_model()
        text_pipeline = TextPipeline(classifier=text_classifier)

        print("  Running text benchmark...")
        text_results = benchmark_text(
            text_pipeline, sample_texts, args.iterations, args.warmup
        )
        print(format_results("Text Moderation", text_results))

    if not args.text_only:
        print("\n  Loading image model...")
        image_classifier = ImageClassifier(
            model_name_or_path=settings.model.image_model_name,
            device=settings.model.device,
            cache_dir=settings.model.model_cache_dir,
        )
        image_classifier.load_model()
        image_pipeline = ImagePipeline(classifier=image_classifier)

        print("  Creating test image...")
        test_image = create_test_image()

        print("  Running image benchmark...")
        image_results = benchmark_image(
            image_pipeline, test_image, args.iterations, args.warmup
        )
        print(format_results("Image Moderation", image_results))

    print("\n  Benchmark complete!\n")


if __name__ == "__main__":
    main()
