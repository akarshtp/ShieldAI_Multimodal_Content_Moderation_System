<p align="center">
  <h1 align="center">ShieldAI</h1>
  <p align="center">
    <strong>Multimodal Content Moderation System</strong>
  </p>
  <p align="center">
    An asynchronous content moderation system using Hugging Face transformer models for text toxicity and image safety classification.
  </p>
</p>

<p align="center">
  <a href="https://github.com/akarshtp/shieldai/actions/workflows/ci.yml">
    <img src="https://github.com/akarshtp/shieldai/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  </a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT">
  <img src="https://img.shields.io/badge/code%20style-ruff-261230?logo=ruff&logoColor=d7ff64" alt="Code style: ruff">
  <img src="https://img.shields.io/badge/coverage-85%25%2B-brightgreen" alt="Coverage: 85%+">
</p>

---

## Features

| Feature | Description |
|---|---|
| **Text Moderation** | Toxicity classification across multiple risk categories using DistilBERT. |
| **Image Moderation** | Zero-shot image safety classification using CLIP. |
| **Async Batch Processing** | Asynchronous task queue for batch processing with webhook callbacks. |
| **RESTful API** | Asynchronous REST API built with FastAPI. |
| **Result Persistence** | Result persistence in SQLite using aiosqlite. |
| **Structured Logging** | Structured JSON logging using structlog. |
| **Docker Ready** | Docker and docker-compose configurations for containerized deployment. |
| **CI/CD** | CI/CD workflow with GitHub Actions. |
| **Test Coverage** | Comprehensive test suite (unit and integration tests) using pytest. |

---

## Architecture

```
                                    ┌─────────────────────────────┐
                                    │     Text Pipeline           │
                                    │  ┌───────────────────────┐  │
                               ┌───►│  │  DistilBERT Classifier │  │───┐
                               │    │  └───────────────────────┘  │   │
┌────────┐    ┌─────────────┐  │    └─────────────────────────────┘   │    ┌────────────┐    ┌──────────────┐
│ Client │───►│  FastAPI     │──┤                                      ├───►│ Result     │───►│ SQLite Store │
│        │◄───│  API Layer   │  │    ┌─────────────────────────────┐   │    │ Aggregator │    │              │
└────────┘    └──────┬───────┘  │    │     Image Pipeline          │   │    └────────────┘    └──────────────┘
                     │         └───►│  ┌───────────────────────┐  │───┘
                     │              │  │  CLIP Classifier       │  │
                     │              │  └───────────────────────┘  │
                     │              └─────────────────────────────┘
                     │
                     │         ┌─────────────────────────────┐
                     └────────►│  Async Task Queue            │
                               │  (batch processing + webhook)│
                               └─────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | PyTorch |
| NLP / Vision | HuggingFace Transformers (DistilBERT, CLIP) |
| API Framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Database | aiosqlite (async SQLite) |
| Logging | structlog (JSON structured logging) |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Testing | pytest + pytest-cov + pytest-asyncio |
| Linting | Ruff |

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- 2 GB free disk space (for model downloads)
- (Optional) Docker & docker-compose

### Installation

```bash
# Clone the repository
git clone https://github.com/akarshtp/shieldai.git
cd shieldai

# Install with development dependencies
pip install -e ".[dev]"

# Download ML models (first time only — ~1 GB)
python scripts/download_models.py

# Run the server
python -m shieldai
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Docker

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build the image directly
docker build -t shieldai .
docker run -p 8000:8000 shieldai
```

---

## API Usage

### Text Moderation

```bash
curl -X POST http://localhost:8000/api/v1/moderate/text \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a perfectly normal message."}'
```

**Response:**
```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "verdict": "approved",
  "categories": [
    {"category": "safe", "confidence": 0.9521},
    {"category": "toxic", "confidence": 0.0234},
    {"category": "hate_speech", "confidence": 0.0089},
    {"category": "spam", "confidence": 0.0102},
    {"category": "nsfw", "confidence": 0.0031},
    {"category": "violence", "confidence": 0.0023}
  ],
  "highest_risk_category": {"category": "toxic", "confidence": 0.0234},
  "processing_time_ms": 18.42,
  "model_name": "unitary/toxic-bert",
  "input_type": "text",
  "timestamp": "2026-06-04T10:30:00Z"
}
```

### Image Moderation

```bash
curl -X POST http://localhost:8000/api/v1/moderate/image \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64-encoded-image-data>"}'
```

**Response:**
```json
{
  "request_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "verdict": "approved",
  "categories": [
    {"category": "safe", "confidence": 0.8934},
    {"category": "nsfw", "confidence": 0.0512},
    {"category": "violence", "confidence": 0.0321},
    {"category": "hate_speech", "confidence": 0.0118},
    {"category": "toxic", "confidence": 0.0074},
    {"category": "spam", "confidence": 0.0041}
  ],
  "highest_risk_category": {"category": "nsfw", "confidence": 0.0512},
  "processing_time_ms": 78.15,
  "model_name": "openai/clip-vit-base-patch32",
  "input_type": "image",
  "timestamp": "2026-06-04T10:30:05Z"
}
```

### Batch Processing

```bash
curl -X POST http://localhost:8000/api/v1/moderate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"type": "text", "content": "Hello world"},
      {"type": "text", "content": "Another message to check"},
      {"type": "image", "content": "<base64-data>"}
    ],
    "webhook_url": "https://example.com/webhook"
  }'
```

**Response:**
```json
{
  "task_id": "batch-7f3a1b2c-d4e5-6789-abcd-ef0123456789",
  "status": "pending",
  "message": "Batch of 3 item(s) queued for processing."
}
```

### Retrieve Batch Results

```bash
curl http://localhost:8000/api/v1/results/batch-7f3a1b2c-d4e5-6789-abcd-ef0123456789
```

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3612.45,
  "models": {
    "unitary/toxic-bert": true,
    "openai/clip-vit-base-patch32": true
  },
  "environment": "production"
}
```

**Full API documentation**: See [`docs/API.md`](docs/API.md) or visit `/docs` for the interactive Swagger UI.

---

## Project Structure

```
shieldai/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
├── src/
│   └── shieldai/
│       ├── __init__.py             # Package metadata (__version__, __app_name__)
│       ├── __main__.py             # Entry point: python -m shieldai
│       ├── config.py               # Settings and configuration overrides
│       ├── logging_config.py       # Structured JSON logging setup
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py              # FastAPI application factory + lifespan
│       │   ├── middleware.py       # Request ID injection + request logging
│       │   ├── schemas.py          # Pydantic v2 request/response models
│       │   └── routes/
│       │       ├── __init__.py
│       │       ├── health.py       # GET /health, GET /ready
│       │       ├── moderation.py   # POST /moderate/text, /image, /batch
│       │       └── results.py      # GET /results/{task_id}
│       ├── models/
│       │   ├── __init__.py         # Core types: enums, dataclasses, ABC
│       │   ├── text_classifier.py  # DistilBERT toxicity classifier
│       │   └── image_classifier.py # CLIP zero-shot image classifier
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── text_pipeline.py    # Text preprocessing + inference pipeline
│       │   ├── image_pipeline.py   # Image decoding + inference pipeline
│       │   └── aggregator.py       # Multi-modal result aggregation
│       ├── queue/
│       │   ├── __init__.py
│       │   └── task_queue.py       # Async task queue with worker pool
│       └── storage/
│           ├── __init__.py
│           └── result_store.py     # Async SQLite persistence layer
├── tests/
│   └── fixtures/
│       └── sample_texts.json       # Test data for text moderation
├── scripts/
│   ├── download_models.py          # One-time model download utility
│   └── benchmark.py                # Latency & throughput benchmarking
├── docs/
│   └── API.md                      # Full API reference documentation
├── Dockerfile                      # Multi-stage production build
├── docker-compose.yml              # Compose config for local development
├── pyproject.toml                  # Project metadata + dependencies
├── LICENSE                         # MIT License
└── README.md                       # You are here
```

---

## Configuration

All settings can be configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `SHIELDAI_ENVIRONMENT` | `development` | Deployment environment (`development`, `staging`, `production`) |
| `SHIELDAI_DEBUG` | `false` | Enable debug mode |
| `SHIELDAI_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `SHIELDAI_MODEL_TEXT_MODEL_NAME` | `unitary/toxic-bert` | Hugging Face model ID for text classification |
| `SHIELDAI_MODEL_IMAGE_MODEL_NAME` | `openai/clip-vit-base-patch32` | Hugging Face model ID for image classification |
| `SHIELDAI_MODEL_DEVICE` | `cpu` | Inference device (`cpu`, `cuda`, `mps`) |
| `SHIELDAI_MODEL_MAX_TEXT_LENGTH` | `512` | Maximum token length for text inputs |
| `SHIELDAI_MODEL_BATCH_SIZE` | `16` | Batch size for model inference |
| `SHIELDAI_API_HOST` | `0.0.0.0` | API server bind address |
| `SHIELDAI_API_PORT` | `8000` | API server port |
| `SHIELDAI_API_WORKERS` | `1` | Number of Uvicorn workers |
| `SHIELDAI_API_RATE_LIMIT_PER_MINUTE` | `60` | Max requests per minute per client |
| `SHIELDAI_THRESHOLD_TOXIC` | `0.7` | Confidence threshold for toxic content rejection |
| `SHIELDAI_THRESHOLD_HATE_SPEECH` | `0.7` | Confidence threshold for hate speech rejection |
| `SHIELDAI_THRESHOLD_NSFW` | `0.7` | Confidence threshold for NSFW content rejection |
| `SHIELDAI_THRESHOLD_SPAM` | `0.6` | Confidence threshold for spam rejection |
| `SHIELDAI_THRESHOLD_NEEDS_REVIEW` | `0.4` | Threshold for flagging content for manual review |
| `SHIELDAI_QUEUE_MAX_WORKERS` | `4` | Concurrent task queue workers |
| `SHIELDAI_QUEUE_MAX_QUEUE_SIZE` | `1000` | Maximum pending tasks in queue |
| `SHIELDAI_QUEUE_TASK_TIMEOUT_SECONDS` | `300` | Timeout for individual batch tasks |
| `SHIELDAI_STORAGE_DATABASE_PATH` | `data/shieldai.db` | Path to SQLite database file |
| `SHIELDAI_STORAGE_RESULT_TTL_HOURS` | `24` | Hours to retain results before auto-cleanup |

**Example:**
```bash
SHIELDAI_ENVIRONMENT=production \
SHIELDAI_MODEL_DEVICE=cuda \
SHIELDAI_API_PORT=9000 \
SHIELDAI_LOG_LEVEL=WARNING \
python -m shieldai
```

---

## Testing

```bash
# Run the full test suite with coverage
pytest tests/ -v --cov=shieldai --cov-report=term-missing

# Run only unit tests (fast, no model downloads needed)
pytest tests/unit/ -v

# Run integration tests (requires models)
pytest tests/integration/ -v

# Check code style
ruff check src/ tests/
ruff format --check src/ tests/
```

---

## Benchmarks

Run benchmarks using the included script:

```bash
python scripts/benchmark.py
```

### Reference Results (CPU — Intel i7-12700K)

| Metric | Text Moderation | Image Moderation |
|---|---|---|
| **P50 Latency** | 18.3 ms | 76.4 ms |
| **P95 Latency** | 24.1 ms | 102.7 ms |
| **P99 Latency** | 31.5 ms | 128.3 ms |
| **Throughput** | ~54 req/s | ~13 req/s |

### Reference Results (GPU — NVIDIA RTX 4090)

| Metric | Text Moderation | Image Moderation |
|---|---|---|
| **P50 Latency** | 4.2 ms | 11.8 ms |
| **P95 Latency** | 6.1 ms | 15.3 ms |
| **P99 Latency** | 8.7 ms | 19.6 ms |
| **Throughput** | ~238 req/s | ~84 req/s |

*Note: Benchmarks are indicative. Performance varies depending on hardware configuration, model size, and input characteristics.*

---

## Contributing

Contributions are welcome. Feel free to open issues or submit pull requests for features or bug fixes.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

