# ShieldAI API Reference

> **Base URL:** `http://localhost:8000/api/v1`
>
> **Interactive Docs:** [`/docs`](http://localhost:8000/docs) (Swagger UI) | [`/redoc`](http://localhost:8000/redoc) (ReDoc)
>
> **OpenAPI Schema:** [`/api/v1/openapi.json`](http://localhost:8000/api/v1/openapi.json)

---

## Table of Contents

- [Authentication](#authentication)
- [Common Headers](#common-headers)
- [Endpoints](#endpoints)
  - [Text Moderation](#post-apiv1moderatetext)
  - [Image Moderation](#post-apiv1moderateimage)
  - [Batch Moderation](#post-apiv1moderatebatch)
  - [Batch Results](#get-apiv1resultstask_id)
  - [Health Check](#get-apiv1health)
  - [Readiness Probe](#get-apiv1ready)
- [Schemas](#schemas)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

---

## Authentication

ShieldAI currently operates without authentication for local/development deployments. For production use, place the API behind a reverse proxy (e.g., nginx, Traefik) with your preferred auth mechanism (API keys, OAuth2, JWT).

---

## Common Headers

| Header | Required | Description |
|---|---|---|
| `Content-Type` | Yes | Must be `application/json` for all POST requests |
| `X-Request-ID` | No | Custom request ID for tracing. Auto-generated if not provided |

All responses include:

| Header | Description |
|---|---|
| `X-Request-ID` | Unique request identifier (echoed or generated) |

---

## Endpoints

### `POST /api/v1/moderate/text`

Classify a single text string for toxicity, hate speech, spam, NSFW, and violence content.

#### Request Body

```json
{
  "text": "string (1–10,000 characters, required)"
}
```

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `text` | `string` | ✅ | 1–10,000 chars | Text content to moderate |

#### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/moderate/text \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a perfectly normal message."}'
```

#### Example Response — `200 OK`

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

---

### `POST /api/v1/moderate/image`

Classify a base64-encoded image (JPEG, PNG, or WebP) for safety violations.

#### Request Body

```json
{
  "image_base64": "string (base64-encoded image data, required)"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `image_base64` | `string` | ✅ | Base64-encoded image data (JPEG, PNG, or WebP) |

#### Example Request

```bash
# Encode an image and send it
IMAGE_B64=$(base64 -w0 photo.jpg)
curl -X POST http://localhost:8000/api/v1/moderate/image \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_B64\"}"
```

#### Example Response — `200 OK`

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

---

### `POST /api/v1/moderate/batch`

Submit multiple items (text and/or images) for asynchronous moderation. Returns a task ID for polling.

#### Request Body

```json
{
  "items": [
    {"type": "text", "content": "string"},
    {"type": "image", "content": "base64-string"}
  ],
  "webhook_url": "string | null"
}
```

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `items` | `array[BatchItem]` | ✅ | 1–100 items | List of items to moderate |
| `items[].type` | `"text" \| "image"` | ✅ | — | Content type |
| `items[].content` | `string` | ✅ | — | Text content or base64-encoded image |
| `webhook_url` | `string \| null` | ❌ | Valid URL | URL to POST results to when processing completes |

#### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/moderate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"type": "text", "content": "Hello world"},
      {"type": "text", "content": "Check this message"}
    ],
    "webhook_url": "https://example.com/webhook"
  }'
```

#### Example Response — `200 OK`

```json
{
  "task_id": "batch-7f3a1b2c-d4e5-6789-abcd-ef0123456789",
  "status": "pending",
  "message": "Batch of 2 item(s) queued for processing."
}
```

---

### `GET /api/v1/results/{task_id}`

Retrieve the status and results of a previously submitted batch moderation task.

#### Path Parameters

| Parameter | Type | Description |
|---|---|---|
| `task_id` | `string` | The task ID returned by `POST /moderate/batch` |

#### Example Request

```bash
curl http://localhost:8000/api/v1/results/batch-7f3a1b2c-d4e5-6789-abcd-ef0123456789
```

#### Example Response — `200 OK` (completed)

```json
{
  "task_id": "batch-7f3a1b2c-d4e5-6789-abcd-ef0123456789",
  "status": "completed",
  "results": [
    {
      "request_id": "item-001",
      "verdict": "approved",
      "categories": [
        {"category": "safe", "confidence": 0.9612}
      ],
      "highest_risk_category": null,
      "processing_time_ms": 15.23,
      "model_name": "unitary/toxic-bert",
      "input_type": "text",
      "timestamp": "2026-06-04T10:31:00Z"
    }
  ],
  "created_at": "2026-06-04T10:30:55Z",
  "completed_at": "2026-06-04T10:31:02Z"
}
```

#### Example Response — `200 OK` (still processing)

```json
{
  "task_id": "batch-7f3a1b2c-d4e5-6789-abcd-ef0123456789",
  "status": "processing",
  "results": null,
  "created_at": "2026-06-04T10:30:55Z",
  "completed_at": null
}
```

#### Example Response — `404 Not Found`

```json
{
  "error": "not_found",
  "detail": "Task 'batch-invalid-id' not found.",
  "request_id": "c3d4e5f6-a7b8-9012-cdef-012345678901"
}
```

---

### `GET /api/v1/health`

Returns application health status, model-load state, and uptime.

#### Example Request

```bash
curl http://localhost:8000/api/v1/health
```

#### Example Response — `200 OK`

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

| `status` Value | Meaning |
|---|---|
| `healthy` | All models loaded and operational |
| `degraded` | At least one model loaded, but not all |
| `unhealthy` | No models loaded |

---

### `GET /api/v1/ready`

Kubernetes-style readiness probe. Returns `200` when all models are loaded, `503` otherwise.

#### Example Request

```bash
curl http://localhost:8000/api/v1/ready
```

#### Responses

**`200 OK`** — Service is ready:
```json
{"status": "ready"}
```

**`503 Service Unavailable`** — Service is not ready:
```json
{"status": "not_ready"}
```

---

## Schemas

### ModerationResponse

Returned by `POST /moderate/text` and `POST /moderate/image`.

| Field | Type | Description |
|---|---|---|
| `request_id` | `string` | Unique identifier for this request |
| `verdict` | `string` | `"approved"`, `"rejected"`, or `"needs_review"` |
| `categories` | `array[CategoryScore]` | Per-category confidence scores |
| `highest_risk_category` | `CategoryScore \| null` | Category with highest non-safe score |
| `processing_time_ms` | `float` | Inference time in milliseconds |
| `model_name` | `string` | Name of the model used |
| `input_type` | `string` | `"text"` or `"image"` |
| `timestamp` | `string` | ISO 8601 timestamp |

### CategoryScore

| Field | Type | Description |
|---|---|---|
| `category` | `string` | One of: `safe`, `toxic`, `hate_speech`, `spam`, `nsfw`, `violence` |
| `confidence` | `float` | Model confidence (0.0–1.0) |

### TaskResponse

Returned by `POST /moderate/batch`.

| Field | Type | Description |
|---|---|---|
| `task_id` | `string` | Unique task identifier |
| `status` | `string` | `"pending"`, `"processing"`, `"completed"`, or `"failed"` |
| `message` | `string` | Human-readable status message |

### TaskResultResponse

Returned by `GET /results/{task_id}`.

| Field | Type | Description |
|---|---|---|
| `task_id` | `string` | Unique task identifier |
| `status` | `string` | `"pending"`, `"processing"`, `"completed"`, or `"failed"` |
| `results` | `array[ModerationResponse] \| null` | Results (present when `completed`) |
| `created_at` | `string` | ISO 8601 creation timestamp |
| `completed_at` | `string \| null` | ISO 8601 completion timestamp |

### HealthResponse

Returned by `GET /health`.

| Field | Type | Description |
|---|---|---|
| `status` | `string` | `"healthy"`, `"degraded"`, or `"unhealthy"` |
| `version` | `string` | Application version |
| `uptime_seconds` | `float` | Seconds since application start |
| `models` | `object` | Map of model name → loaded (boolean) |
| `environment` | `string` | Deployment environment |

### ErrorResponse

Returned for all error responses (4xx, 5xx).

| Field | Type | Description |
|---|---|---|
| `error` | `string` | Short error code or title |
| `detail` | `string` | Human-readable error description |
| `request_id` | `string \| null` | Request ID for tracing |

---

## Error Handling

All errors return a consistent JSON structure:

```json
{
  "error": "error_code",
  "detail": "Human-readable description of what went wrong.",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### HTTP Status Codes

| Code | Meaning | When |
|---|---|---|
| `200` | OK | Request succeeded |
| `422` | Unprocessable Entity | Request body validation failed (e.g., empty text, invalid JSON) |
| `404` | Not Found | Task ID does not exist in result store |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Unexpected error during inference or processing |
| `503` | Service Unavailable | Model pipeline or task queue not loaded/available |

### Validation Error Response — `422`

```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "String should have at least 1 character",
      "type": "string_too_short",
      "input": ""
    }
  ]
}
```

---

## Rate Limiting

| Setting | Default | Environment Variable |
|---|---|---|
| Requests per minute (per client) | 60 | `SHIELDAI_API_RATE_LIMIT_PER_MINUTE` |
| Max request body size | 10 MB | `SHIELDAI_API_MAX_REQUEST_SIZE_MB` |
| Max batch items | 100 | Hardcoded in schema validation |

When a rate limit is exceeded, the API returns:

```json
{
  "error": "rate_limit_exceeded",
  "detail": "Too many requests. Please retry after 60 seconds.",
  "request_id": "..."
}
```

**Headers included in rate-limited responses:**

| Header | Description |
|---|---|
| `Retry-After` | Seconds until the rate limit resets |
| `X-RateLimit-Limit` | Maximum requests allowed per window |
| `X-RateLimit-Remaining` | Requests remaining in current window |

---

## Verdict Logic

The moderation verdict is determined by comparing category confidence scores against configurable thresholds:

```
if any category score >= rejection threshold → REJECTED
elif any category score >= needs_review threshold → NEEDS_REVIEW
else → APPROVED
```

Default thresholds:

| Category | Rejection Threshold | Needs Review Threshold |
|---|---|---|
| Toxic | 0.7 | 0.4 |
| Hate Speech | 0.7 | 0.4 |
| Spam | 0.6 | 0.4 |
| NSFW | 0.7 | 0.4 |
| Violence | 0.7 | 0.4 |

All thresholds are configurable via environment variables (see [Configuration](../README.md#️-configuration)).
