"""ASGI middleware for request tracing and structured access logging."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from shieldai.logging_config import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every HTTP request.

    The middleware:
    1. Generates a UUID-4 ``request_id`` (or reuses an incoming
       ``X-Request-ID`` header if present).
    2. Binds the ID to *structlog* context vars so every log line emitted
       during the request includes it.
    3. Returns the ID in the ``X-Request-ID`` response header for client-side
       correlation.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process an incoming request and inject the request ID."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Make the ID available to downstream handlers / loggers
        structlog.contextvars.bind_contextvars(request_id=request_id)
        request.state.request_id = request_id

        try:
            response: Response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            structlog.contextvars.clear_contextvars()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status code, and duration.

    Emits a single structured log event at ``info`` level once the response
    has been produced.  Timing is measured with :func:`time.perf_counter` for
    sub-millisecond accuracy.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Log request details after the response is generated."""
        start = time.perf_counter()

        response: Response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1_000

        logger.info(
            "request_handled",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        return response
