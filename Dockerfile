# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency specification first for layer caching
COPY pyproject.toml README.md ./
RUN mkdir -p src/shieldai && touch src/shieldai/__init__.py

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy source code
COPY src/ ./src/

# Install the package itself (clean build artifacts first to prevent setuptools caching)
RUN rm -rf build/ *.egg-info src/*.egg-info && \
    pip install --no-cache-dir --no-deps --force-reinstall .


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root user
RUN groupadd --gid 1000 shieldai && \
    useradd --uid 1000 --gid 1000 --create-home shieldai

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY --from=builder /app/src ./src

# Create data directory for SQLite
RUN mkdir -p /app/data && chown -R shieldai:shieldai /app

# Switch to non-root user
USER shieldai

# Environment variables
ENV SHIELDAI_ENVIRONMENT=production \
    SHIELDAI_LOG_LEVEL=INFO \
    SHIELDAI_API_HOST=0.0.0.0 \
    SHIELDAI_API_PORT=8000 \
    SHIELDAI_STORAGE_DATABASE_PATH=/app/data/shieldai.db

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/v1/health'); r.raise_for_status()"

# Run the application
CMD ["python", "-m", "shieldai"]
