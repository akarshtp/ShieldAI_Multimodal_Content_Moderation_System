"""Integration tests for the health and readiness API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_endpoint(self, client: TestClient) -> None:
        """The health endpoint should return 200 with a HealthResponse shape."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

        data = resp.json()
        assert "status" in data
        assert data["status"] in {"healthy", "degraded", "unhealthy"}
        assert "version" in data
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert "models" in data
        assert isinstance(data["models"], dict)
        assert "environment" in data


class TestReadyEndpoint:
    """Tests for GET /api/v1/ready."""

    def test_ready_endpoint(self, client: TestClient) -> None:
        """When models are loaded, the readiness probe should return 200."""
        resp = client.get("/api/v1/ready")
        # The mocked pipelines have is_loaded() → MagicMock (truthy),
        # so the probe should return 200.
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "ready"
