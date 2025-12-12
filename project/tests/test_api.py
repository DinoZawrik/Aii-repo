"""
Tests for FastAPI service.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        # Import here to avoid loading model during test collection
        from src.service.app import app
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_type" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_predict_request_schema(self, client):
        """Test predict endpoint request schema."""
        # Test with missing text field
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_with_valid_text(self, client):
        """Test predict endpoint with valid input."""
        response = client.post(
            "/predict",
            json={"text": "This is a great product!"}
        )

        # Either 200 (model loaded) or 503 (model not loaded)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "text" in data
            assert "sentiment" in data
            assert "confidence" in data
            assert "label" in data
            assert data["sentiment"] in ["positive", "negative"]

    def test_batch_predict_request_schema(self, client):
        """Test batch predict endpoint request schema."""
        # Test with missing texts field
        response = client.post("/predict/batch", json={})
        assert response.status_code == 422  # Validation error

    def test_batch_predict_with_valid_texts(self, client):
        """Test batch predict endpoint with valid input."""
        response = client.post(
            "/predict/batch",
            json={"texts": ["Great product!", "Terrible service."]}
        )

        # Either 200 (model loaded) or 503 (model not loaded)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_latency_ms" in data
            assert len(data["predictions"]) == 2


class TestAPIValidation:
    """Tests for API input validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.service.app import app
        return TestClient(app)

    def test_predict_empty_text(self, client):
        """Test prediction with empty text."""
        response = client.post("/predict", json={"text": ""})
        # Should handle empty text gracefully
        assert response.status_code in [200, 503]

    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list."""
        response = client.post("/predict/batch", json={"texts": []})
        assert response.status_code in [200, 503]

    def test_predict_long_text(self, client):
        """Test prediction with very long text."""
        long_text = "This is great! " * 1000
        response = client.post("/predict", json={"text": long_text})
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
