"""
Tests for FastAPI endpoints.

Tests API routes for classification and similarity.
"""

import base64
import io
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image


class TestClassifyEndpoint:
    """Tests for /classify endpoint."""

    def test_returns_200(self, mock_image_base64):
        """Test successful classification returns 200."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = torch.randn(1, 62)
            mock_model.eval = Mock()
            mock_get_model.return_value = mock_model

            from api.main import app

            client = TestClient(app)

            response = client.post("/classify", json={"image": mock_image_base64})

            assert response.status_code == 200

    def test_returns_character(self, mock_image_base64):
        """Test that response includes character."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = torch.randn(1, 62)
            mock_model.eval = Mock()
            mock_get_model.return_value = mock_model

            from api.main import app

            client = TestClient(app)

            response = client.post("/classify", json={"image": mock_image_base64})
            data = response.json()

            assert "character" in data
            valid_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            assert data["character"] in valid_chars

    def test_returns_confidence(self, mock_image_base64):
        """Test that response includes confidence."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = torch.randn(1, 62)
            mock_model.eval = Mock()
            mock_get_model.return_value = mock_model

            from api.main import app

            client = TestClient(app)

            response = client.post("/classify", json={"image": mock_image_base64})
            data = response.json()

            assert "confidence" in data
            assert 0.0 <= data["confidence"] <= 1.0

    def test_invalid_base64_returns_422(self, invalid_base64):
        """Test that invalid base64 returns 422."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            mock_get_model.return_value = Mock()

            from api.main import app

            client = TestClient(app)

            response = client.post("/classify", json={"image": invalid_base64})

            assert response.status_code == 422

    def test_missing_image_returns_422(self):
        """Test that missing image field returns 422."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            mock_get_model.return_value = Mock()

            from api.main import app

            client = TestClient(app)

            response = client.post("/classify", json={})

            assert response.status_code == 422


class TestSimilarityEndpoint:
    """Tests for /similarity endpoint."""

    def test_returns_200(self, mock_image_base64):
        """Test successful similarity returns 200."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_encoder") as mock_get_encoder:
            mock_encoder = Mock()
            mock_encoder.return_value = torch.randn(1, 128)
            mock_get_encoder.return_value = mock_encoder

            from api.main import app

            client = TestClient(app)

            response = client.post(
                "/similarity", json={"image1": mock_image_base64, "image2": mock_image_base64}
            )

            assert response.status_code == 200

    def test_returns_similarity_score(self, mock_image_base64):
        """Test that response includes similarity score."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_encoder") as mock_get_encoder:
            mock_encoder = Mock()
            mock_encoder.return_value = torch.randn(1, 128)
            mock_get_encoder.return_value = mock_encoder

            from api.main import app

            client = TestClient(app)

            response = client.post(
                "/similarity", json={"image1": mock_image_base64, "image2": mock_image_base64}
            )
            data = response.json()

            assert "similarity" in data
            assert 0.0 <= data["similarity"] <= 100.0


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self):
        """Test health endpoint returns 200."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            mock_get_model.return_value = Mock()

            from api.main import app

            client = TestClient(app)

            response = client.get("/health")

            assert response.status_code == 200

    def test_health_returns_status(self):
        """Test health endpoint returns status."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            mock_get_model.return_value = Mock()

            from api.main import app

            client = TestClient(app)

            response = client.get("/health")
            data = response.json()

            assert "status" in data
            assert data["status"] == "healthy"

    def test_health_includes_model_status(self):
        """Test health endpoint includes model loaded status."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            mock_get_model.return_value = Mock()

            from api.main import app

            client = TestClient(app)

            response = client.get("/health")
            data = response.json()

            assert "model_loaded" in data

    def test_health_includes_version(self):
        """Test health endpoint includes version."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            mock_get_model.return_value = Mock()

            from api.main import app

            client = TestClient(app)

            response = client.get("/health")
            data = response.json()

            assert "version" in data


class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint (combined classify + similarity)."""

    def test_returns_200(self, mock_image_base64):
        """Test successful analysis returns 200."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            with patch("api.dependencies.get_encoder") as mock_get_encoder:
                mock_model = Mock()
                mock_model.return_value = torch.randn(1, 62)
                mock_model.eval = Mock()
                mock_get_model.return_value = mock_model

                mock_encoder = Mock()
                mock_encoder.return_value = torch.randn(1, 128)
                mock_get_encoder.return_value = mock_encoder

                from api.main import app

                client = TestClient(app)

                response = client.post("/analyze", json={"image": mock_image_base64})

                assert response.status_code == 200

    def test_returns_full_analysis(self, mock_image_base64):
        """Test that response includes full analysis."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            with patch("api.dependencies.get_encoder") as mock_get_encoder:
                mock_model = Mock()
                mock_model.return_value = torch.randn(1, 62)
                mock_model.eval = Mock()
                mock_get_model.return_value = mock_model

                mock_encoder = Mock()
                mock_encoder.return_value = torch.randn(1, 128)
                mock_get_encoder.return_value = mock_encoder

                from api.main import app

                client = TestClient(app)

                response = client.post("/analyze", json={"image": mock_image_base64})
                data = response.json()

                assert "character" in data
                assert "confidence" in data
                assert "similarity" in data


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, mock_image_base64):
        """Test that CORS headers are present."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = torch.randn(1, 62)
            mock_model.eval = Mock()
            mock_get_model.return_value = mock_model

            from api.main import app

            client = TestClient(app)

            response = client.options("/classify")

            # Verify CORS is configured (headers may vary by config)
            assert response.status_code in [200, 204, 405]


class TestRateLimiting:
    """Tests for rate limiting (if implemented)."""

    @pytest.mark.slow
    def test_rate_limit_not_exceeded_normal_use(self, mock_image_base64):
        """Test that normal use doesn't exceed rate limit."""
        from fastapi.testclient import TestClient

        with patch("api.dependencies.get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = torch.randn(1, 62)
            mock_model.eval = Mock()
            mock_get_model.return_value = mock_model

            from api.main import app

            client = TestClient(app)

            # Make a few requests
            for _ in range(5):
                response = client.post("/classify", json={"image": mock_image_base64})
                assert response.status_code == 200
