"""
API tests for translation microservice.

These tests verify the FastAPI endpoints, including request validation,
error handling, and response formatting. They use FastAPI's TestClient
to simulate HTTP requests without running a real server.
"""
import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from app.main import app
from app.config import config


# Create test client
client = TestClient(app)

class TestTranslateEndpoint:
    """Tests for the /translate endpoint."""

    @patch('app.main.translation_service')
    def test_translate_success(self, mock_service):
        """Test successful translation request."""
        mock_service.translate.return_value = "Bonjour le monde"
        mock_service.model_name = "Helsinki-NLP/opus-mt-en-fr"

        response = client.post(
            "/translate",
            json={
                "source_language": "en",
                "target_language": "fr",
                "text": "Hello world"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["source_language"] == "en"
        assert data["target_language"] == "fr"
        assert data["translated_text"] == "Bonjour le monde"
        assert data["model_name"] == "Helsinki-NLP/opus-mt-en-fr"

        # Verify service was called correctly
        mock_service.translate.assert_called_once_with("Hello world")

    @patch('app.main.translation_service')
    def test_translate_with_punctuation(self, mock_service):
        """Test translation with punctuation and longer text."""
        input_text = "Hello, how are you today?"
        translated = "Bonjour, comment allez-vous aujourd'hui?"

        mock_service.translate.return_value = translated
        mock_service.model_name = "Helsinki-NLP/opus-mt-en-fr"

        response = client.post(
            "/translate",
            json={
                "source_language": "en",
                "target_language": "fr",
                "text": input_text
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["translated_text"] == translated

    def test_translate_unsupported_language_pair(self):
        """Test that unsupported language pairs return 400."""
        response = client.post(
            "/translate",
            json={
                "source_language": "en",
                "target_language": "es",  # Spanish not supported
                "text": "Hello world"
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert "Unsupported language pair" in data["detail"]
        assert "en->fr" in data["detail"]  # Should mention supported pair

    def test_translate_empty_text(self):
        """Test that empty text returns 400."""
        response = client.post(
            "/translate",
            json={
                "source_language": "en",
                "target_language": "fr",
                "text": ""
            }
        )

        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

    def test_translate_whitespace_only_text(self):
        """Test that whitespace-only text returns 400."""
        response = client.post(
            "/translate",
            json={
                "source_language": "en",
                "target_language": "fr",
                "text": "   "
            }
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_translate_missing_fields(self):
        """Test that missing required fields return 422."""
        # Missing text field
        response = client.post(
            "/translate",
            json={
                "source_language": "en",
                "target_language": "fr"
            }
        )

        assert response.status_code == 422

        # Missing source_language
        response = client.post(
            "/translate",
            json={
                "target_language": "fr",
                "text": "Hello"
            }
        )

        assert response.status_code == 422

    def test_translate_invalid_json(self):
        """Test that invalid JSON returns 422."""
        response = client.post(
            "/translate",
            data="not json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    @patch('app.main.translation_service')
    def test_translate_service_error(self, mock_service):
        """Test that service errors are handled gracefully."""
        mock_service.translate.side_effect = RuntimeError("Model error")

        response = client.post(
            "/translate",
            json={
                "source_language": "en",
                "target_language": "fr",
                "text": "Hello"
            }
        )

        assert response.status_code == 500
        data = response.json()
        assert "Translation service error" in data["detail"]

    @patch('app.main.translation_service')
    def test_translate_strips_whitespace(self, mock_service):
        """Test that input text whitespace is stripped."""
        mock_service.translate.return_value = "Bonjour"
        mock_service.model_name = "test-model"

        response = client.post(
            "/translate",
            json={
                "source_language": "en",
                "target_language": "fr",
                "text": "  Hello  "
            }
        )

        assert response.status_code == 200
        # Verify that stripped text was passed to service
        mock_service.translate.assert_called_once_with("Hello")

    def test_translate_very_long_text_rejected(self):
        """Test that excessively long text is rejected."""
        # Text longer than max_length (5000 chars)
        very_long_text = "a" * 5001

        response = client.post(
            "/translate",
            json={
                "source_language": "en",
                "target_language": "fr",
                "text": very_long_text
            }
        )

        assert response.status_code == 422  # Validation error

    @patch('app.main.translation_service')
    def test_translate_accepts_max_length_text(self, mock_service):
        """Test that text at max length (5000 chars) is accepted."""
        max_length_text = "a" * 5000
        mock_service.translate.return_value = "translated"
        mock_service.model_name = "test-model"

        response = client.post(
            "/translate",
            json={
                "source_language": "en",
                "target_language": "fr",
                "text": max_length_text
            }
        )

        assert response.status_code == 200


# THIS TEST IS NOT CURRENTLY WORKING
# Integration test with real service
class TestTranslateEndpointIntegration:
    """
    Integration tests with real translation service.

    These tests use the actual model and should be run selectively.
    """

    @pytest.mark.slow
    def test_translate_end_to_end(self):
        """
        End-to-end test with real model.

        This test loads the actual model and performs a real translation.
        Mark as slow test for selective execution.
        """
        response = client.post(
            "/translate",
            json={
                "source_language": "en",
                "target_language": "fr",
                "text": "Hello"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["source_language"] == "en"
        assert data["target_language"] == "fr"
        assert len(data["translated_text"]) > 0
        assert data["model_name"] == config.MODEL_NAME
        # Basic sanity check - French translation should contain French characters
        assert "bonjour" in data["translated_text"].lower()


# Run with:
# pytest tests/test_api.py -v              # Run all tests
# pytest tests/test_api.py -v -m "not slow"  # Skip slow integration tests
# pytest tests/test_api.py -v -m slow        # Run only integration tests
