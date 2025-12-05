"""
Unit tests for TranslationService.

These tests verify the core translation logic in isolation from the API layer.
Tests include both mocked tests (for fast CI/CD pipelines) and integration tests
(for verifying actual model behavior).
"""
import pytest
from unittest.mock import Mock, patch

from app.services.translation import TranslationService
from app.config import config


class TestTranslationServiceMocked:
    """
    Mocked tests for TranslationService.

    These tests mock the Hugging Face pipeline to avoid downloading models
    during test execution. Useful for:
    - Fast CI/CD pipelines
    - Testing error handling logic
    - Testing without model dependencies
    """

    @patch('app.services.translation.pipeline')
    def test_service_initialization(self, mock_pipeline):
        """Test that service initializes and loads model correctly."""
        mock_pipeline.return_value = Mock()

        # Reset the class variable to simulate fresh initialization
        TranslationService._pipeline = None

        service = TranslationService()

        # Verify pipeline was called with correct parameters
        mock_pipeline.assert_called_once_with(
            "translation",
            model=config.MODEL_NAME,
            device=-1
        )
        assert service.is_loaded

    @patch('app.services.translation.TranslationService._pipeline')
    def test_translate_success(self, mock_pipeline):
        """Test successful translation."""
        # Setup mock to return translation result
        mock_pipeline.return_value = [{"translation_text": "Bonjour"}]
        TranslationService._pipeline = mock_pipeline

        service = TranslationService()
        result = service.translate("Hello")

        assert result == "Bonjour"
        mock_pipeline.assert_called_once_with("Hello")

    @patch('app.services.translation.TranslationService._pipeline')
    def test_translate_empty_result_raises_error(self, mock_pipeline):
        """Test that empty model results raise appropriate error."""
        mock_pipeline.return_value = []
        TranslationService._pipeline = mock_pipeline

        service = TranslationService()

        with pytest.raises(RuntimeError, match="Translation error"):
            service.translate("Hello")

    @patch('app.services.translation.TranslationService._pipeline')
    def test_translate_model_exception_raises_error(self, mock_pipeline):
        """Test that model exceptions are properly handled."""
        mock_pipeline.side_effect = Exception("Model inference failed")
        TranslationService._pipeline = mock_pipeline

        service = TranslationService()

        with pytest.raises(RuntimeError, match="Translation error"):
            service.translate("Hello")

    def test_translate_without_loaded_model_raises_error(self):
        """Test that translating without a loaded model raises error."""
        # Temporarily set pipeline to None
        original_pipeline = TranslationService._pipeline
        TranslationService._pipeline = None

        service = TranslationService.__new__(TranslationService)

        with pytest.raises(RuntimeError, match="Translation model not loaded"):
            service.translate("Hello")

        # Restore original state
        TranslationService._pipeline = original_pipeline

    @patch('app.services.translation.TranslationService._pipeline')
    def test_model_info(self, mock_pipeline):
        """Test that model info is correctly returned."""
        TranslationService._pipeline = mock_pipeline
        TranslationService._model_name = "Helsinki-NLP/opus-mt-en-fr"

        info = TranslationService.get_model_info()

        assert info["model_name"] == "Helsinki-NLP/opus-mt-en-fr"
        assert info["source_language"] == config.SOURCE_LANGUAGE
        assert info["target_language"] == config.TARGET_LANGUAGE
        assert info["is_loaded"] is True

    @patch('app.services.translation.TranslationService._pipeline')
    def test_is_loaded_property(self, mock_pipeline):
        """Test the is_loaded property."""
        TranslationService._pipeline = mock_pipeline
        service = TranslationService()
        assert service.is_loaded is True

        TranslationService._pipeline = None
        service = TranslationService.__new__(TranslationService)
        assert service.is_loaded is False
