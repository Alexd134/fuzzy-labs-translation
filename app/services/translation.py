"""
Translation service

This module implements the core translation logic using Hugging Face transformers.
It encapsulates all model-specific details, providing a clean interface to the API layer.

Design considerations:
- Model is loaded once per process (singleton pattern via class variable)
- Thread-safe for concurrent requests within a single worker
"""
import logging
from typing import Optional

from transformers import pipeline, Pipeline

from app.config import config
from app.utils import measure_latency

# Configure module logger
logger = logging.getLogger(__name__)


class TranslationService:
    """
    Service for translating text using Hugging Face transformer models.
    """

    # Class-level variable to hold the model pipeline (singleton per process)
    _pipeline: Optional[Pipeline] = None
    _model_name: Optional[str] = None

    def __init__(self):
        """
        Initialize the translation service.
        """
        if TranslationService._pipeline is None:
            self._load_model()

    def _load_model(self) -> None:
        """
        Load the Hugging Face translation model and pipeline.

        This method is called once per worker process. The model is cached
        in a class variable for reuse across all requests handled by this worker.
        """
        logger.info(f"Loading translation model: {config.MODEL_NAME}")
        logger.info(f"Language pair: {config.SOURCE_LANGUAGE} -> {config.TARGET_LANGUAGE}")

        try:
            # Determine device
            device = config.DEVICE
            device_name = "GPU" if device >= 0 else "CPU"
            logger.info(f"Using device: {device_name} (device={device})")

            # Load the translation pipeline
            # Note: This downloads the model on first run (~200-300MB for MarianMT)
            TranslationService._pipeline = pipeline(
                "translation",
                model=config.MODEL_NAME,
                device=device,  # -1 for CPU, 0+ for GPU
            )
            TranslationService._model_name = config.MODEL_NAME

            logger.info(f"Model loaded successfully on {device_name}")

        except Exception as e:
            logger.error(f"Failed to load translation model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    @measure_latency
    def translate(self, text: str) -> str:
        """
        Translate text using the loaded model.

        Args:
            text: Source text to translate (assumed to be in SOURCE_LANGUAGE)

        Returns:
            Translated text in TARGET_LANGUAGE

        Raises:
            RuntimeError: If model is not loaded or translation fails
        """
        if TranslationService._pipeline is None:
            logger.error("Translation pipeline not initialized")
            raise RuntimeError("Translation model not loaded")

        try:
            logger.debug(f"Translating text (length: {len(text)} chars)")

            # Perform translation
            # The pipeline returns a list of dicts: [{'translation_text': '...'}]
            result = TranslationService._pipeline(text)

            if not result or len(result) == 0:
                raise ValueError("Model returned empty result")

            translated_text = result[0]["translation_text"]

            logger.debug(f"Translation completed (output length: {len(translated_text)})")

            return translated_text

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise RuntimeError(f"Translation error: {str(e)}") from e


    ## used in the api health check
    @property
    def model_name(self) -> str:
        """Return the name of the loaded model."""
        return TranslationService._model_name or config.MODEL_NAME

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return TranslationService._pipeline is not None

    @classmethod
    def get_model_info(cls) -> dict:
        """
        Get information about the loaded model.

        Useful for health checks and debugging.
        """
        return {
            "model_name": cls._model_name or config.MODEL_NAME,
            "source_language": config.SOURCE_LANGUAGE,
            "target_language": config.TARGET_LANGUAGE,
            "is_loaded": cls._pipeline is not None,
        }
