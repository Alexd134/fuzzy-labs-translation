"""
Configuration module for the translation microservice.

This module manages all configuration settings, primarily loading from environment
variables. 
"""
import os
from typing import Optional


class Config:
    """
    Application configuration with environment variable support.

    All settings can be overridden via environment variables, making this
    service portable across dev, staging, and production environments.
    """

    # Model configuration
    MODEL_NAME: str = os.getenv(
        "TRANSLATION_MODEL_NAME",
        "Helsinki-NLP/opus-mt-en-fr"
    )

    # Supported language pair (configurable via environment variables)
    SOURCE_LANGUAGE: str = os.getenv("SOURCE_LANGUAGE", "en")
    TARGET_LANGUAGE: str = os.getenv("TARGET_LANGUAGE", "fr")

    # Model loading configuration
    # Cache directory for Hugging Face models (useful in containerised environments)
    MODEL_CACHE_DIR: Optional[str] = os.getenv("TRANSFORMERS_CACHE")

    # Device configuration: -1 for CPU, 0 for GPU:0, 1 for GPU:1, etc.
    DEVICE: int = int(os.getenv("DEVICE", "-1"))

    # API configuration
    API_TITLE: str = "Translation Microservice"
    API_DESCRIPTION: str = (
        "Production-ready translation API using Hugging Face transformers. "
        "Translates text from English to French using MarianMT models."
    )
    API_VERSION: str = "1.0.0"

    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Logging configuration - info for dev work, up for prod
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Production deployment notes:
    # - Set TRANSFORMERS_CACHE to a persistent volume in containerised deployments
    #   to avoid re-downloading models on container restart. Makes for quicker start up


# Create a singleton instance for easy importing
config = Config()
