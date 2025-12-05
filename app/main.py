"""
FastAPI application for translation microservice.

This module implements the API layer, exposing REST endpoints for text translation.
It handles request validation, error handling, and structured logging.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from app.config import config
from app.services import TranslationService

# Configure structured logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instance (initialized at startup)
# Model loads during FastAPI startup so we can catch logging etc
translation_service: TranslationService = None


# Pydantic models for request/response validation
class TranslationRequest(BaseModel):
    """
    Request model for translation endpoint.

    Attributes:
        source_language: Source language code (e.g., "en")
        target_language: Target language code (e.g., "fr")
        text: Text to translate (must be non-empty)
    """
    source_language: str = Field(
        ...,
        description="Source language code",
        example="en",
        min_length=2,
        max_length=5
    )
    target_language: str = Field(
        ...,
        description="Target language code",
        example="fr",
        min_length=2,
        max_length=5
    )
    text: str = Field(
        ...,
        description="Text to translate",
        example="Hello, how are you?",
        min_length=1,
        max_length=5000  # Reasonable limit to prevent abuse
    )

    @validator("text")
    def text_must_not_be_empty(cls, v):
        """Validate that text is not empty or only whitespace."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "source_language": "en",
                "target_language": "fr",
                "text": "Hello, how are you today?"
            }
        }


class TranslationResponse(BaseModel):
    """
    Response model for translation endpoint.

    Attributes:
        source_language: Source language code
        target_language: Target language code
        translated_text: The translated text
        model_name: Name of the model used for translation
    """
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    translated_text: str = Field(..., description="Translated text")
    model_name: str = Field(..., description="Model used for translation")

    class Config:
        schema_extra = {
            "example": {
                "source_language": "en",
                "target_language": "fr",
                "translated_text": "Bonjour, comment allez-vous aujourd'hui?",
                "model_name": "Helsinki-NLP/opus-mt-en-fr"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str
    error_code: str


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Handles startup and shutdown events:
    - Startup: Initialize translation service and load model
    - Shutdown: Cleanup resources (if needed)

    This ensures the model is loaded once per worker process before
    handling any requests, avoiding cold starts on first request.
    """
    logger.info("Starting translation microservice...")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Supported language pair: {config.SOURCE_LANGUAGE} -> {config.TARGET_LANGUAGE}")

    global translation_service
    try:
        translation_service = TranslationService()
        logger.info("Translation service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize translation service: {str(e)}")
        raise

    yield

    logger.info("Shutting down translation microservice...")


# Initialize FastAPI application
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Detailed health check endpoint.

    Useful for Kubernetes liveness/readiness probes or load balancer health checks.

    Returns:
        Detailed health status including model loading state
    """
    if translation_service is None or not translation_service.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "reason": "Translation service not initialized"
            }
        )

    return {
        "status": "healthy",
        "model_info": TranslationService.get_model_info(),
        "supported_languages": {
            "source": config.SOURCE_LANGUAGE,
            "target": config.TARGET_LANGUAGE
        }
    }


@app.post(
    "/translate",
    response_model=TranslationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Translation"],
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid request (unsupported language pair or empty text)"
        },
        500: {
            "model": ErrorResponse,
            "description": "Translation service error"
        }
    }
)
async def translate(request: TranslationRequest):
    """
    Translate text from source language to target language.

    Currently supports only English (en) to French (fr) translation.

    Args:
        request: Translation request containing source/target languages and text

    Returns:
        Translation response with translated text and model information

    Raises:
        HTTPException 400: If language pair is not supported or text is invalid
        HTTPException 500: If translation service encounters an error
    """
    logger.info(
        f"Translation request received",
        extra={
            "source_lang": request.source_language,
            "target_lang": request.target_language,
            "text_length": len(request.text)
        }
    )

    # Validate language pair against configured languages
    if (request.source_language != config.SOURCE_LANGUAGE or
        request.target_language != config.TARGET_LANGUAGE):
        error_msg = (
            f"Unsupported language pair: {request.source_language}->{request.target_language}. "
            f"Only {config.SOURCE_LANGUAGE}->{config.TARGET_LANGUAGE} is supported."
        )
        logger.warning(error_msg)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )

    # Future: To support multiple pairs, replace above with:
    # supported_pairs = [(config.SOURCE_LANGUAGE, config.TARGET_LANGUAGE)]
    # if (request.source_language, request.target_language) not in supported_pairs:
    #     raise HTTPException(400, f"Supported pairs: {supported_pairs}")

    try:
        translated_text = translation_service.translate(request.text)

        # Log request/response details (latency is logged by decorator)
        logger.info(
            f"Translation request processed",
            extra={
                "input_length": len(request.text),
                "output_length": len(translated_text)
            }
        )

        return TranslationResponse(
            source_language=request.source_language,
            target_language=request.target_language,
            translated_text=translated_text,
            model_name=translation_service.model_name
        )

    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation service error: {str(e)}"
        )



