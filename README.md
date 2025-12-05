# Translation Microservice

A production-ready REST API for text translation using Hugging Face transformers. This microservice demonstrates MLOps best practices including clean architecture, stateless design, and deployment-ready containerization.

## Overview

This service provides English to French translation using the MarianMT model (`Helsinki-NLP/opus-mt-en-fr`) from Hugging Face.

### Tech Stack

- **Framework**: FastAPI (async-first, automatic OpenAPI docs)
- **ASGI Server**: Uvicorn with Gunicorn for production
- **ML/NLP**: Hugging Face Transformers (MarianMT)
- **Language**: Python 3.11
- **Package Manager**: UV
- **Container**: Docker with multi-stage builds

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and routes
│   ├── utils.py   
│   ├── config.py         # Configuration management
│   └── services/
│       ├── __init__.py
│       └── translation.py   # Translation service layer
├── tests/
│   ├── __init__.py
│   ├── test_api.py          # API endpoint tests
│   └── test_translation_service.py  # Service layer tests
├── pyproject.toml          # Python dependencies and project config
├── .python-version         # Python version for UV
├── pytest.ini              # Pytest configuration
├── Dockerfile              # Production container image
├── .dockerignore           # Docker build exclusions
└── README.md               # This file
```

## Quick Start

### Prerequisites

- Python 3.11 or higher
- UV - Python package installer 
- (Optional) Docker for containerized deployment

### Local Development Setup

1. **Create virtual environment and install dependencies with UV**

```bash
# UV automatically creates a venv and installs dependencies
uv sync
```

Note: First run will download the translation model (~300MB). This happens automatically on first use.

2. **Activate the virtual environment**

```bash
source .venv/bin/activate
```

3. **Run the service locally**

```bash
# Development mode (single worker, auto-reload)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`


### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Usage

### Translation Endpoint

**POST** `/translate`

Translates text from English to French.

#### Request Body

```json
{
  "source_language": "en",
  "target_language": "fr",
  "text": "Hello, how are you today?"
}
```

#### Response

```json
{
  "source_language": "en",
  "target_language": "fr",
  "translated_text": "Bonjour, comment allez-vous aujourd'hui?",
  "model_name": "Helsinki-NLP/opus-mt-en-fr"
}
```


### Health Check Endpoints

#### GET `/`
Basic service information and status.

#### GET `/health`
Detailed health check including model loading status. Returns:
- 200 OK: Service is healthy
- 503 Service Unavailable: Service not ready (model not loaded)

## Running Tests

### Install test dependencies

```bash
# With UV (includes dev dependencies)
uv sync --all-extras
```

### Run all tests

```bash
uv run pytest tests/ -v --cov=app --cov-report=html

# Run only fast tests (skip integration tests that load models)
uv run pytest tests/ -v -m "not slow"

# Run only integration tests
uv run pytest tests/ -v -m slow
```

### Test Structure

- **Unit Tests** (`test_translation_service.py`): Mock-based tests for service layer
- **API Tests** (`test_api.py`): FastAPI TestClient-based endpoint tests
- **Integration Tests**: Marked with `@pytest.mark.slow`, test with real models

## Production Deployment

### Running with Gunicorn

For production, use Gunicorn with multiple Uvicorn workers:

```bash
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --graceful-timeout 30 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
```

**Worker Configuration**:
- **Workers**: 2-4 per CPU core (each worker loads model into memory)
- **Memory**: ~500-800MB per worker (for MarianMT model)
- **Timeout**: 120s to handle longer translations
- **Example**: 4 workers = ~2-3GB RAM minimum

### Docker Deployment

#### Build the image

```bash
docker build -t translation-service:latest .
```

Build time: ~5-10 minutes (includes model download)

#### Run the container

```bash
docker run -d \
  --name translation-service \
  -p 8000:8000 \
  -e LOG_LEVEL=INFO \
  translation-service:latest
```

#### Environment Variables

Configure the service using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSLATION_MODEL_NAME` | `Helsinki-NLP/opus-mt-en-fr` | Hugging Face model name |
| `SOURCE_LANGUAGE` | `en` | Source language code |
| `TARGET_LANGUAGE` | `fr` | Target language code |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `TRANSFORMERS_CACHE` | `/home/appuser/.cache/huggingface` | Model cache directory |
| `PORT` | `8000` | Service port |
| `HOST` | `0.0.0.0` | Service host |
