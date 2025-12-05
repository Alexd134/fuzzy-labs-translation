# Translation Microservice - Production Dockerfile
# Multi-stage build for optimized image size and security
#
# Build strategy:
# - Stage 1: Build dependencies and download models using UV
# - Stage 2: Create minimal runtime image with only necessary files
#
# Production considerations:
# - Uses slim Python image to reduce attack surface
# - Non-root user for security
# - Model caching to avoid re-downloads
# - Health checks for orchestration platforms

# Stage 1: Builder - Install dependencies and download models
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
# - gcc, g++: For compiling Python extensions
# - git: For some packages that install from git
# - curl: To download uv installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV - the fast Python package installer
# UV is 10-100x faster than pip and has better dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for better layer caching
# (dependencies change less frequently than application code)
COPY pyproject.toml .
COPY .python-version .

# Install Python dependencies using UV
# UV automatically creates a virtual environment and installs dependencies
# --no-dev: Skip development dependencies in production
ENV UV_SYSTEM_PYTHON=1
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY app/ ./app/

# Pre-download the translation model to avoid cold start in production
# This downloads ~300MB but makes startup instant
# Set HF_HOME to control cache location
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from transformers import pipeline; pipeline('translation', model='Helsinki-NLP/opus-mt-en-fr', cache_dir='/app/.cache/huggingface')"


# Stage 2: Runtime - Minimal image with only runtime dependencies
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only runtime system dependencies
# - libgomp1: Required by PyTorch for multi-threading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
# In production, never run containers as root
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy Python dependencies from builder stage
# UV installs packages to the system Python site-packages when UV_SYSTEM_PYTHON=1
COPY --from=builder --chown=appuser:appuser /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=appuser:appuser /usr/local/bin /usr/local/bin

# Copy application code from builder
COPY --from=builder --chown=appuser:appuser /app/app ./app

# Copy pre-downloaded models from builder
COPY --from=builder --chown=appuser:appuser /app/.cache /home/appuser/.cache

# Set environment variables
# - PYTHONUNBUFFERED: Ensure logs are flushed immediately (important for containers)
# - TRANSFORMERS_CACHE: Point to our pre-downloaded models
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface \
    TRANSFORMERS_OFFLINE=0 \
    HF_HOME=/home/appuser/.cache/huggingface

# Switch to non-root user
USER appuser

# Expose port
# Note: In Kubernetes, this is informational; actual port is configured in Service
EXPOSE 8000

# Health check for container orchestration platforms
# Checks if the service is responsive and healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: Run with Gunicorn for production
# - 4 workers: Adjust based on CPU cores (recommended: 2-4 per core)
# - uvicorn.workers.UvicornWorker: ASGI worker for FastAPI
# - bind: Listen on all interfaces
# - timeout: 120s for model inference (adjust based on expected translation time)
# - graceful-timeout: Allow 30s for graceful shutdown
# - access-logformat: Custom log format for better monitoring
CMD ["gunicorn", \
     "app.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]

# Alternative: For development, use uvicorn directly (single worker, auto-reload)
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Build and run instructions:
#
# Build:
#   docker build -t translation-service:latest .
#
# Run locally:
#   docker run -p 8000:8000 translation-service:latest
#
# Run with custom worker count:
#   docker run -p 8000:8000 -e GUNICORN_WORKERS=8 translation-service:latest
#
# Run with environment overrides:
#   docker run -p 8000:8000 \
#     -e LOG_LEVEL=DEBUG \
#     -e TRANSLATION_MODEL_NAME=Helsinki-NLP/opus-mt-en-fr \
#     translation-service:latest
