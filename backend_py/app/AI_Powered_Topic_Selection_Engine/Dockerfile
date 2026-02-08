FROM python:3.12-slim

WORKDIR /app

# Install python deps
COPY backend_py/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy engine code
COPY backend_py/app /app

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    USE_QDRANT=true \
    QDRANT_AUTO_INDEX=true \
    QDRANT_VECTOR_SIZE=256 \
    OPENALEX_FALLBACK=true

EXPOSE 8000

# Run FastAPI app directly
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
