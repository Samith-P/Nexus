FROM python:3.12-slim

WORKDIR /app

# Copy requirements first (for caching)
COPY backend_py/requirements.txt /app/requirements.txt

# 🔥 Install CPU-only PyTorch FIRST (very important)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Then install rest
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY backend_py/app /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    USE_QDRANT=true \
    QDRANT_AUTO_INDEX=true \
    OPENALEX_FALLBACK=true

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
