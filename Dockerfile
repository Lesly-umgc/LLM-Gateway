# LNM Gateway — container image (FastAPI gateway; model server comes from compose).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch keeps the image small; the semantic cache only needs embeddings.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/ ./gateway/
COPY rails/ ./rails/
COPY monitors/ ./monitors/
COPY optimizer/ ./optimizer/
COPY backends/ ./backends/
COPY common/ ./common/

# Compose overrides these; the defaults make `docker run` alone do something sane.
ENV LNM_BACKEND=ollama \
    OLLAMA_BASE_URL=http://ollama:11434 \
    OLLAMA_MODEL=llama3.2:3b \
    LNM_REDIS_URL=redis://redis:6379/0 \
    LNM_AUDIT_PATH=/data/audit.jsonl

EXPOSE 8000
VOLUME ["/data"]

CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
