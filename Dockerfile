FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-default-groups --no-editable

COPY data/chroma /app/data/chroma
COPY reports /app/reports

EXPOSE 8000

ENV LEGRAG_CHROMA_PERSIST_PATH=/app/data/chroma \
    LEGRAG_COLLECTION_NAME=egyptian_civil_code \
    LEGRAG_REPORTS_DIR=/app/reports

CMD ["/app/.venv/bin/uvicorn", "legrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
