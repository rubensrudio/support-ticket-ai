# CPU-only image for the ticket-classifier REST API (OPS-02).
# Build from the repository root:
#   docker build -f docker/Dockerfile -t support-ticket-ai .
# Model artifacts are never baked into the image: mount them at /app/artifacts.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependency layer: runtime dependencies only (no dev group, no `train` extra).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Project layer.
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app \
    && mkdir -p /app/artifacts /app/var \
    && chown app:app /app/artifacts /app/var

ENV PATH="/app/.venv/bin:${PATH}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TICKET_ARTIFACTS_DIR=/app/artifacts \
    TICKET_DB_PATH=/app/var/tickets.db

USER app

EXPOSE 8000

CMD ["uvicorn", "--factory", "ticket_classifier.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
