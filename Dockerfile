# Multi-stage: the build stage carries compilers and build headers that a
# running container has no use for and that widen its attack surface.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY backend ./backend

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir ".[providers]"


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/backend:/app/sdk/python \
    PATH="/opt/venv/bin:$PATH"

# Runs as a non-root user: a container process that can write to its own code
# turns a single RCE into a persistent one.
RUN groupadd --system --gid 1001 modelops \
    && useradd --system --uid 1001 --gid modelops --create-home modelops

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=modelops:modelops backend ./backend
COPY --chown=modelops:modelops sdk ./sdk
COPY --chown=modelops:modelops alembic.ini ./
COPY --chown=modelops:modelops README.md ./

USER modelops

EXPOSE 8000

# Hits /health, which reports database and provider status. A TCP check would
# call a container healthy while every request failed on a dead database.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

# One worker per container. Scale with replicas rather than in-process workers,
# so a crash loses one request's worth of work and the orchestrator can see it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
