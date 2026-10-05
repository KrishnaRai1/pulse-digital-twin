# syntax=docker/dockerfile:1
# Multi-stage build:
#   train   - installs PyTorch (CPU), trains the card CNN and the thermal model, and builds the
#             field-dataset store (tables + LightGBM models) from data/field. Needs ~1 GB RAM.
#   runtime - API only: no PyTorch (the CNN runs on NumPy), ~320 MB RAM, starts in seconds.
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# LightGBM needs the OpenMP runtime; curl is used by the health check
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /srv

# ---------------------------------------------------------------- train
FROM base AS train
COPY backend/requirements.txt backend/requirements-train.txt /srv/backend/
RUN python -m venv /opt/train \
 && /opt/train/bin/pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3" \
 && /opt/train/bin/pip install -r /srv/backend/requirements-train.txt
COPY backend/ /srv/backend/
COPY data/ /srv/data/
WORKDIR /srv/backend
RUN /opt/train/bin/python -m scripts.train_models --models-dir /srv/backend/models \
 && /opt/train/bin/python -m scripts.build_field_store --data /srv/data/field --out /srv/backend/field_store

# ---------------------------------------------------------------- runtime
FROM base AS runtime
COPY backend/requirements.txt /srv/backend/
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r /srv/backend/requirements.txt
ENV PATH="/opt/venv/bin:$PATH"
COPY backend/ /srv/backend/
COPY data/ /srv/data/
COPY --from=train /srv/backend/models /srv/backend/models
COPY --from=train /srv/backend/field_store /srv/backend/field_store
RUN useradd --system --uid 10001 --home /srv pulse \
 && mkdir -p /srv/backend/data \
 && chown -R pulse /srv/backend/data /srv/backend/models /srv/backend/field_store
USER pulse
WORKDIR /srv/backend
ENV PULSE_ENV=prod \
    PULSE_MODELS_DIR=/srv/backend/models \
    PULSE_SAMPLES_DIR=/srv/data/samples \
    PULSE_FIELD_DATA_DIR=/srv/data/field \
    PULSE_FIELD_STORE_DIR=/srv/backend/field_store \
    PULSE_FIELD_AUTO_BUILD=false \
    PULSE_AUTO_TRAIN=false \
    PULSE_DATABASE_URL=sqlite:////srv/backend/data/pulse.db \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fsS "http://localhost:${PORT}/api/v1/health" || exit 1
# $PORT is set by most PaaS hosts (Render, Railway, Fly, Hugging Face); proxy headers give the
# rate limiter the real client address behind the platform's load balancer
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
