# syntax=docker/dockerfile:1
# D2 Sprint qualification task - team Anote
# Runs fully offline: model weights are baked into the image.
# Optional local wheelhouse: `--build-context wheels=<dir>` installs from pre-downloaded
# wheels (useful offline / behind slow proxies). Default is empty, so pip uses PyPI.
FROM scratch AS wheels

FROM python:3.11-slim

LABEL org.opencontainers.image.title="anote-d2-qualification-task" \
      org.opencontainers.image.description="Team Anote - DARPA D2 Sprint qualification task: four-site extremity injury classification (ICD v1.0)" \
      org.opencontainers.image.source="https://github.com/anote-ai/anote-d2-qualification-task" \
      org.opencontainers.image.version="1.3.0" \
      org.opencontainers.image.vendor="Anote"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TORCH_HOME=/app/.torch

WORKDIR /app

# PyPI torch wheels bundle the CUDA 12.8 runtime (no kernel driver), which runs on
# A40 / L40 / H100 with NVIDIA driver >= 580. Falls back to CPU if no GPU is visible.
COPY requirements.txt .
# Optional build secret "pipca": a CA bundle for builds behind a TLS-intercepting proxy.
# It is mounted only during this step and never stored in the image.
RUN --mount=type=secret,id=pipca,required=false \
    --mount=type=bind,from=wheels,target=/wheels \
    if [ -s /run/secrets/pipca ]; then export PIP_CERT=/run/secrets/pipca; fi && \
    pip install --no-cache-dir --find-links /wheels -r requirements.txt && \
    find /usr/local/lib/python3.11 -name '__pycache__' -prune -exec rm -rf {} +

COPY d2/ d2/
COPY inference.py validate_predictions.py entrypoint.sh ./
COPY model/ model/
RUN chmod +x entrypoint.sh && mkdir -p /app/input /app/output

ENTRYPOINT ["/app/entrypoint.sh"]
