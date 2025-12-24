# SPDX-License-Identifier: MIT
# Copyright (C) 2025 red1239109-cmd

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal system deps (optional but common)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    prometheus-client \
    cryptography \
    pydantic \
    psutil

# Copy app
COPY godel_guardrail_enterprise_v10_1.py /app/app.py

# Optional: provide a default (override in runtime for production)
# ENV GODEL_KEY=""

EXPOSE 8000

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
