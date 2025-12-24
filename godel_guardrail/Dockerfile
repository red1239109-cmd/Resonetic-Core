FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional but handy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY godel_guardrail_enterprise_v10_1.py /app/godel_guardrail_enterprise_v10_1.py

EXPOSE 8000

# Healthcheck expects /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "godel_guardrail_enterprise_v10_1:app", "--host", "0.0.0.0", "--port", "8000"]
