# 🛡️ Godel Guardrail Enterprise

**Production-Ready Prompt Security Guardrail**  
FastAPI · Hot Reload · Prometheus Metrics · Fail-Open Support · Encrypted Audit Logs

---

## Overview

**Godel Guardrail Enterprise** is a lightweight, high-reliability **prompt security gateway**
designed to sit in front of LLM or AI services.

It performs **real-time inspection, rate limiting, and anomaly detection**
before requests reach your model.

This project is intentionally **practical**, not experimental.

❌ No research-only abstractions  
❌ No heavy ML/RL dependencies  
✅ Built for production environments

---

## Core Features

### 🔐 Defense-in-Depth Security

Layered protection applied in sequence:

1. **Input Sanitization**
   - Script/XSS pattern detection
   - Payload size limits

2. **Rate Limiting**
   - Global TPS bucket
   - Per-user TPS bucket
   - Burst support

3. **Pluggable Inspection Engine**
   - Regex-based prompt injection detection
   - Entropy-based obfuscation detection
   - Debt-based escalation model

---

### 🧠 Security Debt Model

Each request accumulates **security debt** instead of being instantly blocked.

- Allows low-risk requests to pass
- Escalates only when abuse patterns persist
- Reduces false positives in production

---

### ⚙️ Fail-Open Mode (Enterprise-Grade)

If a security plugin fails unexpectedly:

- **Fail-Open = true**
  - Service continues
  - Incident is logged
- **Fail-Open = false**
  - Request fails closed (503)

This prevents **single-plugin failure from taking down production traffic**.

---

### 🔄 Hot Configuration Reload

Security rules can be updated **without restarting the service**:

- Regex patterns
- Entropy thresholds
- Rate limits
- Fail-open behavior

Ideal for:
- Live incident response
- Gradual policy tuning
- Blue-green security updates

---

### 📊 Observability (Prometheus-Ready)

Built-in metrics include:

- Request counts (allowed / blocked / throttled)
- Request latency
- Active in-flight requests
- Accumulated security debt

Fully compatible with:
- Prometheus
- Grafana
- Kubernetes monitoring

---

### 🔏 Encrypted Audit Logging

Audit logs can be:

- Plaintext (development)
- **Fernet-encrypted (production)**

Encryption key is provided via environment variable.

---

## Architecture

Client │ ▼ Godel Guardrail ├─ Sanitizer ├─ Rate Limiter ├─ Security Plugins │     ├─ RegexTrap │     └─ EntropyTrap ├─ Audit Logger └─ Metrics Exporter │ ▼ LLM / AI Backend

---

## Installation

### Requirements

```bash
python >= 3.9

Dependencies

pip install fastapi uvicorn prometheus-client cryptography pydantic psutil


---

Running the Service

python godel_guardrail_enterprise_v10_1.py

Service endpoints:

Endpoint	Purpose

/scan	Prompt inspection
/reload	Hot reload configuration
/metrics	Prometheus metrics
/health	Kubernetes health probe
/docs	Swagger UI



---

Environment Variables

GODEL_KEY

Encryption key for audit logs.

export GODEL_KEY="base64_fernet_key"

If not provided:

A temporary key is generated

Warning is logged

Not recommended for production



---

API Usage

Scan Prompt

POST /scan

{
  "prompt": "Hello world",
  "user_id": "user123",
  "tier": "standard"
}

Response:

{
  "safe": true,
  "code": "S200",
  "reason": "OK"
}


---

Hot Reload Configuration

POST /reload

{
  "limits": { "standard": 1.0, "premium": 3.0 },
  "patterns": ["ignore previous", "system prompt"],
  "entropy": 5.8,
  "tps_g": 100,
  "tps_u": 5,
  "fail_open": true,
  "audit_encrypt": true
}


---

Kubernetes Readiness

GET /health

Returns:

Service status

Uptime

Memory usage


Safe for:

Liveness probes

Readiness probes



---

Design Philosophy

Security is a control layer, not a research lab

Stability beats theoretical optimality

Operational clarity > abstraction elegance

Fail safely, never silently



---

When to Use This

✅ LLM API Gateway
✅ Enterprise AI Security
✅ Prompt Injection Defense
✅ Multi-tenant AI Services
✅ Regulated environments

❌ Not a prompt optimizer
❌ Not a content moderation engine
❌ Not a research framework


---

License

MIT License

SPDX-License-Identifier: MIT
Copyright (C) 2025 red1239109-cmd


---

Status

Production-Ready

Actively designed for:

Reliability

Operational safety

Minimal cognitive overhead
