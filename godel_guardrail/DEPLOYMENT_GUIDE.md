# 📘 Godel Guardrail Deployment Guide

**Version:** v10.0 Enterprise Edition
**Target Audience:** DevOps, SRE, and Security Architects

---

## 🏗️ 1. High Availability & Scalability

The current `v10.0` implementation uses in-memory `TokenBucket` for rate limiting. While fast, this creates a **"Split-Brain"** issue when deployed across multiple Kubernetes pods (e.g., a user gets 3x limit if you run 3 pods).

### ✅ Recommendation: Distributed Rate Limiting (Redis)

For horizontal scaling (Scale-Out), state must be externalized.

* **Architecture:** Replace in-memory `TokenBucket` with **Redis + Lua Scripts**.
* **Benefits:** Atomic counter increments across the entire cluster.
* **Implementation Strategy:**

```python
# Pseudo-code for Redis implementation
class RedisTokenBucket:
    def __init__(self, redis_client, key, rate, capacity):
        self.redis = redis_client
        self.key = key
        # ...

    async def allow(self):
        # Use Lua script to ensure atomicity (Check + Decrement)
        return await self.redis.evalsha(LUA_SCRIPT_HASH, 1, self.key, ...)

```

### ✅ Recommendation: CPU Optimization (Rust/Cython)

The **Entropy Trap** and **Regex Trap** are CPU-intensive. Python's Global Interpreter Lock (GIL) may become a bottleneck under extreme TPS (Transactions Per Second).

* **Strategy:** Rewrite the `inspect()` logic of plugins using **Rust** (via PyO3) or **Cython**.
* **Trigger:** Consider this transition if CPU usage consistently exceeds 70% per pod.

---

## 🛡️ 2. Security & Compliance (Bank-Grade)

Current implementation injects `GODEL_KEY` via environment variables. For regulated industries (Finance, Healthcare), this is insufficient.

### ✅ Recommendation: Secret Management (KMS)

Never store encryption keys in `deployment.yaml` or environment variables where they can be exposed via `docker inspect`.

* **AWS:** AWS KMS (Key Management Service) + IAM Roles for Service Accounts (IRSA).
* **Google:** Google Secret Manager.
* **Kubernetes:** HashiCorp Vault Sidecar Injector.
* **Mechanism:** The application should fetch the key directly from the Secret Manager into memory upon startup.

### ✅ Recommendation: PII Protection (GDPR/CCPA)

Logging raw `user_id` allows re-identification of users in audit logs.

* **Strategy:** Store two versions of the ID in logs:
1. **Analytical ID (Hash):** `SHA-256(user_id + salt)` → For statistical analysis (e.g., "How many distinct users?").
2. **Audit ID (Encrypted):** `AES-256(user_id, audit_key)` → For legal compliance (only decryptable with a specific warrant/key).



---

## 📊 3. Observability & Monitoring

The current `PrometheusMetrics` implementation is a solid foundation. However, care must be taken with **High Cardinality**.

### ✅ Recommendation: Prometheus Label Management

Prometheus stores a separate time series for every unique combination of label values.

* **Risk:** Do **NOT** use `user_id` or `prompt_hash` as a metric label.
* *Bad:* `requests_total{user_id="alice"}` → Explodes memory if you have 1M users.
* *Good:* `requests_total{tier="premium", status="blocked"}`.


* **Solution:** If user-level granularity is needed, send logs to **Elasticsearch (ELK)** or **Loki** instead of Prometheus metrics.

### ✅ Recommendation: Alerting Rules

Configure the following alerts in AlertManager:

* **P0 (Critical):** `godel_health_status == 0` (Service Down).
* **P1 (High):** `rate(godel_security_debt[5m]) > Threshold` (Massive Attack Detected).
* **P2 (Warn):** `process_cpu_seconds_total > 80%` (Scale-out needed).

---

## 🚀 4. Production Checklist

| Category | Item | Status |
| --- | --- | --- |
| **Infra** | Set `replicas: 3` (min) in Kubernetes Deployment | ⬜ |
| **Infra** | Configure `readinessProbe` to `/health` endpoint | ⬜ |
| **Infra** | Set `terminationGracePeriodSeconds: 30` | ⬜ |
| **Network** | Enable Ingress Rate Limiting (e.g., NGINX/ALB) as Layer 1 defense | ⬜ |
| **Security** | Rotate `GODEL_KEY` via Secret Manager | ⬜ |
| **Metrics** | Verify Grafana dashboard is consuming `/metrics` | ⬜ |

---
