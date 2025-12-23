# 📘 Godel Guardrail Deployment Guide  
**Version**: v10.1 Enterprise Edition  
**Target Audience**: DevOps, SRE, Security Architects

---

## 🏗️ 1. High Availability & Scalability
### ❗ Problem: Split-Brain Rate-Limit  
Current **in-memory TokenBucket** causes **split-brain** when scaling horizontally (e.g., 3 pods → 3× user quota).

### ✅ Solution: Distributed Rate-Limit (Redis + Lua)
**Architecture**: Replace in-memory bucket with **Redis** + **atomic Lua script**.

```python
# RedisTokenBucket (pseudo)
class RedisTokenBucket:
    def __init__(self, redis, key: str, rate: float, capacity: float):
        self.redis = redis
        self.key = key
        self.rate = rate
        self.capacity = capacity
        self.script = redis.register_script("""
            local key = KEYS[1]
            local rate = tonumber(ARGV[1])
            local capacity = tonumber(ARGV[2])
            local now = tonumber(ARGV[3])
            local ttl = math.ceil(capacity / rate)

            local last = redis.call('GET', key .. ':last') or now
            local tokens = tonumber(redis.call('GET', key .. ':tokens') or capacity)

            tokens = math.min(capacity, tokens + (now - last) * rate)
            local allowed = 0
            if tokens >= 1 then
                tokens = tokens - 1
                allowed = 1
            end

            redis.call('SET', key .. ':tokens', tokens, 'EX', ttl)
            redis.call('SET', key .. ':last', now, 'EX', ttl)
            return allowed
        """)
    
    async def allow(self) -> bool:
        now = time.time()
        return bool(await self.script.execute([self.key], [self.rate, self.capacity, now]))
```

**Benefits**  
- **Atomic** across cluster → **no split-brain**  
- **Millisecond-level** latency (Lua inside Redis)  
- **Horizontal scale** → add pods, **limit stays global**

---

### ✅ CPU Optimisation (Rust/Cython)
**Trigger**: CPU > 70 % per pod **sustained**.  
**Solution**: Rewrite `EntropyTrap.inspect()` & `RegexTrap.inspect()` in **Rust** (PyO3) or **Cython** to bypass GIL.

---

## 🛡️ 2. Security & Compliance (Bank-Grade)
### ❌ Never hard-code keys in env / yaml
```yaml
# ❌ BAD - visible in docker inspect
env:
  - name: GODEL_KEY
    value: "s3cr3t"
```

### ✅ Use Cloud KMS / Secret Manager
| Cloud | Service | IAM Pattern |
|---|---|---|
| AWS | **KMS** + **SSM Parameter Store** | IRSA (IAM Role for Service Accounts) |
| GCP | **Secret Manager** | Workload Identity |
| Azure | **Key Vault** | Pod Identity |
| K8s | **Vault Sidecar Injector** | Short-lived token |

**Flow**  
1. Pod starts → **fetches key from KMS** (in-memory only)  
2. **Never** written to disk or env  
3. Key rotation → **zero-downtime** (`/reload` endpoint)

---

### ✅ PII Protection (GDPR/CCPA)
**Dual-ID Strategy**  
| ID Type | Usage | Storage | Reversible |
|---|---|---|---|
| **Analytical ID** | metrics, stats | `SHA-256(user_id + salt)` | ❌ |
| **Audit ID** | legal evidence | `AES-256(user_id, audit_key)` | ✅ (warrant) |

**Code** (already in v10.1)
```python
# audit log
audit_clear = {"user": user_id, "safe": True, "debt": ctx.debt}
audit_cipher = cipher.encrypt(json.dumps(audit_clear).encode()).decode()
logger.info(f"AUDIT_ENC: {audit_cipher}")
```

---

## 📊 3. Observability & Monitoring
### ✅ Prometheus Label Hygiene
**❌ High Cardinality** (memory bomb)  
```python
# ❌ NEVER
requests_total{user_id="alice123"}   # 1 M time-series
```

**✅ Safe Labels**  
```python
# ✅ OK
requests_total{tier="premium", status="blocked", plugin="RegexTrap"}
```

**User-level detail** → ship to **ELK/Loki**, **not Prometheus**.

---

### ✅ Alerting Rules (AlertManager)
| Severity | Condition | Action |
|---|---|---|
| **P0 Critical** | `up == 0` or `godel_health_status == 0` | Page on-call |
| **P1 High** | `rate(godel_security_debt[5m]) > 0.8 * limit` | Slack + ticket |
| **P2 Warn** | `rate(process_cpu_seconds_total[5m]) > 0.8` | Auto-scale HPA |

---

## ✅ 4. Production Checklist
| Category | Item | Status |
|---|---|---|
| **Infra** | Min **3 replicas** (K8s Deployment) | ⬜ |
| **Infra** | `readinessProbe: /health` | ⬜ |
| **Infra** | `terminationGracePeriodSeconds: 30` | ⬜ |
| **Network** | **Ingress rate-limit** (NGINX / ALB) | ⬜ |
| **Security** | **KMS** key rotation (no env vars) | ⬜ |
| **Obs** | **Grafana** dashboard consumes `/metrics` | ⬜ |
| **Obs** | **AlertManager** rules applied | ⬜ |

---

## 📄 License
MIT → **Enterprise-friendly**

---

**Godel Guardrail** – *"Reason & Security in one loop."*
```
