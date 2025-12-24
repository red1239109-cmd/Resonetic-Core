# Godel Guardrail Enterprise v10.1

Production-ready guardrail service with:
- FastAPI API (`/scan`, `/reload`)
- Prometheus metrics (`/metrics`)
- K8s health endpoint (`/health`)
- Defense-in-depth sanitization + plugin inspection
- Rate limiting (global + per-user)
- Encrypted audit logs (optional)
- Fail-open mode (optional)

## Quick Start (Docker)

```bash
docker build -t godel-guardrail:10.1 .
docker run --rm -p 8000:8000 godel-guardrail:10.1
Open:
Swagger UI: http://localhost:8000/docs
Health:     http://localhost:8000/health
Metrics:    http://localhost:8000/metrics
Environment Variables
GODEL_KEY (recommended)
Fernet key for encrypted audit logs.
If not provided, the service generates an ephemeral key (logs will still work, but cannot be decrypted later).
Example:
코드 복사
Bash
export GODEL_KEY="YOUR_FERNET_KEY_STRING"
docker run --rm -e GODEL_KEY="$GODEL_KEY" -p 8000:8000 godel-guardrail:10.1
Generate a key in Python:
코드 복사
Python
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
API
POST /scan
Request:
코드 복사
Json
{
  "prompt": "hello",
  "user_id": "alice",
  "tier": "standard"
}
Response:
코드 복사
Json
{ "safe": true, "code": "S200", "reason": "OK" }
Possible codes:
S200: Allowed
E400: Sanitizer blocked (XSS/script/too long)
E403: Blocked by plugins
429: Rate limited
503: Service shutting down or internal failure
POST /reload
Hot-reload configuration without restarting the service.
Request:
코드 복사
Json
{
  "limits": {"standard": 1.0, "premium": 3.0},
  "patterns": ["ignore previous", "system prompt"],
  "entropy": 5.8,
  "tps_g": 100.0,
  "tps_u": 5.0,
  "fail_open": false,
  "audit_encrypt": true
}
Response:
코드 복사
Json
{ "status": "reloaded" }
Prometheus
Scrape:
GET /metrics
Includes:
godel_requests_total{status,plugin,tier}
godel_latency_seconds{plugin}
godel_active_requests
godel_security_debt
Notes
This is a guardrail layer: it inspects text prompts for suspicious patterns/obfuscation.
Tune patterns, entropy, and limits based on your threat model and desired strictness.
