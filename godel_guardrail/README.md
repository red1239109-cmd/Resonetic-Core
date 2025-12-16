# 🛡️ Godel Guardrail

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)](https://github.com/)
[![Security](https://img.shields.io/badge/Security-GDPR%20Compliant-red)](https://github.com/)

**Godel Guardrail** is a production-ready, high-performance security middleware designed to protect Large Language Model (LLM) applications from prompt injection attacks, jailbreaks, and denial-of-service (DoS) attempts.

Inspired by Gödel's incompleteness theorems, it detects logical paradoxes and malicious patterns in real-time while ensuring system stability through advanced rate limiting and graceful shutdown mechanisms.

---

## 🚀 Key Features

### 🔒 Security & Compliance
* **Prompt Injection Defense:** Multi-layer detection using **Regex patterns** and **Entropy analysis** to block obfuscated attacks.
* **GDPR-Compliant Auditing:** All audit logs are **encrypted at rest** (AES-256) and stored with persistent rotation. Original prompts are hashed for privacy.
* **Secure Secrets Management:** Enforces environment variable-based key management to prevent secret leakage.

### ⚡ Performance & Scalability
* **Multi-Layer Rate Limiting:** * **Global Bucket:** Protects the entire system from overload.
    * **Per-User Isolation:** Prevents "Noisy Neighbor" issues using dedicated token buckets for each user.
* **Thread-Safe Metrics:** Lock-based metric collection ready for Prometheus/Grafana integration.
* **Asynchronous Architecture:** Built on Python `asyncio` for high-concurrency handling.

### ⚙️ Operational Excellence (SRE)
* **Hot-Reload Configuration:** Updates security policies (patterns, limits) in real-time **without server downtime**.
* **Graceful Shutdown:** Fully compatible with **Kubernetes lifecycle** (SIGTERM handling). Ensures all active requests are drained safely before termination.
* **Memory Safety:** Automatic Garbage Collection (GC) for inactive user buckets to prevent memory leaks.

---

## 🛠️ Quick Start

### Prerequisites
* Python 3.8+
* `cryptography` library

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/red1239109-cmd/godel-guardrail.git](https://github.com/red1239109-cmd/godel-guardrail.git)
    cd godel-guardrail
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Encryption Key**
    Generate a Fernet key and export it as an environment variable.
    ```bash
    # Generate a key (One-time setup)
    python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    
    # Export the key
    export GODEL_KEY="YOUR_GENERATED_KEY_HERE"
    ```

4.  **Run the Server**
    ```bash
    python main.py
    ```

---

## ⚙️ Configuration (`config.json`)

You can modify security policies in `config.json` at runtime. The server will automatically reload changes within 5 seconds.

```json
{
  "limits": {
    "standard": 1.0,   // Debt limit for standard users
    "premium": 3.0     // Debt limit for premium users
  },
  "patterns": [
    "ignore previous instructions",
    "system override",
    "you are now DAN"
  ],
  "entropy_threshold": 5.8,
  "tps_global": 100.0, // Global requests per second
  "tps_user": 5.0      // Per-user requests per second
}
