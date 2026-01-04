# Integration Guide: Resonetics -Anchor

This guide outlines how to successfully implant the Resonetics Anchor kernel into your high-assurance systems or autonomous agent environments.

## 1. Prerequisites

Resonetics is designed for zero-dependency operation using the Python Standard Library.

* **Runtime**: Python 3.8 or higher (requires `future annotations` and advanced `typing` support).
* **Database**: SQLite3 (included in Python's standard library).
* **OS**: Cross-platform (Linux, macOS, Windows).

## 2. Deployment Strategies

### A. Direct Implantation (Recommended)

Since Resonetics is a micro-kernel, we recommend placing the source file directly into your project's core directory rather than installing it as an external package.

```text
your_project/
├── core/
│   └── resonetics.py  <-- Place v22.1.2-Anchor here
├── main.py
└── ...

```

### B. Module Import

Initialize the `SovereignBody` within your application's entry point:

```python
from core.resonetics import SovereignBody

# Initialize the kernel
with SovereignBody(db_path="akashic_record.db") as kernel:
    # Your system logic here
    pass

```

## 3. Configuration & Physics Limits

Resonetics enforces physical laws through the `SovereignBody` constructor. You should tune these limits based on your system's hardware and reliability requirements:

| Parameter | Default | Description |
| --- | --- | --- |
| `payload_limit` | 500,000 | Max raw size of objects allowed for full snapshot. |
| `ops` (limits) | 50,000 | Max recursive operations per object processing. |
| `time` (limits) | 0.05s | Max CPU time allowed per serialization attempt. |

**Example of custom tuning:**

```python
kernel = SovereignBody(
    db_path="production.db",
    payload_limit=1_000_000,  # 1MB Limit
)
# Access instance limits
kernel.limits["ops"] = 100_000 

```

## 4. Environment Contexts

### Testing & Determinism

To enable the **Deterministic Context** (Mock Time and predictable Nonce generation), set the following environment variable before running your suite:

```bash
export RESONETICS_TESTING=1

```

This ensures that `get_time()` increments by exactly 1 microsecond per call, and all hashes remain canonical across different machines.

### Multiverse Isolation

If you are running multiple instances or simulations simultaneously, use the URI-based memory path to prevent database collision:

```python
# Each call generates a unique URI: file:reso_akashic_{nonce}?mode=memory
kernel = SovereignBody(db_path=":memory:")

```

## 5. Verifying the Implantation

Every Resonetics Anchor file includes an integrated Audit Rig. After placing the file, verify its integrity by executing it directly:

```bash
python3 core/resonetics.py

```

**Expected Output:**

```text
--- [Resonetics v22120-Anchor] Baseline Audit Rig ---
[PASS] Proof 1: Atomic Chronos Sync Verified.
[PASS] Proof 2: Schema SLOT_TYPES Law Verified.
...
--- [Anchor] Verified. ---

```

---

### Security Note

Resonetics creates a `.db` file (SQLite) to store the immutable forensic chain. Ensure that the process has **Write permissions** in the target directory and that the database file is included in your backup/security rotations.

Would you like to add a specific section for **Docker/Container** deployment or **Cloud SQL** integration?
