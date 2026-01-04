# Technical Specification: Resonetics Anchor

## 1. Executive Summary

**Resonetics-Anchor** is a high-assurance judicial micro-kernel designed for autonomous agent governance and high-integrity environments. Unlike standard logging libraries, Resonetics treats data as **Forensic Evidence**. It enforces strict "physical laws" upon the software to ensure that every observation is deterministic, immutable, and accountable under a formal judicial framework.

## 2. Core Architectural Pillars

### 2.1 Zero-Execution Stasis (ZES)

To protect the kernel from malicious or unstable objects, Resonetics employs a "Zero-Execution" policy during state capture.

* **Mechanism**: The kernel bypasses standard Python methods like `__repr__` or `__str__`, which can be exploited via "Toxic Object Attacks" (infinite recursion or side-effect execution during printing).
* **Implementation**: It utilizes `_get_static_raw` to capture the raw session identity and memory state, ensuring the observer (Kernel) remains unaffected by the observed (Object).

### 2.2 Thermodynamic Resource Budgeting

The kernel treats computational resources as finite physical quantities. Every operation carries a "tax" that depletes a pre-allocated budget.

* **Operational Budget (`_PHYSICS_OP_LIMIT`)**: Limits the maximum number of recursive steps allowed per object.
* **Time Budget (`_PHYSICS_TIME_BUDGET`)**: Enforces a strict CPU time limit for serialization.
* **Graceful Degradation**: If a budget is exhausted, the system does not crash. Instead, it triggers **Forensic Sampling** (`_SAMPLED_BY_SIZE` or `_SAMPLED_BY_TIME`), preserving the head and tail of the data while flagging the record for review.

### 2.3 Deterministic Canonicalization

To achieve a "Single Version of Truth," Resonetics eliminates the inherent entropy of the Python runtime.

* **Stable Sorting**: Unordered structures (sets, dictionaries) are forcibly sorted using a deterministic key generated from content hashes.
* **Collision Preservation**: In the event of a hash collision, the kernel appends a unique sequence (`#C1`, `#C2`) to prevent data overwriting, ensuring 100% data retention.

## 3. Forensic Indicator Lexicon

The `obs_kind` field acts as a "Judicial Verdict" on the integrity of the captured data:

| Indicator | Meaning |
| --- | --- |
| **`_SNAPOK`** | Maximum Integrity. Data is perfectly preserved and passed hash verification. |
| **`_SNAPFAIL`** | Integrity Compromised. Format errors or hash mismatches detected. |
| **`_TRUNC`** | Data Truncated. Resource limits prevented a full capture. |
| **`_SCAN_INCOMPLETE`** | Complexity Alert. The deep scanner hit a depth limit before finishing. |

## 4. Concurrency & Locking Jurisprudence

To prevent deadlocks in multi-threaded environments, the kernel enforces a strict **Lock Hierarchy**:

1. **`DB_LOCK`**: Authority to write to the persistent ledger.
2. **`REBIRTH_LOCK`**: Authority to reinitialize or restructure system schema.
3. **`RULING_LOCK`**: Authority to issue a final Forensic Ruling.

**The Golden Rule**: A process holding a higher-level lock may request a lower-level lock, but requesting a higher-level lock from a lower state is strictly forbidden and will trigger a `Hierarchy Violation`.

## 5. Deployment Mandate: No Compression

**"Logic is Evidence."**
The internal logic of Resonetics is architected for transparency. Performance-oriented "code compression," summarization, or the use of opaque libraries is **strictly prohibited**. Every conditional branch and explicit variable must remain visible to ensure a clear forensic audit trail of how a "Ruling" was reached.

---

### How to use this document

1. **File Placement**: Save this as `TECHNICAL_SPEC.md` in your repository root.
2. **Developer Onboarding**: Require all contributors to read this before touching the `v22.1.2-Anchor` source file.
3. **Auditing**: Use the "Forensic Indicator Lexicon" as a guide when analyzing the `observation_log` table in your database.

