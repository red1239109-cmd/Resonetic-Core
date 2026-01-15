# Resonetics

**Resonetics** is a cryptographically verifiable, append-only event ledger
designed to study how *truth, identity, and integrity* can be preserved
in long-running and distributed systems.

It is not a blockchain.
It is not a consensus protocol.

Resonetics is a **truth-preserving substrate** upon which simulation,
governance, auditing, or historical reasoning systems can be built.

---

## Motivation

Modern systems often assume correctness once data is written.
In reality, long-lived systems fail in quieter ways:

- logs drift
- authorship becomes ambiguous
- partial corruption goes unnoticed
- verification is skipped for performance reasons
- services mutate privately behind APIs

Resonetics explores a different stance:

> Truth is not assumed.  
> Truth is *continuously verified*.

---

## Core Ideas

### 1. Append-only, cryptographically linked state

All state transitions are written as immutable events.
Each entry is cryptographically chained to the previous one.

Silent corruption becomes detectable by design.

---

### 2. Explicit authorship

Every ledger entry carries a signed `author` identity:

```json
"author": {
  "id": "node-A",
  "key_fp": "ab12cd34ef56"
}

Authorship is part of the signed payload, not metadata. Forgery is detectable, not inferred.


---

3. Declared trust, not implicit trust

Trusted peers are loaded from an explicit trust registry. Verification selects the correct secret key per author.

There is no global assumption of trust.


---

4. Continuous verification

The ledger can be verified:

incrementally

under time budgets

after partial progress

at any moment during runtime


Verification is designed as an ongoing process, not a startup ritual.


---

5. Network-aware openness (AGPL-3.0)

Resonetics is intended to run as a networked system.

If it is used to provide a service over a network, modifications to the core must remain open.

This prevents private forks of truth-critical infrastructure while still enabling research and collaboration.


---

What Resonetics Is Not

❌ Not a blockchain

❌ Not a consensus algorithm

❌ Not optimized for throughput

❌ Not designed to replace databases


Consensus, replication, and governance are deliberately left out so they can be studied on top of a verifiable substrate.


---

Architecture Overview

┌────────────┐
│   Author   │
│ (Node ID)  │
└─────┬──────┘
      │ signed event
      ▼
┌────────────┐
│   Ledger   │  ← append-only
│ (Event Log)│
└─────┬──────┘
      │ verification
      ▼
┌────────────┐
│  Validator │
│ (Chain +   │
│  Signature)│
└────────────┘


---

Intended Use Cases

Distributed simulations

Historical or philosophical system modeling

Integrity-first logging and auditing

Research into long-term system truth

Experimental governance or rule engines



---

License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

Why AGPL?

Resonetics is designed to run as a networked service. If its core is modified and used to serve users, those modifications should remain open.

This preserves collaborative research and prevents silent privatization of truth-critical systems.

See LICENSE for details.


---

Status

This is a research-grade prototype.

The focus is correctness, auditability, and explicit guarantees — not convenience or performance.

The project is expected to evolve. Breaking changes may occur.


---

Author

red1239109-cmd

If you are exploring integrity, truth, or long-running systems and want to collaborate, discussion is welcome.

