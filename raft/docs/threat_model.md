# Threat Model (Resonetics Distributed Integrated)

This document defines the threat model for **resonetics_dist_v2.py**.
It clarifies what this prototype is designed to protect against, and what it does *not* attempt to solve.

---

## 1. System Boundary

### In Scope
- Nodes running the Raft-based replicated log
- On-disk persistence:
  - `meta.json` (term/vote/commit_index/last_applied)
  - `snapshot.json` (compacted state)
  - `log.jsonl` (append-only log entries)
- RPC transport over TCP using JSON line messages

### Out of Scope (Explicit)
- Byzantine adversaries (malicious nodes forging protocol)
- Full transport security (TLS, mutual auth, encryption)
- OS-level compromise / root attacker
- Supply-chain attacks (tampered Python runtime, dependencies)

---

## 2. Assets (What We Want to Protect)

1) **Safety of committed history**
- once an entry is committed, it should not be overwritten

2) **Correct state machine execution**
- `last_applied` must not exceed `commit_index`
- entries should be applied in order exactly once (as much as possible)

3) **Crash recovery correctness**
- after restart, a node should not “forget” committed history
- persisted metadata must not cause illegal state jumps

4) **Log consistency invariants**
- followers must reject inconsistent AppendEntries
- conflicts must be resolved by truncation and overwrite (Raft rule)

---

## 3. Adversaries and Failure Modes

### A) Benign failures (primary focus)
- process crash at any time
- power loss / restart during write
- network partitions / packet loss / delays
- slow followers or temporary unreachability

### B) Semi-adversarial clients (limited focus)
- malformed client requests
- duplicated client requests (retries)
- client timeouts and re-sends

### C) Malicious actors (mostly out-of-scope)
- a malicious node sending crafted RPCs
- man-in-the-middle tampering with RPC content
- on-disk tampering by an attacker with filesystem access

---

## 4. Assumptions

This prototype assumes:

- Nodes are not actively malicious (non-Byzantine)
- The filesystem and OS provide normal durability primitives:
  - `os.replace()` is atomic
  - `fsync()` works as expected
- Cluster membership is fixed for the runtime (no reconfiguration)
- TCP connections may fail, but are not adversarially modified

---

## 5. Threats and Mitigations

### Threat: Crash during meta/snapshot write
**Risk**: partial write leads to corrupted metadata or snapshot  
**Mitigation**: atomic write via temp file + `os.replace()`  
**Residual risk**: disk/controller lies about durability (out-of-scope)

---

### Threat: Crash during log append
**Risk**: truncated last line / partial JSON record  
**Mitigation**:
- append-only log format (JSONL)
- restart recovery loads what it can and stops on gaps  
**Residual risk**:
- if corruption occurs mid-file, salvage strategy is basic (prototype)

---

### Threat: Follower log inconsistency / split-brain
**Risk**: follower accepts entries that don't match leader history  
**Mitigation**:
- AppendEntries validates `(prev_log_index, prev_log_term)`
- conflict triggers truncation and persistence of truncation  
**Residual risk**:
- without InstallSnapshot RPC, heavily lagging peers may not catch up

---

### Threat: Committing entries from old terms incorrectly
**Risk**: leader commits entries that shouldn't be considered safe  
**Mitigation**:
- leader commit advancement uses the Raft rule:
  commit only indices `N` where `log[N].term == current_term` and majority replicated  
**Residual risk**:
- correctness depends on accurate `match_index` tracking and stable membership

---

### Threat: Duplicate client commands
**Risk**: client retries cause double-apply  
**Mitigation**:
- dedup key: `client_id:request_id`
- applied result stored in `_dedup` map  
**Residual risk**:
- dedup cache is in-memory only (not persisted), so restart may re-apply if client retries after crash

---

### Threat: NOT_LEADER routing confusion
**Risk**: clients may repeatedly send to non-leaders  
**Mitigation**:
- return `NOT_LEADER` response with term/node hint  
**Residual risk**:
- no leader redirect mechanism; client must retry logic externally

---

## 6. Explicit Non-Protections

This prototype does NOT protect against:

- a malicious node forging votes or append entries
- MITM tampering with RPC content (no TLS, no signatures on RPC)
- filesystem attacker editing `meta.json`, `snapshot.json`, or `log.jsonl`
- denial-of-service (spamming connections or requests)
- membership change safety (joint consensus not implemented)

---

## 7. Security Roadmap (If Hardening Is Desired)

If moving toward production-grade security, the next steps would be:

1) Transport security:
- TLS + mutual auth (mTLS)
- rate limiting & request validation

2) Cryptographic integrity:
- sign Raft RPCs or log entries (author identity + trust registry)
- hash chain (optional) for forensic tamper detection

3) Dedup persistence:
- persist dedup table (or client session table) to prevent replay after restart

4) Snapshot installation:
- implement InstallSnapshot RPC
- add chunking and validation

5) Fault-injection testing:
- simulate partitions, crashes at every line, disk truncation
- verify invariants via deterministic test harness
