# Resonetics Architecture (Distributed Integrated)

This document describes the architecture of **resonetics_dist_v2.py**, a research-grade prototype that combines:

1) **A truth-preserving event model** (append-only log + verifiable history mindset)  
2) **Raft-style replication and leader election** (consensus for a replicated log)

This is a single-file integrated prototype intended for clarity and experimentation.

---

## Goals

Resonetics (Distributed Integrated) aims to provide:

- **Durable history**: state changes are represented as log entries
- **Replicated correctness**: entries are replicated across nodes using Raft rules
- **Auditable progress**: nodes can restart and resume from persisted metadata, snapshot, and log
- **Deterministic replay**: applied state can be reconstructed from snapshot + log

---

## Non-Goals (Explicitly Out of Scope)

- Byzantine fault tolerance (malicious nodes)
- Membership changes (joint consensus) beyond placeholders
- Snapshot installation RPC (InstallSnapshot) is not implemented
- Production-grade transport security (TLS, authn/authz)
- Strong backpressure and flow-control tuning

---

## High-Level Components

### 1) Persistent Store (`PersistStore`)

Disk layout per node:

reson_data/<node_id>/ meta.json        # term, voted_for, commit_index, last_applied snapshot.json    # compacted state + cluster config log.jsonl        # append-only log entries (JSON lines)

Persistence methods:

- `append_log_entry()`: append new log entry line
- `rewrite_log_from()`: rewrite compacted log after snapshot or truncation
- `save_snapshot()`: write snapshot atomically
- `save_meta()`: write meta atomically

Durability:
- atomic writes use `*.tmp` + `os.replace()`
- log entries can be fsynced per append (`FSYNC_EVERY_APPEND`)

---

### 2) Raft Node (`RaftNode`)

A node owns:

- **Persistent state**
  - `current_term`
  - `voted_for`
  - `log[]` (entries after snapshot base)
- **Volatile state**
  - `commit_index`
  - `last_applied`
- **Leader-only state**
  - `next_index[peer]`
  - `match_index[peer]`

Roles:
- `FOLLOWER`
- `CANDIDATE`
- `LEADER`

---

## Log Model

Each log entry:

```json
{
  "term": 3,
  "index": 42,
  "type": "CLIENT_CMD",
  "data": { ... }
}

Important invariants:

indexes are strictly increasing

the log is truncated on conflicts (follower-side) when term mismatch occurs

the leader drives replication using next_index and match_index



---

RPCs and Message Flow

All RPCs are JSON line messages over TCP.

PreVote (optional safety)

candidate asks peers whether an election would likely succeed

reduces disruption in some cases


RequestVote

candidate requests votes for a new term

vote granted only if candidate log is “up-to-date”


AppendEntries

leader sends heartbeats and log entries

follower validates (prev_log_index, prev_log_term)

on conflict, follower truncates its log and persists truncation


ClientRequest

client sends an operation to a node

if node is not leader, it returns NOT_LEADER

leader appends a CLIENT_CMD entry and replicates it

client waits until entry is committed (or timeout)



---

Commit Advancement (Leader Rule)

Leader advances commit_index by the Raft rule:

Find N > commit_index such that:

log[N].term == current_term

a majority of voters have match_index[i] >= N


Then set commit_index = N.

This is implemented in:

_advance_commit_index_locked()



---

Apply Loop and State Machine

Committed entries are applied in-order:

last_applied advances up to commit_index

applied CLIENT_CMD updates the in-memory kv state machine

deduplication map _dedup caches results by client_id:request_id


Persistence:

last_applied and commit_index are persisted in meta.json

this enables correct restart behavior and prevents re-applying beyond committed state



---

Snapshotting and Log Compaction

A snapshot is taken after N applied entries:

SNAPSHOT_EVERY_APPLIED


Snapshot includes:

last_included_index

last_included_term

state (kv)

cluster config


Compaction behavior:

log is rewritten to keep only entries after the snapshot base

a SNAP_DUMMY entry anchors the compacted log


Note:

InstallSnapshot RPC is not implemented, so snapshot distribution is out-of-scope.



---

Safety Properties (What This Prototype Preserves)

Election safety (single leader per term, as long as assumptions hold)

Log matching property through (prev_log_index, prev_log_term) checks

Leader completeness for committed entries via majority match_index rule

Crash recovery with persisted (term, vote, commit_index, last_applied) and snapshot base



---

Known Limitations / Next Steps

1. Implement InstallSnapshot RPC to support peers lagging behind compacted logs


2. Add membership change via joint consensus (currently placeholder)


3. Strengthen network layer (timeouts, retries, backpressure, TLS)


4. Integrate cryptographic author/trust signing into Raft entries (optional next stage)


5. Add deterministic testing and fault injection harness




---

Glossary

term: election epoch

index: log position

commit_index: highest log index known to be committed

last_applied: highest log index applied to state machine

match_index: highest index replicated on a follower (leader tracked)

next_index: next log index leader should send to a follower
