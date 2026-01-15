Resonetics Protocol Specification

Version: 3.x (Draft)
Status: Research / Reference Implementation
Repository: https://github.com/red1239109-cmd/resonetics


---

1. Purpose

Resonetics defines a truth-preserving, append-only protocol for distributed systems.

It is not a consensus protocol by itself, but a substrate that ensures:

explicit authorship,

verifiable history,

replayable state,

and continuous auditability


even under partial failure, restarts, or adversarial conditions.

This document specifies the wire-level RPC protocol used by Resonetics nodes.


---

2. Design Constraints (Normative)

All messages are stateless RPC calls over TCP.

Each request maps to exactly one response.

Messages are encoded as single-line JSON (UTF-8, \n terminated).

Nodes may crash, restart, or rejoin at any time.

Safety > Liveness (the system may stall rather than corrupt truth).



---

3. Message Envelope (Common)

All messages MUST follow this envelope:

{
  "kind": "MessageType",
  "payload": { ... },
  "t": 1700000000000,
  "v": "v3.10.0"
}

Fields

Field	Type	Description

kind	string	RPC message type
payload	object	Message-specific data
t	integer	Sender timestamp (ms since epoch)
v	string	Application / protocol version



---

4. Node Roles

Nodes operate in exactly one role at a time:

FOLLOWER

CANDIDATE

LEADER


Role transitions follow Raft-style rules, but Resonetics emphasizes verifiability over optimization.


---

5. RPC Messages

5.1 PreVote

Used to probe election viability without incrementing term.

Request

{
  "term": 3,
  "candidate_id": "n2",
  "last_log_index": 42,
  "last_log_term": 3
}

Response

{
  "term": 3,
  "vote_granted": true
}

Rules

Receiver MUST NOT change persistent state.

Vote is granted only if candidate log is at least as up-to-date.



---

5.2 RequestVote

Formal election request.

Request

{
  "term": 4,
  "candidate_id": "n2",
  "last_log_index": 42,
  "last_log_term": 3
}

Response

{
  "term": 4,
  "vote_granted": true
}

Rules

A node votes for at most one candidate per term.

Vote persistence is REQUIRED before responding true.



---

5.3 AppendEntries

Used for both log replication and heartbeats.

Request

{
  "term": 4,
  "leader_id": "n1",
  "prev_log_index": 41,
  "prev_log_term": 3,
  "entries": [
    {
      "term": 4,
      "index": 42,
      "type": "CLIENT_CMD",
      "data": { ... }
    }
  ],
  "leader_commit": 40
}

Response

{
  "term": 4,
  "success": true,
  "match_index": 42
}

Rules

Strict log matching is enforced.

On conflict, followers MUST truncate conflicting suffix.

leader_commit is advisory; followers clamp safely.



---

6. Client Interaction

6.1 ClientRequest

Clients may only submit commands to the current leader.

Request

{
  "client_id": "clientA",
  "request_id": "abc123",
  "op": "kv_set",
  "args": {
    "k": "x",
    "v": 10
  }
}

Response (Success)

{
  "ok": true,
  "code": "OK",
  "result": { "ok": true },
  "dedup": false
}

Response (Not Leader)

{
  "ok": false,
  "code": "NOT_LEADER",
  "result": {
    "term": 4,
    "node": "n2"
  }
}

Rules

Requests are idempotent via (client_id, request_id).

Duplicate requests MUST NOT be re-applied.



---

7. Log Entry Semantics

Each log entry contains:

{
  "term": 4,
  "index": 42,
  "type": "CLIENT_CMD",
  "data": { ... }
}

Entry Types

Type	Meaning

GENESIS	Initial anchor
NOOP	Leadership confirmation
CLIENT_CMD	Deterministic state transition
SNAP_DUMMY	Snapshot boundary marker



---

8. Commit Rule (Safety-Critical)

A leader may advance commit_index only if:

Entry term == leader’s current term

A majority of voters have replicated the entry


This rule is mandatory to prevent stale-term commitment.


---

9. Snapshot & Compaction

Snapshots occur after a configurable number of applied entries.

Snapshot includes:

last included index

last included term

materialized state

cluster configuration


Logs before snapshot index are discarded.



---

10. Failure Semantics

Network failure → retry / election

Node restart → recover from disk

Partial write → detected via log invariants

Corruption → verification failure halts progress


Silence is preferred to false truth.


---

11. Non-Goals (Explicit)

Resonetics does not attempt to provide:

Byzantine fault tolerance

Optimized throughput

Dynamic sharding

Implicit trust


These may be layered above, not assumed within.


---

12. Implementation Status

This protocol is fully implemented in:

resonetics_dist_v2.py

The implementation serves as a reference, not the only valid realization.


---

13. Closing Principle

> Truth is not agreed upon.

Truth is verified, replayed, and survived.
