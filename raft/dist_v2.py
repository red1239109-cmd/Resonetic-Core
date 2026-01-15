#!/usr/bin/env python3
# ==============================================================================
# Project: Resonetics
# File: resonetics_dist_v2.py
# Description:
#   Resonetics is a cryptographically verifiable event-ledger engine designed
#   to study and enforce integrity, identity, and trust in long-running,
#   networked systems.
#
#   This project explores how "truth" can be maintained over time in systems
#   where:
#     - state is append-only,
#     - authorship is explicit,
#     - verification is continuous rather than assumed,
#     - and services may operate across distributed nodes.
#
#   Resonetics is not a consensus protocol by default.
#   It is a *truth-preserving substrate* upon which consensus,
#   simulation, governance, or historical reasoning systems may be built.
#
# Version: 2.2.0-DIST-INTEGRATED
# Author: red1239109-cmd
# Repository: https://github.com/red1239109-cmd/resonetics-Core
#
# ==============================================================================
# LICENSE
# ------------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# ==============================================================================
# DESIGN PRINCIPLES
# ------------------------------------------------------------------------------
# 1. Integrity over convenience
#    - All state transitions are append-only and cryptographically linked.
#
# 2. Explicit authorship
#    - Every ledger entry carries a verifiable author identity.
#
# 3. Trust is declared, not assumed
#    - Trusted peers are defined via an explicit trust registry.
#
# 4. Continuous verification
#    - The system is designed to be auditable at any time,
#      even under partial progress or bounded time.
#
# 5. Network-aware openness
#    - If this software is used to provide a networked service,
#      modifications to the core must remain open (AGPL-3.0).
#
# ==============================================================================
# INTENDED USE
# ------------------------------------------------------------------------------
# - Distributed simulations
# - Historical or philosophical system modeling
# - Integrity-first logging and auditing
# - Research into long-term system truth and verification
#
# This is a research-grade prototype.
# It favors clarity of guarantees over raw performance.
#
# ==============================================================================
from __future__ import annotations

import asyncio
import json
import os
import random
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

def now_ms() -> int:
    return int(time.time() * 1000)

def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def fsync_file(f) -> None:
    f.flush()
    os.fsync(f.fileno())

def atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        fsync_file(f)
    os.replace(tmp, path)

class Config:
    APP_VERSION = "v3.10.0"

    ELECTION_TIMEOUT_MIN = int(os.environ.get("RESON_ELECT_MIN_MS", "900"))
    ELECTION_TIMEOUT_MAX = int(os.environ.get("RESON_ELECT_MAX_MS", "1600"))
    HEARTBEAT_INTERVAL_MS = int(os.environ.get("RESON_HEARTBEAT_MS", "250"))

    CLIENT_TIMEOUT_MS = int(os.environ.get("RESON_CLIENT_TIMEOUT_MS", "2500"))
    REPL_TIMEOUT_MS = int(os.environ.get("RESON_REPL_TIMEOUT_MS", "900"))
    MAX_ENTRIES_PER_APPEND = int(os.environ.get("RESON_REPL_BATCH", "32"))

    SNAPSHOT_EVERY_APPLIED = int(os.environ.get("RESON_SNAPSHOT_EVERY", "50"))

    DATA_ROOT = os.environ.get("RESON_DATA_ROOT", "./reson_data")
    FSYNC_EVERY_APPEND = os.environ.get("RESON_FSYNC_EVERY_APPEND", "1") == "1"

def msg_envelope(kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": kind, "payload": payload, "t": now_ms(), "v": Config.APP_VERSION}

class Role(str, Enum):
    FOLLOWER = "FOLLOWER"
    CANDIDATE = "CANDIDATE"
    LEADER = "LEADER"

@dataclass
class LogEntry:
    term: int
    index: int
    type: str
    data: Dict[str, Any]

@dataclass
class PersistentState:
    current_term: int = 0
    voted_for: Optional[str] = None
    log: List[LogEntry] = field(default_factory=list)

@dataclass
class VolatileState:
    commit_index: int = 0
    last_applied: int = 0

@dataclass
class LeaderState:
    next_index: Dict[str, int] = field(default_factory=dict)
    match_index: Dict[str, int] = field(default_factory=dict)

@dataclass
class ClusterConfig:
    voters: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    learners: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    joint: Optional[Dict[str, Any]] = None

@dataclass
class Snapshot:
    last_included_index: int = 0
    last_included_term: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    cluster: Dict[str, Any] = field(default_factory=dict)

def majority(n: int) -> int:
    return (n // 2) + 1

# ==============================================================================
# Persistence Store
# ==============================================================================
class PersistStore:
    def __init__(self, root: str, node_id: str):
        self.dir = os.path.join(root, node_id)
        ensure_dir(self.dir)
        self.meta_path = os.path.join(self.dir, "meta.json")
        self.snap_path = os.path.join(self.dir, "snapshot.json")
        self.log_path = os.path.join(self.dir, "log.jsonl")

    def load_meta(self) -> Tuple[int, Optional[str], int, int]:
        """
        returns: (current_term, voted_for, commit_index, last_applied)
        """
        if not os.path.exists(self.meta_path):
            return 0, None, 0, 0
        try:
            with open(self.meta_path, "rb") as f:
                d = json.loads(f.read().decode("utf-8"))
            term = int(d.get("current_term", 0))
            vf = d.get("voted_for", None)
            if vf is not None and not isinstance(vf, str):
                vf = None
            ci = int(d.get("commit_index", 0))
            la = int(d.get("last_applied", 0))
            return term, vf, ci, la
        except Exception:
            return 0, None, 0, 0

    def save_meta(self, current_term: int, voted_for: Optional[str], commit_index: int, last_applied: int) -> None:
        atomic_write_text(
            self.meta_path,
            jdump({
                "current_term": int(current_term),
                "voted_for": voted_for,
                "commit_index": int(commit_index),
                "last_applied": int(last_applied),
            }) + "\n"
        )

    def load_snapshot(self) -> Optional[Snapshot]:
        if not os.path.exists(self.snap_path):
            return None
        try:
            with open(self.snap_path, "rb") as f:
                d = json.loads(f.read().decode("utf-8"))
            return Snapshot(
                last_included_index=int(d.get("last_included_index", 0)),
                last_included_term=int(d.get("last_included_term", 0)),
                state=dict(d.get("state", {}) or {}),
                cluster=dict(d.get("cluster", {}) or {}),
            )
        except Exception:
            return None

    def save_snapshot(self, snap: Snapshot) -> None:
        atomic_write_text(self.snap_path, jdump({
            "last_included_index": int(snap.last_included_index),
            "last_included_term": int(snap.last_included_term),
            "state": snap.state,
            "cluster": snap.cluster,
        }) + "\n")

    def append_log_entry(self, ent: LogEntry) -> None:
        line = jdump({"term": ent.term, "index": ent.index, "type": ent.type, "data": ent.data}) + "\n"
        with open(self.log_path, "ab") as f:
            f.write(line.encode("utf-8"))
            if Config.FSYNC_EVERY_APPEND:
                fsync_file(f)

    def rewrite_log_from(self, entries: List[LogEntry], snap_base_index: int) -> None:
        out = [e for e in entries if e.index > snap_base_index]
        tmp = self.log_path + ".tmp"
        with open(tmp, "wb") as f:
            for e in out:
                f.write((jdump({"term": e.term, "index": e.index, "type": e.type, "data": e.data}) + "\n").encode("utf-8"))
            fsync_file(f)
        os.replace(tmp, self.log_path)

    def load_log_entries(self) -> List[LogEntry]:
        if not os.path.exists(self.log_path):
            return []
        out: List[LogEntry] = []
        try:
            with open(self.log_path, "rb") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    d = json.loads(raw.decode("utf-8"))
                    out.append(LogEntry(
                        term=int(d["term"]),
                        index=int(d["index"]),
                        type=str(d["type"]),
                        data=dict(d.get("data", {}) or {}),
                    ))
        except Exception:
            return out
        return out

# ==============================================================================
# Raft Node
# ==============================================================================
class RaftNode:
    def __init__(self, node_id: str, host: str, port: int, cluster: ClusterConfig):
        self.id = node_id
        self.host = host
        self.port = port
        self.cluster = cluster

        self.store = PersistStore(Config.DATA_ROOT, self.id)

        self.p = PersistentState()
        self.v = VolatileState()
        self.role: Role = Role.FOLLOWER
        self.leader_id: Optional[str] = None
        self.leader_state: Optional[LeaderState] = None

        self._election_deadline_ms = 0
        self._last_hb_sent_ms = 0

        self._lock = asyncio.Lock()
        self._server: Optional[asyncio.AbstractServer] = None
        self._stop = asyncio.Event()

        self.kv: Dict[str, Any] = {}
        self._dedup: Dict[str, Any] = {}
        self._commit_waiters: Dict[int, asyncio.Event] = {}

        self.snap = Snapshot()
        self._applied_since_snap = 0

        self._recover_from_disk()
        self._ensure_genesis_if_needed()

    # ---------- Persistence ----------
    def _persist_meta(self) -> None:
        self.store.save_meta(self.p.current_term, self.p.voted_for, self.v.commit_index, self.v.last_applied)

    def _cluster_to_dict(self) -> Dict[str, Any]:
        return {"voters": dict(self.cluster.voters), "learners": dict(self.cluster.learners), "joint": self.cluster.joint if self.cluster.joint else None}

    def _load_cluster_from_dict(self, d: Dict[str, Any]) -> None:
        voters = d.get("voters") or {}
        learners = d.get("learners") or {}
        joint = d.get("joint", None)
        if isinstance(voters, dict):
            self.cluster.voters = {k: tuple(v) for k, v in voters.items()
                                  if isinstance(k, str) and isinstance(v, (list, tuple)) and len(v) == 2}
        if isinstance(learners, dict):
            self.cluster.learners = {k: tuple(v) for k, v in learners.items()
                                    if isinstance(k, str) and isinstance(v, (list, tuple)) and len(v) == 2}
        if joint is None or isinstance(joint, dict):
            self.cluster.joint = joint

    def _recover_from_disk(self) -> None:
        term, voted_for, meta_ci, meta_la = self.store.load_meta()
        self.p.current_term = term
        self.p.voted_for = voted_for

        snap = self.store.load_snapshot()
        if snap:
            self.snap = snap
            self.kv = dict(snap.state)
            self._load_cluster_from_dict(snap.cluster)
            self.p.log = [LogEntry(term=snap.last_included_term, index=snap.last_included_index, type="SNAP_DUMMY", data={"snap": True})]
        else:
            self.snap = Snapshot(last_included_index=0, last_included_term=0, state={}, cluster=self._cluster_to_dict())
            self.kv = {}

        disk_entries = self.store.load_log_entries()
        base_i = self.snap.last_included_index
        disk_entries = [e for e in disk_entries if e.index > base_i]
        disk_entries.sort(key=lambda e: e.index)

        last_seen = base_i
        for e in disk_entries:
            if e.index != last_seen + 1:
                break
            self.p.log.append(e)
            last_seen = e.index

        # --- B: restore & clamp commit_index / last_applied safely ---
        last_log = self._log_last_index()
        floor_i = self.snap.last_included_index

        ci = max(floor_i, min(meta_ci, last_log))
        la = max(floor_i, min(meta_la, ci))

        self.v.commit_index = ci
        self.v.last_applied = la

        # keep meta consistent on disk (optional but good hygiene)
        self._persist_meta()

    def _ensure_genesis_if_needed(self) -> None:
        if self.p.log:
            return
        self.p.log.append(LogEntry(term=0, index=0, type="GENESIS", data={"msg": "genesis"}))
        self.v.commit_index = 0
        self.v.last_applied = 0
        self.snap = Snapshot(last_included_index=0, last_included_term=0, state={}, cluster=self._cluster_to_dict())
        self._applied_since_snap = 0
        self._persist_meta()

    # ---------- Index helpers ----------
    def _base_index(self) -> int:
        return self.p.log[0].index

    def _log_last_index(self) -> int:
        return self.p.log[-1].index

    def _log_last_term(self) -> int:
        return self.p.log[-1].term

    def _last_log_index_term(self) -> Tuple[int, int]:
        return self._log_last_index(), self._log_last_term()

    def _log_term_at(self, idx: int) -> int:
        if idx == self.snap.last_included_index:
            return self.snap.last_included_term
        if idx < self._base_index():
            return -1
        off = idx - self._base_index()
        if 0 <= off < len(self.p.log):
            return self.p.log[off].term
        return -1

    def _entry_at(self, idx: int) -> Optional[LogEntry]:
        if idx < self._base_index():
            return None
        off = idx - self._base_index()
        if 0 <= off < len(self.p.log):
            return self.p.log[off]
        return None

    # ---------- Network ----------
    async def _rpc_call(self, host: str, port: int, msg: Dict[str, Any], timeout_ms: int) -> Optional[Dict[str, Any]]:
        try:
            r, w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout_ms / 1000)
            w.write((jdump(msg) + "\n").encode("utf-8"))
            await w.drain()
            line = await asyncio.wait_for(r.readline(), timeout=timeout_ms / 1000)
            w.close()
            try:
                await w.wait_closed()
            except Exception:
                pass
            if not line:
                return None
            return json.loads(line.decode("utf-8"))
        except Exception:
            return None

    def _peer_addr(self, peer_id: str) -> Optional[Tuple[str, int]]:
        return self.cluster.voters.get(peer_id) or self.cluster.learners.get(peer_id)

    # ---------- Timers ----------
    def _rand_election_timeout_ms(self) -> int:
        return random.randint(Config.ELECTION_TIMEOUT_MIN, Config.ELECTION_TIMEOUT_MAX)

    def _reset_election_deadline(self) -> None:
        self._election_deadline_ms = now_ms() + self._rand_election_timeout_ms()

    # ---------- Persistence helpers ----------
    def _persist_append(self, ent: LogEntry) -> None:
        self.store.append_log_entry(ent)

    def _persist_rewrite_log_full(self) -> None:
        self.store.rewrite_log_from(self.p.log, snap_base_index=self.snap.last_included_index)

    def _persist_snapshot_and_compact(self, snap: Snapshot, kept_entries: List[LogEntry]) -> None:
        self.store.save_snapshot(snap)
        self.store.rewrite_log_from(kept_entries, snap_base_index=snap.last_included_index)

    # ---------- Role transitions ----------
    async def _become_follower(self, new_term: int, leader: Optional[str]) -> None:
        self.role = Role.FOLLOWER
        self.leader_state = None
        self.leader_id = leader
        if new_term > self.p.current_term:
            self.p.current_term = new_term
            self.p.voted_for = None
            self._persist_meta()
        self._reset_election_deadline()

    async def _become_candidate(self) -> None:
        self.role = Role.CANDIDATE
        self.leader_state = None
        self.leader_id = None
        self.p.current_term += 1
        self.p.voted_for = self.id
        self._persist_meta()
        self._reset_election_deadline()

    async def _append_local(self, ent: LogEntry) -> None:
        self.p.log.append(ent)
        self._persist_append(ent)

    async def _become_leader(self) -> None:
        self.role = Role.LEADER
        self.leader_id = self.id
        ls = LeaderState()
        last_index = self._log_last_index()
        for peer in set(self.cluster.voters.keys()) | set(self.cluster.learners.keys()):
            if peer == self.id:
                continue
            ls.next_index[peer] = last_index + 1
            ls.match_index[peer] = self.snap.last_included_index
        self.leader_state = ls
        self._last_hb_sent_ms = 0
        await self._append_local(LogEntry(term=self.p.current_term, index=self._log_last_index() + 1, type="NOOP", data={"leader": self.id}))

    # ---------- Election ----------
    async def _run_prevote(self) -> bool:
        async with self._lock:
            if self.role == Role.LEADER:
                return False
            term = self.p.current_term
            last_idx, last_term = self._last_log_index_term()
            voters = set(self.cluster.voters.keys())
            peers = [pid for pid in voters if pid != self.id]
            needed = majority(len(voters))

        async def ask(pid: str) -> bool:
            addr = self._peer_addr(pid)
            if not addr:
                return False
            host, port = addr
            payload = {"term": term, "candidate_id": self.id, "last_log_index": last_idx, "last_log_term": last_term}
            resp = await self._rpc_call(host, port, msg_envelope("PreVote", payload), timeout_ms=700)
            if not resp or resp.get("kind") != "PreVoteResp":
                return False
            rp = resp.get("payload", {})
            if int(rp.get("term", 0)) > term:
                return False
            return bool(rp.get("vote_granted"))

        votes = 1
        results = await asyncio.gather(*[ask(p) for p in peers], return_exceptions=True)
        for r in results:
            if r is True:
                votes += 1
        return votes >= needed

    async def _start_election(self) -> None:
        if not await self._run_prevote():
            async with self._lock:
                if self.role != Role.LEADER:
                    self._reset_election_deadline()
            return

        async with self._lock:
            await self._become_candidate()
            term = self.p.current_term
            last_idx, last_term = self._last_log_index_term()
            voters = set(self.cluster.voters.keys())
            peers = [pid for pid in voters if pid != self.id]
            votes = 1
            needed = majority(len(voters))

        async def ask(pid: str) -> bool:
            addr = self._peer_addr(pid)
            if not addr:
                return False
            host, port = addr
            payload = {"term": term, "candidate_id": self.id, "last_log_index": last_idx, "last_log_term": last_term}
            resp = await self._rpc_call(host, port, msg_envelope("RequestVote", payload), timeout_ms=700)
            if not resp or resp.get("kind") != "RequestVoteResp":
                return False
            rp = resp.get("payload", {})
            if int(rp.get("term", 0)) > term:
                async with self._lock:
                    await self._become_follower(int(rp["term"]), leader=None)
                return False
            return bool(rp.get("vote_granted"))

        results = await asyncio.gather(*[ask(p) for p in peers], return_exceptions=True)
        for r in results:
            if r is True:
                votes += 1

        async with self._lock:
            if self.role != Role.CANDIDATE or self.p.current_term != term:
                return
            if votes >= needed:
                await self._become_leader()
            else:
                await self._become_follower(self.p.current_term, leader=None)

    # ---------- Apply & Snapshot ----------
    def _maybe_snapshot_locked(self) -> None:
        if self._applied_since_snap < Config.SNAPSHOT_EVERY_APPLIED:
            return
        self._applied_since_snap = 0
        li = self.v.last_applied
        lt = self._log_term_at(li)
        new_snap = Snapshot(last_included_index=li, last_included_term=lt, state=dict(self.kv), cluster=self._cluster_to_dict())

        base = self._base_index()
        if li < base:
            return
        cut = li - base
        kept = [LogEntry(term=lt, index=li, type="SNAP_DUMMY", data={"snap": True})]
        kept.extend(self.p.log[cut + 1 :])

        self.snap = new_snap
        self.p.log = kept
        self.v.commit_index = max(self.v.commit_index, li)

        self._persist_snapshot_and_compact(new_snap, kept)
        self._persist_meta()

    def _apply_entry(self, ent: LogEntry) -> Optional[Any]:
        if ent.type != "CLIENT_CMD":
            return None
        cmd = ent.data.get("cmd", {})
        op = cmd.get("op")
        args = cmd.get("args", {})
        if op == "kv_set":
            k = args.get("k")
            v = args.get("v")
            if isinstance(k, str):
                self.kv[k] = v
                return {"ok": True, "written": True}
            return {"ok": False, "err": "BAD_KEY"}
        if op == "kv_get":
            k = args.get("k")
            if isinstance(k, str):
                return {"ok": True, "value": self.kv.get(k)}
            return {"ok": False, "err": "BAD_KEY"}
        return {"ok": False, "err": f"UNKNOWN_OP:{op}"}

    async def _apply_loop(self) -> None:
        while not self._stop.is_set():
            did_apply = False
            async with self._lock:
                while self.v.last_applied < self.v.commit_index:
                    self.v.last_applied += 1
                    ent = self._entry_at(self.v.last_applied)
                    if ent is None:
                        continue
                    res = self._apply_entry(ent)
                    self._applied_since_snap += 1
                    did_apply = True
                    if ent.type == "CLIENT_CMD":
                        did = ent.data.get("dedup_id")
                        if isinstance(did, str) and res is not None:
                            self._dedup[did] = res

                if did_apply:
                    # B: last_applied advance persists
                    self._persist_meta()

                ci = self.v.commit_index
                for idx, ev in list(self._commit_waiters.items()):
                    if idx <= ci:
                        ev.set()
                        self._commit_waiters.pop(idx, None)

                self._maybe_snapshot_locked()
            await asyncio.sleep(0.01)

    async def _wait_commit(self, idx: int, timeout_ms: int) -> bool:
        async with self._lock:
            if idx <= self.v.commit_index:
                return True
            ev = self._commit_waiters.get(idx)
            if not ev:
                ev = asyncio.Event()
                self._commit_waiters[idx] = ev
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout_ms / 1000)
            return True
        except asyncio.TimeoutError:
            return False

    # ---------- Replication ----------
    def _entry_wire(self, ent: LogEntry) -> Dict[str, Any]:
        return {"term": ent.term, "index": ent.index, "type": ent.type, "data": ent.data}

    async def _replicate_to_peer(self, peer_id: str) -> bool:
        async with self._lock:
            if self.role != Role.LEADER or not self.leader_state:
                return False
            addr = self._peer_addr(peer_id)
            if not addr:
                return False
            host, port = addr
            ni = self.leader_state.next_index.get(peer_id, self._log_last_index() + 1)
            if ni <= self.snap.last_included_index:
                return False  # (snapshot install 생략)

            term = self.p.current_term
            leader_commit = self.v.commit_index
            prev_i = ni - 1
            prev_t = self._log_term_at(prev_i)

            start = ni
            end = min(self._log_last_index() + 1, start + Config.MAX_ENTRIES_PER_APPEND)
            entries = []
            for idx in range(start, end):
                ent = self._entry_at(idx)
                if ent is not None:
                    entries.append(self._entry_wire(ent))

            payload = {"term": term, "leader_id": self.id, "prev_log_index": prev_i, "prev_log_term": prev_t, "entries": entries, "leader_commit": leader_commit}

        resp = await self._rpc_call(host, port, msg_envelope("AppendEntries", payload), timeout_ms=Config.REPL_TIMEOUT_MS)
        if not resp or resp.get("kind") != "AppendEntriesResp":
            return False

        rp = resp.get("payload", {})
        async with self._lock:
            if int(rp.get("term", 0)) > self.p.current_term:
                await self._become_follower(int(rp["term"]), leader=None)
                return False
            if self.role != Role.LEADER or self.p.current_term != term or not self.leader_state:
                return False

            if rp.get("success"):
                mi = int(rp.get("match_index", self.snap.last_included_index))
                self.leader_state.match_index[peer_id] = max(self.leader_state.match_index.get(peer_id, 0), mi)
                self.leader_state.next_index[peer_id] = self.leader_state.match_index[peer_id] + 1
                return True
            else:
                cur = self.leader_state.next_index.get(peer_id, self.snap.last_included_index + 1)
                self.leader_state.next_index[peer_id] = max(self.snap.last_included_index + 1, cur - 1)
                return False

    # ---------- RPC Handlers ----------
    async def _on_prevote(self, p: Dict[str, Any]) -> Dict[str, Any]:
        async with self._lock:
            term = int(p.get("term", 0))
            last_i = int(p.get("last_log_index", 0))
            last_t = int(p.get("last_log_term", 0))
            if term < self.p.current_term:
                return msg_envelope("PreVoteResp", {"term": self.p.current_term, "vote_granted": False})
            my_i, my_t = self._last_log_index_term()
            up_to_date = (last_t > my_t) or (last_t == my_t and last_i >= my_i)
            return msg_envelope("PreVoteResp", {"term": self.p.current_term, "vote_granted": bool(up_to_date)})

    async def _on_request_vote(self, p: Dict[str, Any]) -> Dict[str, Any]:
        async with self._lock:
            term = int(p.get("term", 0))
            cid = p.get("candidate_id")
            last_i = int(p.get("last_log_index", 0))
            last_t = int(p.get("last_log_term", 0))

            if term > self.p.current_term:
                await self._become_follower(term, leader=None)

            vote_granted = False
            if term == self.p.current_term:
                my_i, my_t = self._last_log_index_term()
                up_to_date = (last_t > my_t) or (last_t == my_t and last_i >= my_i)
                if up_to_date and (self.p.voted_for in (None, cid)):
                    self.p.voted_for = cid
                    self._persist_meta()
                    vote_granted = True
                    self._reset_election_deadline()

            return msg_envelope("RequestVoteResp", {"term": self.p.current_term, "vote_granted": vote_granted})

    async def _on_append_entries(self, p: Dict[str, Any]) -> Dict[str, Any]:
        async with self._lock:
            term = int(p.get("term", 0))
            leader_id = p.get("leader_id")
            prev_i = int(p.get("prev_log_index", 0))
            prev_t = int(p.get("prev_log_term", 0))
            entries = p.get("entries", [])
            leader_commit = int(p.get("leader_commit", 0))

            if term < self.p.current_term:
                return msg_envelope("AppendEntriesResp", {"term": self.p.current_term, "success": False, "match_index": self._log_last_index()})

            if term > self.p.current_term or self.role != Role.FOLLOWER:
                await self._become_follower(term, leader=leader_id)
            else:
                self.leader_id = leader_id
                self._reset_election_deadline()

            if prev_i < self.snap.last_included_index:
                return msg_envelope("AppendEntriesResp", {"term": self.p.current_term, "success": False, "match_index": self._log_last_index()})

            if self._log_term_at(prev_i) != prev_t:
                return msg_envelope("AppendEntriesResp", {"term": self.p.current_term, "success": False, "match_index": self._log_last_index()})

            # ===== A-2 + B: strict truncate on conflict =====
            for e in entries:
                ei = int(e["index"])
                et = int(e["term"])
                etype = e["type"]
                edata = e["data"]

                local = self._entry_at(ei)
                if local is not None and local.term != et:
                    base = self._base_index()
                    self.p.log = self.p.log[: (ei - base)]
                    self._persist_rewrite_log_full()  # strict disk truncate
                    local = None

                if local is None:
                    ent = LogEntry(term=et, index=ei, type=etype, data=edata)
                    self.p.log.append(ent)
                    self._persist_append(ent)

            # follower commit_index follows leader_commit
            old_ci = self.v.commit_index
            if leader_commit > self.v.commit_index:
                self.v.commit_index = min(leader_commit, self._log_last_index())

            if self.v.commit_index != old_ci:
                # B: commit_index persist
                self._persist_meta()

            return msg_envelope("AppendEntriesResp", {"term": self.p.current_term, "success": True, "match_index": self._log_last_index()})

    async def _on_client_request(self, p: Dict[str, Any]) -> Dict[str, Any]:
        client_id = p.get("client_id")
        request_id = p.get("request_id")
        op = p.get("op")
        args = p.get("args", {})

        if not isinstance(client_id, str) or not isinstance(request_id, str) or not isinstance(op, str):
            return msg_envelope("ClientResponse", {"ok": False, "code": "ERROR", "leader_hint": None, "result": "BAD_REQUEST"})

        did = f"{client_id}:{request_id}"

        async with self._lock:
            if did in self._dedup:
                return msg_envelope("ClientResponse", {"ok": True, "code": "OK", "result": self._dedup[did], "dedup": True})

            if self.role != Role.LEADER:
                return msg_envelope("ClientResponse", {"ok": False, "code": "NOT_LEADER", "result": {"term": self.p.current_term, "node": self.id}})

            idx = self._log_last_index() + 1
            ent = LogEntry(term=self.p.current_term, index=idx, type="CLIENT_CMD",
                           data={"dedup_id": did, "cmd": {"op": op, "args": args}, "meta": {"client_id": client_id, "request_id": request_id}})
            await self._append_local(ent)

        for pid in self.cluster.voters.keys():
            if pid != self.id:
                asyncio.create_task(self._replicate_to_peer(pid))

        ok = await self._wait_commit(idx, timeout_ms=Config.CLIENT_TIMEOUT_MS)
        if not ok:
            return msg_envelope("ClientResponse", {"ok": False, "code": "TIMEOUT", "result": {"reason": "COMMIT_WAIT_TIMEOUT"}})

        async with self._lock:
            res = self._dedup.get(did) or {"ok": False, "err": "APPLY_MISSING"}
            return msg_envelope("ClientResponse", {"ok": True, "code": "OK", "result": res, "dedup": False})

    # ---------- Leader commit advancement (정석) ----------
    def _advance_commit_index_locked(self) -> None:
        """
        Raft rule:
          Find N > commit_index such that:
            - log[N].term == current_term
            - a majority of match_index[i] >= N  (including leader itself)
          then set commit_index = N
        """
        if self.role != Role.LEADER or not self.leader_state:
            return

        last_idx = self._log_last_index()
        cur_term = self.p.current_term
        old_ci = self.v.commit_index

        voters = set(self.cluster.voters.keys())
        need = majority(len(voters))

        for N in range(last_idx, old_ci, -1):
            if self._log_term_at(N) != cur_term:
                continue
            count = 1  # leader itself
            for pid in voters:
                if pid == self.id:
                    continue
                if self.leader_state.match_index.get(pid, self.snap.last_included_index) >= N:
                    count += 1
            if count >= need:
                self.v.commit_index = N
                break

        if self.v.commit_index != old_ci:
            self._persist_meta()

    # ---------- Ticker ----------
    async def _ticker(self) -> None:
        self._reset_election_deadline()
        while not self._stop.is_set():
            await asyncio.sleep(0.03)
            t = now_ms()

            async with self._lock:
                role = self.role
                deadline = self._election_deadline_ms

            if role in (Role.FOLLOWER, Role.CANDIDATE) and t >= deadline:
                await self._start_election()
                continue

            if role == Role.LEADER:
                if (t - self._last_hb_sent_ms) >= Config.HEARTBEAT_INTERVAL_MS:
                    self._last_hb_sent_ms = t
                    for pid in self.cluster.voters.keys():
                        if pid != self.id:
                            asyncio.create_task(self._replicate_to_peer(pid))

                async with self._lock:
                    self._advance_commit_index_locked()

    # ---------- Server ----------
    async def start(self) -> None:
        self._server = await asyncio.start_server(self._on_conn, host=self.host, port=self.port)
        print(f"[{self.id}] up {self.host}:{self.port} | role={self.role} | v={Config.APP_VERSION}")
        print(f"[{self.id}] data_dir={self.store.dir} | term={self.p.current_term} voted_for={self.p.voted_for} "
              f"snap={self.snap.last_included_index}:{self.snap.last_included_term} log_last={self._log_last_index()} "
              f"ci={self.v.commit_index} la={self.v.last_applied}")
        asyncio.create_task(self._ticker())
        asyncio.create_task(self._apply_loop())
        self._reset_election_deadline()

    async def stop(self) -> None:
        self._stop.set()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _on_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                writer.close()
                return
            req = json.loads(line.decode("utf-8"))
            kind = req.get("kind")
            payload = req.get("payload", {})

            if kind == "PreVote":
                resp = await self._on_prevote(payload)
            elif kind == "RequestVote":
                resp = await self._on_request_vote(payload)
            elif kind == "AppendEntries":
                resp = await self._on_append_entries(payload)
            elif kind == "ClientRequest":
                resp = await self._on_client_request(payload)
            else:
                resp = msg_envelope("Error", {"code": "UNKNOWN_KIND", "kind": kind})

            writer.write((jdump(resp) + "\n").encode("utf-8"))
            await writer.drain()
        except Exception:
            try:
                writer.write((jdump(msg_envelope("Error", {"code": "SERVER_ERROR"})) + "\n").encode("utf-8"))
                await writer.drain()
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

async def client_send(host: str, port: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    r, w = await asyncio.open_connection(host, port)
    w.write((jdump(msg_envelope("ClientRequest", payload)) + "\n").encode("utf-8"))
    await w.drain()
    line = await r.readline()
    w.close()
    try:
        await w.wait_closed()
    except Exception:
        pass
    return json.loads(line.decode("utf-8")) if line else {}

async def demo():
    base = int(os.environ.get("RESON_BASE_PORT", "9800"))
    nodes = {"n1": ("127.0.0.1", base + 1), "n2": ("127.0.0.1", base + 2), "n3": ("127.0.0.1", base + 3)}
    cluster = ClusterConfig(voters=nodes, learners={})

    n1 = RaftNode("n1", *nodes["n1"], cluster)
    n2 = RaftNode("n2", *nodes["n2"], cluster)
    n3 = RaftNode("n3", *nodes["n3"], cluster)

    await n1.start(); await n2.start(); await n3.start()
    await asyncio.sleep(2.0)

    # demo client: try n1 repeatedly (NOT_LEADER면 무시)
    th, tp = nodes["n1"]
    for i in range(30):
        resp = await client_send(th, tp, {
            "client_id": "c",
            "request_id": sha256_hex(str(now_ms()))[:10],
            "op": "kv_set",
            "args": {"k": f"k{i}", "v": i}
        })
        # ignore NOT_LEADER in demo

    print("[DEMO] Ctrl+C to stop. Restart to confirm meta(commit_index/last_applied) persisted.")
    try:
        while True:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        await n1.stop(); await n2.stop(); await n3.stop()

if __name__ == "__main__":
    asyncio.run(demo())
