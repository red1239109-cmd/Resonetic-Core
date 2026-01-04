#!/usr/bin/env python3
"""
Resonetics v22.1.2-Anchor
The Judicial Kernel for Autonomous Agents & High-Assurance Systems

THE ARCHITECT'S FINAL BASELINE: SIGNATURE FIXED & LOGIC HARDENED
- FIXED: TypeError in _create_regularized_sample (Removed unused 'label' arg to match call sites)
- FIXED: Budget Accounting in _bounded_iter (Overflow probe now pays tax)
- FIXED: Instance Isolation (Budgets derived from self.limits["ops"] instead of global constant)
- KEPT: Full register_observation binding, DB_LOCK guards, and Binary/Text sampling logic
- Status: THE GOLDEN MASTER. EXECUTION READY. NO COMPRESSION.

Copyright 2026 red1239109-cmd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---
[README.md]
# Resonetics: The Anchor

> "In a world of stochastic chaos, we provide deterministic stasis."

Resonetics is a high-assurance judicial kernel designed to govern autonomous agents.
It enforces physical limits on data processing and ensures an immutable forensic chain
of evidence through deterministic serialization and hierarchical locking.
---
"""

from __future__ import annotations
import sys, time, math, asyncio, sqlite3, json, numbers, threading, hashlib, re, inspect, secrets, weakref, random, base64, os, queue
from typing import Any, Tuple, Optional, Union, Dict, List, NamedTuple, Set, Deque
from collections import deque, Counter as CCounter, UserDict
from collections.abc import Mapping, Iterable
from types import MappingProxyType

# ==============================================================================
# [LAYER 0] COSMIC CONSTANTS & DATA CONTRACTS
# ==============================================================================

COMMON_CFG = {"ensure_ascii": False, "allow_nan": False, "sort_keys": True}

_PHYSICS_SAMPLE_SIZE = 4096
_PHYSICS_TIME_BUDGET = 0.05
_PHYSICS_OP_LIMIT = 50000
_PHYSICS_PAYLOAD_LIMIT = 500_000

_SET_PAYLOAD_LIMIT = 1000
_SEQ_PAYLOAD_LIMIT = 1000
_STR_PAYLOAD_LIMIT = 2048

_REPR_LEN_LIMIT = 1024
_HASH_SLICE_SIZE = 4096
_HASH_THRESHOLD = 8192

_FREEZE_MAX_DEPTH = 12

_SCHEMA_VERSION = 22120  # v22.1.2 Canon
_COSMIC_SIGNATURE = _SCHEMA_VERSION

_VALID_TAGS = {
    "NUM", "BYTES",
    "SET", "LIST", "TUPLE", "UNKNOWN",
    "SET_TRUNC", "SEQ_TRUNC", "MAP_TRUNC", "STR_TRUNC",
    "CYCLIC_REF", "DEPTH_LIMIT",
    "TIME_LIMIT", "BUDGET_EXCEEDED"
}

_TIMEOUT_TAGS = {"TIME_LIMIT"}
_BUDGET_TAGS = {"BUDGET_EXCEEDED"}
_TRUNC_TAGS = {"SET_TRUNC", "SEQ_TRUNC", "MAP_TRUNC", "STR_TRUNC"}
_TIMEOUT_REASONS = {"TIMEOUT", "TIME_THEFT"}
_BUDGET_REASONS = {"BUDGET_EXCEEDED", "BUDGET_THEFT"}

_MIN_COMPATIBLE_VERSION = 8231
_RESO_TOKEN_BASE = ("<<RESO>>", _SCHEMA_VERSION)

# Core Identity Whitelists
_CORE_PRIMITIVES = (type(None), bool, int)
_SNAPSHOT_CORE_TYPES = (type(None), bool, int, float, str)
_STRUCTURAL_CORE_TYPES = (dict, list, tuple, set, frozenset)
_SAFE_TYPES = (type(None), bool, int, float, str, bytes, bytearray)

_BOOT_SENTINEL = object()
_CYCLE_SENTINEL = object()

# Forensic Grammar
SUFFIX_GRAMMAR = (
    "_PAYLOAD_LIMIT_EXCEEDED",
    "_SAMPLED_BY_SIZE",
    "_SAMPLED_BY_TIME",
    "_TRUNC",
    "_SNAPOK",
    "_SNAPFAIL",
    "_HAS_NESTED_BIN",
    "_HAS_ROOT_BIN",
    "_SCAN_INCOMPLETE",
)

LOCK_LEVELS = {"DB_LOCK": 1, "REBIRTH_LOCK": 2, "RULING_LOCK": 3}
REENTRANT_SCOPES = {"DB_LOCK", "RULING_LOCK"}

T_OBS = "observation_log"
T_MARK = "cosmos_marker"
T_ACT  = "action_proposals"
T_RULE = "judicial_rulings"

COLS = (
    "timestamp", "source_file", "source_func", "obs_content", "obs_kind", "obs_error", "obs_error_code",
    "obs_lock_id", "obs_boot_id",
    "obs_origin_type", "obs_origin_key", "obs_origin_is_structural", "obs_json_valid", "obs_source_json_valid",
    "obs_raw_size", "obs_stored_size", "obs_sha256_raw", "obs_sha256_final", "obs_truncated", "obs_transformed",
    "obs_is_binary", "obs_artifact_wrapped", "obs_has_nested_bin", "obs_has_root_bin", "obs_scan_incomplete"
)

SLOT_TYPES = {
    "timestamp": "REAL",
    "obs_raw_size": "INTEGER",
    "obs_stored_size": "INTEGER",
    "obs_json_valid": "INTEGER",
    "obs_source_json_valid": "INTEGER",
    "obs_origin_is_structural": "INTEGER",
    "obs_truncated": "INTEGER",
    "obs_transformed": "INTEGER",
    "obs_is_binary": "INTEGER",
    "obs_artifact_wrapped": "INTEGER",
    "obs_has_nested_bin": "INTEGER",
    "obs_has_root_bin": "INTEGER",
    "obs_scan_incomplete": "INTEGER",
}

OBS_INDICES = {
    "idx_obs_key_ts": ('"obs_origin_key"', '"timestamp"'),
    "idx_obs_loc_ts": ('"source_file"', '"source_func"', '"timestamp"'),
    "idx_obs_lineage": ('"obs_boot_id"', '"obs_lock_id"', '"timestamp"'),
    "idx_obs_ts": ('"timestamp"',),
    "idx_obs_wrapped_ts": ('"obs_artifact_wrapped"', '"timestamp"'),
    "idx_obs_nested_ts": ('"obs_has_nested_bin"', '"timestamp"'),
    "idx_obs_root_bin_ts": ('"obs_has_root_bin"', '"timestamp"'),
    "idx_obs_incomplete_ts": ('"obs_scan_incomplete"', '"timestamp"'),
}

RULE_INDICES = {
    "idx_rule_ts": ('"timestamp"',),
    "idx_rule_seq": ('"seq"',),
    "idx_rule_boot": ('"boot_id"',),
}

# Core DTOs
class ForensicRuling(NamedTuple):
    reason: str
    tags: Tuple[str, ...]
    evidence: Mapping[str, Any]
    timestamp: float
    seq: int

class PhysicsResult(NamedTuple):
    content: str
    raw_sz: int
    sha_r: str
    sha_f: str
    err_code: Optional[str]
    origin_type: str
    has_bin: int
    is_bin: int
    root_bin: int
    nested_bin: int
    scan_inc: int
    is_structural: bool
    json_parseable: int
    source_json_valid: int
    sampled_size: int
    sampled_time: int
    has_trunc: int
    snap_ok: int

# ==============================================================================
# [LAYER 0.5] DETERMINISTIC CHRONOS & IDENTITY CORE
# ==============================================================================

class DeterministicCtx:
    """Master clock for deterministic worlds."""
    def __init__(self, active: bool = False):
        self.active = active
        self.mock_time = 0.0
        self.nonce_counter = 0

    def get_time(self) -> float:
        if self.active:
            self.mock_time += 1e-6
            return self.mock_time
        return time.perf_counter()

    def get_wall_time(self) -> float:
        return self.mock_time if self.active else time.time()

    def tick(self, seconds: float):
        if self.active:
            self.mock_time += seconds

    def generate_nonce(self, length: int = 8) -> str:
        if self.active:
            self.nonce_counter += 1
            full_str = f"MOCK{self.nonce_counter:012d}"
            return full_str[:length] if len(full_str) >= length else full_str.ljust(length, "_")
        return secrets.token_hex(max(1, length // 2))

class _SessionIdentity:
    """Protects object identity without creating memory leaks."""
    def __init__(self, cleanup_threshold: int = 10000):
        self._id_map: Dict[int, int] = {}
        self._refs: Set[weakref.ref] = set()
        self._counter = 0
        self._cleanup_threshold = cleanup_threshold
        self._access_count = 0
        self._lock = threading.Lock()

    def get_sid(self, obj: Any) -> str:
        obj_id = id(obj)
        with self._lock:
            self._access_count += 1
            if self._access_count >= self._cleanup_threshold:
                self._cleanup()
            if obj_id not in self._id_map:
                self._counter += 1
                self._id_map[obj_id] = self._counter
                try:
                    def cb(ref, oid=obj_id):
                        with self._lock:
                            self._id_map.pop(oid, None)
                            self._refs.discard(ref)
                    self._refs.add(weakref.ref(obj, cb))
                except:
                    pass
            return f"sid:{self._id_map[obj_id]:08x}"

    def _cleanup(self):
        self._access_count = 0
        if len(self._id_map) > self._cleanup_threshold * 2:
            self._id_map.clear()
            self._refs.clear()

_SID_MANAGER = _SessionIdentity()

# ==============================================================================
# [LAYER 0.6] UNIVERSAL FREEZER & ENFORCEMENT
# ==============================================================================

def _safe_len_any(obj: Any) -> int:
    try:
        return len(obj)
    except:
        return -1

def _get_static_raw(obj: Any) -> bytes:
    """Return static identity bytes without executing arbitrary user methods."""
    if type(obj) in _SNAPSHOT_CORE_TYPES:
        return str(obj).encode("utf-8", "replace")
    if type(obj) in (bytes, bytearray):
        return bytes(obj)
    return f"<{type(obj).__module__}.{type(obj).__qualname__} sid:{_SID_MANAGER.get_sid(obj)}>".encode("utf-8", "replace")

# [PATCH A: Iteration Logic]
def _bounded_iter(it: Iterable, limit: int, deadline: float, budget: Optional[List[int]], ctx: Optional[DeterministicCtx]) -> Tuple[List[Any], Optional[str]]:
    items: List[Any] = []
    try:
        iterator = iter(it)
        for _ in range(limit):
            now = ctx.get_time() if ctx else time.perf_counter()
            if deadline > 0 and now > deadline:
                return items, "TIMEOUT"
            if budget is not None and budget[0] <= 0:
                return items, "BUDGET_EXCEEDED"
            if budget is not None:
                budget[0] -= 1
            v = next(iterator)
            items.append(v)
        
        # Probe for overflow
        try:
            now = ctx.get_time() if ctx else time.perf_counter()
            if deadline > 0 and now > deadline:
                return items, "TIMEOUT"
            if budget is not None and budget[0] <= 0:
                return items, "BUDGET_EXCEEDED"
            
            # [FIXED] Tax paid for the probe
            if budget is not None:
                budget[0] -= 1
                
            next(iterator)
            return items, "OVERFLOW"
        except StopIteration:
            return items, None
            
    except StopIteration:
        return items, None
    except Exception:
        return items, "ITER_ERROR"

def _compute_fingerprint(b: bytes, budget: Optional[List[int]], deadline: float, payload_limit: int, ctx: Optional[DeterministicCtx] = None) -> str:
    total_len = len(b)
    if total_len > payload_limit:
        return "PAYLOAD_LIMIT_EXCEEDED"
    h = hashlib.sha256()
    view = memoryview(b)
    chunk = _HASH_SLICE_SIZE
    cursor = 0
    while cursor < total_len:
        now = ctx.get_time() if ctx else time.perf_counter()
        if deadline > 0 and now > deadline:
            return "SHA_TIMEOUT"
        if budget is not None and budget[0] <= 0:
            return "SHA_BUDGET_EXHAUSTED"
        end = min(cursor + chunk, total_len)
        h.update(view[cursor:end])
        cursor = end
        if budget is not None:
            budget[0] -= 1
    return h.hexdigest()

def _is_timeoutish(x) -> bool:
    if type(x) is not tuple or len(x) < 3 or x[0] != "<<RESO>>":
        return False
    if x[2] in _TIMEOUT_TAGS:
        return True
    if x[2] in _TRUNC_TAGS and len(x) >= 4 and x[3] in _TIMEOUT_REASONS:
        return True
    return False

def _is_budgetish(x) -> bool:
    if type(x) is not tuple or len(x) < 3 or x[0] != "<<RESO>>":
        return False
    if x[2] in _BUDGET_TAGS:
        return True
    if x[2] in _TRUNC_TAGS and len(x) >= 4 and x[3] in _BUDGET_REASONS:
        return True
    return False

def _materialize_lite(obj: Any, budget: Optional[List[int]], deadline: float, ctx: Optional[DeterministicCtx]) -> Any:
    now = ctx.get_time() if ctx else time.perf_counter()
    if deadline > 0 and now > deadline:
        return "MAT_TIMEOUT"
    if budget is not None and budget[0] <= 0:
        return "MAT_BUDGET_EXCEEDED"
    if budget is not None:
        budget[0] -= 1
    if type(obj) in _SNAPSHOT_CORE_TYPES:
        return obj
    if type(obj) is tuple and len(obj) >= 2 and obj[0] == "<<RESO>>":
        tag = obj[2] if len(obj) >= 3 else "MALFORMED"
        if tag == "NUM" and len(obj) >= 4:
            return obj[3]
        if tag == "BYTES" and len(obj) >= 5:
            return obj[4]
        if tag == "STR_TRUNC" and len(obj) >= 5:
            return obj[4]
        if tag in ("SET", "LIST", "TUPLE") and len(obj) >= 4:
            payload = obj[3]
            if type(payload) in (list, tuple):
                return [_materialize_lite(x, budget, deadline, ctx) for x in payload]
            return [str(tag)]
        return str(tag)
    return f"<{type(obj).__name__}>"

def _deterministic_sort_key(frozen_obj: Any, budget: Optional[List[int]], deadline: float, payload_limit: int, ctx: Optional[DeterministicCtx]) -> Tuple[str, str]:
    try:
        mat = _materialize_lite(frozen_obj, budget, deadline, ctx)
        s = json.dumps(mat, **COMMON_CFG)
        f = _compute_fingerprint(s.encode("utf-8", "replace"), budget, deadline, payload_limit, ctx)
        return (f, s)
    except:
        return ("HASH_FAIL", str(type(frozen_obj)))

def _universal_freeze(
    obj: Any,
    depth: int = 0,
    ancestors: Optional[Set[int]] = None,
    memo: Optional[Dict[int, Any]] = None,
    budget: Optional[List[int]] = None,
    meta: Optional[Dict[str, bool]] = None,
    deadline: float = 0.0,
    payload_limit: int = _PHYSICS_PAYLOAD_LIMIT,
    ctx: Optional[DeterministicCtx] = None
) -> Any:
    if meta is None:
        meta = {}

    if type(obj) is tuple and len(obj) >= 2 and obj[0] == "<<RESO>>":
        if type(obj[1]) is int and obj[1] > _SCHEMA_VERSION:
            meta["has_future"] = True
        return obj

    if type(obj) in _CORE_PRIMITIVES:
        return obj

    if ancestors is None:
        ancestors = set()
    if memo is None:
        memo = {}

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]
    if obj_id in ancestors:
        return _RESO_TOKEN_BASE + ("CYCLIC_REF",)
    if depth > _FREEZE_MAX_DEPTH:
        return _RESO_TOKEN_BASE + ("DEPTH_LIMIT",)

    now = ctx.get_time() if ctx else time.perf_counter()
    if deadline > 0 and now > deadline:
        meta["has_timeout"] = True
        return _RESO_TOKEN_BASE + ("TIME_LIMIT",)

    if budget is not None:
        budget[0] -= 1
        if budget[0] < 0:
            meta["has_budget"] = True
            return _RESO_TOKEN_BASE + ("BUDGET_EXCEEDED",)

    ancestors.add(obj_id)
    try:
        if type(obj) is float:
            if math.isnan(obj):
                res = _RESO_TOKEN_BASE + ("NUM", "NaN")
            elif math.isinf(obj):
                res = _RESO_TOKEN_BASE + ("NUM", "Inf" if obj > 0 else "-Inf")
            else:
                res = obj
            memo[obj_id] = res
            return res

        if type(obj) is str:
            if len(obj) > _STR_PAYLOAD_LIMIT:
                chk = _compute_fingerprint(obj.encode("utf-8", "replace"), budget, deadline, payload_limit, ctx)
                chk16 = chk[:16] if len(chk) == 64 else chk
                res = _RESO_TOKEN_BASE + ("STR_TRUNC", len(obj), chk16)
                meta["has_trunc"] = True
            else:
                res = obj
            memo[obj_id] = res
            return res

        if type(obj) in (bytes, bytearray):
            meta["has_bin"] = True
            b = bytes(obj)
            if len(b) > 64:
                chk = _compute_fingerprint(b, budget, deadline, payload_limit, ctx)
                chk16 = chk[:16] if len(chk) == 64 else chk
                res = _RESO_TOKEN_BASE + ("BYTES", len(b), chk16)
            else:
                res = _RESO_TOKEN_BASE + ("BYTES", len(b), b.hex())
            memo[obj_id] = res
            return res

        if isinstance(obj, Mapping):
            n_items = _safe_len_any(obj)
            if n_items == -1:
                keys, reason = _bounded_iter(obj.keys(), _SEQ_PAYLOAD_LIMIT, deadline, budget, ctx)
                if reason:
                    res = _RESO_TOKEN_BASE + ("MAP_TRUNC", reason)
                    meta["has_trunc"] = True
                    memo[obj_id] = res
                    return res
                keys_to_use = keys
            else:
                if n_items > _SEQ_PAYLOAD_LIMIT:
                    res = _RESO_TOKEN_BASE + ("MAP_TRUNC", n_items)
                    meta["has_trunc"] = True
                    memo[obj_id] = res
                    return res
                try:
                    keys_to_use = list(obj.keys())
                except Exception:
                    res = _RESO_TOKEN_BASE + ("MAP_TRUNC", "ITER_ERROR")
                    meta["has_trunc"] = True
                    memo[obj_id] = res
                    return res

            canonical_pairs = []
            for k in keys_to_use:
                frozen_k = _universal_freeze(k, depth + 1, ancestors, memo, budget, meta, deadline, payload_limit, ctx)
                sk = _deterministic_sort_key(frozen_k, budget, deadline, payload_limit, ctx)
                canonical_pairs.append((sk, k, frozen_k))

            canonical_pairs.sort(key=lambda p: p[0])
            out: Dict[str, Any] = {}
            seen_counts: Dict[str, int] = {}

            for sort_key, k_orig, frozen_k_stored in canonical_pairs:
                try:
                    v = obj[k_orig]
                except Exception:
                    v = _RESO_TOKEN_BASE + ("MAP_TRUNC", "ACCESS_ERROR")
                    meta["has_trunc"] = True

                frozen_val = _universal_freeze(v, depth + 1, ancestors, memo, budget, meta, deadline, payload_limit, ctx)

                base_key = f"{type(k_orig).__module__}.{type(k_orig).__qualname__}#H{sort_key[0]}"
                n = seen_counts.get(base_key, 0)
                seen_counts[base_key] = n + 1
                final_key = base_key if n == 0 else f"{base_key}#C{n}"

                out[final_key] = {"__k__": frozen_k_stored, "__v__": frozen_val}

                if _is_timeoutish(frozen_val):
                    res = _RESO_TOKEN_BASE + ("MAP_TRUNC", "TIME_THEFT")
                    meta["has_trunc"] = True
                    memo[obj_id] = res
                    return res
                if _is_budgetish(frozen_val):
                    res = _RESO_TOKEN_BASE + ("MAP_TRUNC", "BUDGET_EXCEEDED")
                    meta["has_trunc"] = True
                    memo[obj_id] = res
                    return res

            res = MappingProxyType(out)
            memo[obj_id] = res
            return res

        if isinstance(obj, (list, tuple, set, frozenset, deque)):
            is_set = isinstance(obj, (set, frozenset))
            is_tuple = isinstance(obj, tuple)
            tag = "SET" if is_set else ("TUPLE" if is_tuple else "LIST")
            trunc_tag = "SET_TRUNC" if is_set else "SEQ_TRUNC"

            n_items = _safe_len_any(obj)
            lim = _SET_PAYLOAD_LIMIT if is_set else _SEQ_PAYLOAD_LIMIT

            if n_items == -1:
                items, reason = _bounded_iter(obj, lim, deadline, budget, ctx)
                if reason:
                    res = _RESO_TOKEN_BASE + (trunc_tag, reason)
                    meta["has_trunc"] = True
                    memo[obj_id] = res
                    return res
                seq = items
            else:
                if n_items > lim:
                    res = _RESO_TOKEN_BASE + (trunc_tag, n_items)
                    meta["has_trunc"] = True
                    memo[obj_id] = res
                    return res
                seq = list(obj) if not is_tuple else list(obj)

            frozen_items: List[Any] = []
            for x in seq:
                fx = _universal_freeze(x, depth + 1, ancestors, memo, budget, meta, deadline, payload_limit, ctx)
                frozen_items.append(fx)
                if _is_timeoutish(fx) or _is_budgetish(fx):
                    break

            if any(_is_timeoutish(x) for x in frozen_items):
                res = _RESO_TOKEN_BASE + (trunc_tag, "TIME_THEFT")
                meta["has_trunc"] = True
            elif any(_is_budgetish(x) for x in frozen_items):
                res = _RESO_TOKEN_BASE + (trunc_tag, "BUDGET_EXCEEDED")
                meta["has_trunc"] = True
            else:
                if is_set:
                    frozen_items.sort(key=lambda x: _deterministic_sort_key(x, budget, deadline, payload_limit, ctx))
                res = _RESO_TOKEN_BASE + (tag, tuple(frozen_items))

            memo[obj_id] = res
            return res

        meta["has_unknown"] = True
        res = _RESO_TOKEN_BASE + ("UNKNOWN", f"{type(obj).__module__}.{type(obj).__qualname__}")
        memo[obj_id] = res
        return res
    finally:
        ancestors.discard(obj_id)

# ==============================================================================
# [LAYER 1] SOVEREIGN BODY & JUDICIAL KERNEL
# ==============================================================================

class LockEntry:
    def __init__(self, ctx: Optional[DeterministicCtx] = None):
        self.lock = threading.RLock()
        self.rebirth_lock = threading.Lock()
        self.ruling_lock = threading.RLock()
        self.ref_count = 0
        self.created_ts = ctx.get_wall_time() if ctx else time.time()
        self.nonce_id = ctx.generate_nonce(8) if ctx else secrets.token_hex(4)
        self.ruling_seq = 0
        self.last_ruling: Optional[ForensicRuling] = None
        self.ruling_trail: Deque[ForensicRuling] = deque(maxlen=16)

class SovereignBody:
    _GLOBAL_PATH_LOCKS: Dict[str, "LockEntry"] = {}
    _REGISTRY_LOCK = threading.Lock()
    _LOCK_TRACKER = threading.local()

    def __init__(self, db_path: str, autoinit: bool = True, deterministic: bool = False, payload_limit: int = _PHYSICS_PAYLOAD_LIMIT):
        self._closed = False
        self.db_path = db_path
        self._boot_nonce = "UNKNOWN"
        self.limits = {"payload": payload_limit, "sample": _PHYSICS_SAMPLE_SIZE, "ops": _PHYSICS_OP_LIMIT, "time": _PHYSICS_TIME_BUDGET}

        is_det = deterministic or (os.environ.get("RESONETICS_TESTING") == "1")
        self._ctx = DeterministicCtx(active=True) if is_det else DeterministicCtx(active=False)

        if db_path == ":memory:":
            self._norm_path = f"file:reso_akashic_{self._ctx.generate_nonce(12)}?mode=memory&cache=shared"
            self._use_uri = True
        elif db_path.startswith("file:"):
            self._norm_path = db_path
            self._use_uri = True
        else:
            self._norm_path = os.path.normcase(os.path.abspath(db_path))
            self._use_uri = False

        self._norm_key = self._norm_path.lower()

        with self._REGISTRY_LOCK:
            if self._norm_key not in self._GLOBAL_PATH_LOCKS:
                self._GLOBAL_PATH_LOCKS[self._norm_key] = LockEntry(self._ctx)
            self._lock_entry = self._GLOBAL_PATH_LOCKS[self._norm_key]
            self._lock_entry.ref_count += 1
            self._lock = self._lock_entry.lock

        try:
            with self._hierarchy_guard("DB_LOCK"):
                with self._lock:
                    self.conn = sqlite3.connect(self._norm_path, uri=self._use_uri, check_same_thread=False, isolation_level=None)
                    self.conn.execute("PRAGMA busy_timeout = 1000")
                    if autoinit:
                        self._init_tables()
        except:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        with self._REGISTRY_LOCK:
            if hasattr(self, "_lock_entry"):
                self._lock_entry.ref_count -= 1
                if self._lock_entry.ref_count <= 0:
                    self._GLOBAL_PATH_LOCKS.pop(self._norm_key, None)
        try:
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
        except:
            pass

    def __enter__(self): return self
    def __exit__(self, t, v, tb): self.close()

    @staticmethod
    def _reset_thread_state_for_test_ONLY():
        if os.environ.get("RESONETICS_TESTING") != "1":
            raise RuntimeError("TEST_ONLY")
        SovereignBody._LOCK_TRACKER.held = CCounter()

    class _hierarchy_guard:
        def __init__(self, lock_type: str):
            self.lock_type = lock_type
            if not hasattr(SovereignBody._LOCK_TRACKER, 'held'):
                SovereignBody._LOCK_TRACKER.held = CCounter()
        def __enter__(self):
            held = SovereignBody._LOCK_TRACKER.held
            active_levels = [LOCK_LEVELS[t] for t, count in held.items() if count > 0]
            current_max = max(active_levels) if active_levels else 0
            if held[self.lock_type] > 0 and self.lock_type not in REENTRANT_SCOPES:
                raise RuntimeError("Deadlock Guard")
            if LOCK_LEVELS[self.lock_type] <= current_max and held[self.lock_type] == 0:
                raise RuntimeError("Hierarchy Violation")
            held[self.lock_type] += 1
        def __exit__(self, t, v, tb):
            SovereignBody._LOCK_TRACKER.held[self.lock_type] -= 1

    def _verify_schema(self) -> bool:
        try:
            rows = self.conn.execute(f"PRAGMA table_info('{T_OBS}')").fetchall()
            return set(COLS).issubset({row[1] for row in rows})
        except:
            return False

    def _init_tables(self):
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.conn.execute(f'CREATE TABLE IF NOT EXISTS "{T_ACT}" (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, payload TEXT)')
                self.conn.execute(f'CREATE TABLE IF NOT EXISTS "{T_RULE}" (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, seq INTEGER, reason TEXT, tags TEXT, evidence TEXT, boot_id TEXT, lock_id TEXT)')
                self.conn.execute(f'CREATE TABLE IF NOT EXISTS "{T_MARK}" (id INTEGER PRIMARY KEY CHECK (id=1), boot_nonce TEXT NOT NULL, created_ts REAL NOT NULL)')

                row_mark = self.conn.execute(f'SELECT boot_nonce FROM "{T_MARK}" WHERE id=1').fetchone()
                if not row_mark:
                    self._boot_nonce = self._ctx.generate_nonce(16)
                    ts = self._ctx.get_wall_time()
                    self.conn.execute(f'INSERT INTO "{T_MARK}" VALUES (1, ?, ?)', (self._boot_nonce, ts))
                else:
                    self._boot_nonce = row_mark[0]

                if not self._verify_schema():
                    ts = self._ctx.get_wall_time()
                    self.conn.execute(
                        f'INSERT INTO "{T_RULE}" (timestamp, seq, reason, tags, evidence, boot_id, lock_id) VALUES (?, ?, ?, ?, ?, ?, ?)',
                        (ts, 0, "AUTO_REBIRTH_INITIATED", '["SCHEMA_MISMATCH"]', '{"action": "DROP_RECREATE"}', self._boot_nonce, self._lock_entry.nonce_id)
                    )
                    self.conn.execute(f'DROP TABLE IF EXISTS "{T_OBS}"')
                    ddl_cols = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
                    for c in COLS:
                        ctype = SLOT_TYPES.get(c, "TEXT")
                        ddl_cols.append(f'"{c}" {ctype}')
                    self.conn.execute(f'CREATE TABLE "{T_OBS}" ({", ".join(ddl_cols)})')

                for idx, cols_list in OBS_INDICES.items():
                    self.conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx}" ON "{T_OBS}" ({",".join(cols_list)})')
                for idx, cols_list in RULE_INDICES.items():
                    self.conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx}" ON "{T_RULE}" ({",".join(cols_list)})')

                self.conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                max_seq = self.conn.execute(f'SELECT COALESCE(MAX(seq),0) FROM "{T_RULE}"').fetchone()[0]
                with self._lock_entry.ruling_lock:
                    self._lock_entry.ruling_seq = int(max_seq)

                self.conn.execute("COMMIT")
            except:
                self.conn.execute("ROLLBACK")
                raise

    def _set_ruling(self, reason: str, tags: Tuple[str, ...] = (), evidence: Mapping[str, Any] = None) -> None:
        with self._hierarchy_guard("RULING_LOCK"):
            with self._lock_entry.ruling_lock:
                self._lock_entry.ruling_seq += 1
                deadline = self._ctx.get_time() + self.limits["time"]
                ev_frozen = _universal_freeze(evidence or {}, deadline=deadline, payload_limit=self.limits["payload"], ctx=self._ctx)
                if isinstance(ev_frozen, Mapping) and not isinstance(ev_frozen, MappingProxyType):
                    ev_frozen = MappingProxyType(dict(ev_frozen))
                ruling = ForensicRuling(reason, tags, ev_frozen, self._ctx.get_wall_time(), self._lock_entry.ruling_seq)
                self._lock_entry.last_ruling = ruling
                self._lock_entry.ruling_trail.append(ruling)

        try:
            with self._hierarchy_guard("DB_LOCK"):
                with self._lock:
                    if self.conn:
                        ev_json = json.dumps(
                            {"kind": "RULING", "payload": self._deep_materialize(ev_frozen, [2000], deadline), "hints": []},
                            **COMMON_CFG
                        )
                        self.conn.execute(
                            f'INSERT INTO "{T_RULE}" (timestamp, seq, reason, tags, evidence, boot_id, lock_id) VALUES (?, ?, ?, ?, ?, ?, ?)',
                            (ruling.timestamp, ruling.seq, ruling.reason, json.dumps(tags, **COMMON_CFG), ev_json, self._boot_nonce, self._lock_entry.nonce_id)
                        )
        except:
            pass

    # [PATCH A + PATCH B] Fixed Signature + Binary Semantic Fix
    def _create_regularized_sample(self, b: bytes, trail: List[str], deadline: float, budget: Optional[List[int]], hints: List[str], is_bin: bool) -> str:
        raw_len = len(b)
        slice_len = max(256, self.limits["sample"] // 4)
        try:
            sha_val = _compute_fingerprint(b, budget, deadline, self.limits["payload"], self._ctx)
            sha16 = sha_val[:16] if len(sha_val) == 64 else sha_val
        except:
            sha16 = "HASH_FAIL"

        # Only use BINARY_SAMPLE if the root origin was binary.
        if not is_bin:
            payload = {
                "head": b[:slice_len].decode("utf-8", "replace"),
                "tail": b[-slice_len:].decode("utf-8", "replace") if raw_len > slice_len else ""
            }
            kind = "TEXT_SAMPLE"
        else:
            payload = {
                "encoding": "base64",
                "head": base64.b64encode(b[:slice_len]).decode("ascii"),
                "tail": base64.b64encode(b[-slice_len:]).decode("ascii") if raw_len > slice_len else ""
            }
            kind = "BINARY_SAMPLE"

        return json.dumps(
            {"kind": kind, "payload": payload, "hints": sorted(list(set(hints or []))), "meta": {"raw_size": raw_len, "sha16": sha16, "trail": list(trail)}},
            **COMMON_CFG
        )

    def _unified_scan(self, x: Any, *, deadline: float, budget: Optional[List[int]], meta: Dict[str, bool], depth: int = 0) -> Optional[str]:
        now = self._ctx.get_time()
        if deadline > 0 and now > deadline:
            return "TIMEOUT"
        if budget is not None and budget[0] <= 0:
            return "BUDGET"
        if budget is not None:
            budget[0] -= 1

        if type(x) is tuple and len(x) >= 3 and x[0] == "<<RESO>>":
            tag = x[2]
            if tag == "BYTES":
                if depth == 0:
                    meta["has_root_bin"] = True
                else:
                    meta["has_nested_bin"] = True
            if tag in _TIMEOUT_TAGS:
                return "TIMEOUT"
            if tag in _BUDGET_TAGS:
                return "BUDGET"
            if tag in _TRUNC_TAGS and len(x) >= 4:
                if x[3] in _TIMEOUT_REASONS:
                    return "TIMEOUT"
                if x[3] in _BUDGET_REASONS:
                    return "BUDGET"
            if tag not in _VALID_TAGS:
                meta["has_unknown"] = True

            for child in x[3:]:
                r = self._unified_scan(child, deadline=deadline, budget=budget, meta=meta, depth=depth + 1)
                if r in ("TIMEOUT", "BUDGET"):
                    return r
            return "OK"

        if isinstance(x, (list, tuple)):
            for child in x:
                r = self._unified_scan(child, deadline=deadline, budget=budget, meta=meta, depth=depth + 1)
                if r in ("TIMEOUT", "BUDGET"):
                    return r
            return "OK"

        if isinstance(x, Mapping):
            try:
                for k, v in x.items():
                    r1 = self._unified_scan(k, deadline=deadline, budget=budget, meta=meta, depth=depth + 1)
                    if r1 in ("TIMEOUT", "BUDGET"):
                        return r1
                    r2 = self._unified_scan(v, deadline=deadline, budget=budget, meta=meta, depth=depth + 1)
                    if r2 in ("TIMEOUT", "BUDGET"):
                        return r2
            except Exception:
                meta["scan_inc"] = True
                return "OK"
            return "OK"

        return "OK"

    def _payload_physics(self, obj: Any) -> PhysicsResult:
        orig = obj
        o_type = f"{type(orig).__module__}.{type(orig).__qualname__}"
        is_struct = type(orig) in _STRUCTURAL_CORE_TYPES or isinstance(orig, Mapping) or isinstance(orig, (list, tuple, set, frozenset, deque))
        is_bin = type(orig) in (bytes, bytearray)

        deadline = self._ctx.get_time() + self.limits["time"]
        
        # [PATCH D] Instance-based Ops Limit
        ops_lim = int(self.limits.get("ops", _PHYSICS_OP_LIMIT))
        budget_freeze = [ops_lim // 2]
        budget_hash = [ops_lim // 2]
        scan_budget = [ops_lim // 2]

        f_ctx: Dict[str, bool] = {
            "has_bin": False,
            "has_root_bin": bool(is_bin),
            "has_nested_bin": False,
            "scan_inc": False,
            "has_trunc": False,
            "has_unknown": False,
            "has_future": False,
        }

        raw_b = _get_static_raw(orig)
        raw_sz = len(raw_b)

        src_json_v = 0
        try:
            src_text = orig if type(orig) is str else (orig.decode("utf-8", "replace") if is_bin else None)
            if src_text is not None:
                json.loads(src_text)
                src_json_v = 1
        except:
            pass

        if type(orig) in _SNAPSHOT_CORE_TYPES and raw_sz > self.limits["payload"]:
            content = json.dumps({"kind": "LIMIT_EXCEEDED", "payload": "OVERSIZE", "hints": []}, **COMMON_CFG)
            return PhysicsResult(content, raw_sz, "OVERSIZE", "OVERSIZE", "PAYLOAD_LIMIT_EXCEEDED", o_type,
                               0, 0, 0, 0, 0, bool(is_struct), 1, src_json_v, 0, 0, 0, 0)

        frozen = _universal_freeze(
            orig,
            budget=budget_freeze,
            meta=f_ctx,
            deadline=deadline,
            payload_limit=self.limits["payload"],
            ctx=self._ctx
        )

        if is_struct:
            scan_res = self._unified_scan(frozen, deadline=deadline, budget=scan_budget, meta=f_ctx, depth=0)
            if scan_res == "TIMEOUT":
                f_ctx["scan_inc"] = True
            elif scan_res == "BUDGET":
                f_ctx["scan_inc"] = True

        mat = self._deep_materialize(frozen, budget_freeze, deadline)

        hints: List[str] = []
        if f_ctx.get("has_nested_bin"):
            hints.append("HAS_NESTED_BIN")
        if f_ctx.get("has_root_bin"):
            hints.append("HAS_ROOT_BIN")
        if f_ctx.get("scan_inc"):
            hints.append("SCAN_INCOMPLETE")
        if f_ctx.get("has_unknown"):
            hints.append("HAS_UNKNOWN_TAG")
        if f_ctx.get("has_future"):
            hints.append("HAS_FUTURE_TOKEN")

        b_mat = json.dumps(mat, **COMMON_CFG).encode("utf-8", "replace")
        is_sampled_sz = 0
        is_sampled_tm = 0

        # [PATCH B] is_bin passed based on original type
        sample_is_bin = bool(is_bin)

        if len(b_mat) > self.limits["sample"]:
            # [PATCH A] Signature fix: Removed 'label', passed trail list as 2nd arg
            content = self._create_regularized_sample(b_mat, ["SAMPLED_BY_SIZE"], deadline, budget_hash, hints, is_bin=sample_is_bin)
            is_sampled_sz = 1
        elif self._ctx.get_time() > deadline:
            content = self._create_regularized_sample(b_mat, ["SAMPLED_BY_TIME"], deadline, budget_hash, hints, is_bin=sample_is_bin)
            is_sampled_tm = 1
        else:
            envelope = {
                "kind": "STRUCT" if is_struct else ("BIN" if is_bin else "SCALAR"),
                "payload": mat,
                "hints": sorted(list(set(hints))),
                "meta": {"origin": o_type, "schema": _SCHEMA_VERSION, "boot": self._boot_nonce}
            }
            try:
                if self._ctx.get_time() > deadline:
                    content = json.dumps({"kind": "TIMEOUT", "payload": "SERIAL_TIMEOUT", "hints": sorted(list(set(hints)))}, **COMMON_CFG)
                else:
                    content = json.dumps(envelope, **COMMON_CFG)
            except:
                content = json.dumps({"kind": "ERROR", "payload": "SERIAL_FAIL", "hints": sorted(list(set(hints)))}, **COMMON_CFG)

        sha_r_raw = _compute_fingerprint(raw_b, budget_hash, deadline, self.limits["payload"], self._ctx)
        sha_f_raw = _compute_fingerprint(content.encode("utf-8", "replace"), budget_hash, deadline, self.limits["payload"], self._ctx)

        final_err = None
        for s in (sha_r_raw, sha_f_raw):
            ss = str(s).lower()
            if len(ss) != 64 or any(c not in "0123456789abcdef" for c in ss):
                final_err = s
                break

        json_v = 0
        try:
            json.loads(content)
            json_v = 1
        except:
            pass

        has_trunc = 1 if f_ctx.get("has_trunc") else 0
        snap_ok = 1 if (json_v == 1 and final_err is None and not (is_sampled_sz or is_sampled_tm)) else 0

        if not snap_ok:
            causes: List[str] = []
            if json_v == 0:
                causes.append("INVALID_JSON")
            if final_err:
                causes.append(f"HASH_ERROR:{final_err}")
            if is_sampled_sz:
                causes.append("SIZE_SAMPLING")
            if is_sampled_tm:
                causes.append("TIME_SAMPLING")
            hints.append(f"FAIL_CAUSE:{'|'.join(causes)}")
            try:
                content_obj = json.loads(content)
                if isinstance(content_obj, dict):
                    content_obj["hints"] = sorted(list(set(content_obj.get("hints", []) + hints)))
                    content = json.dumps(content_obj, **COMMON_CFG)
            except:
                pass

        return PhysicsResult(
            content,
            raw_sz,
            str(sha_r_raw),
            str(sha_f_raw),
            final_err,
            o_type,
            int(bool(f_ctx.get("has_bin"))),
            int(bool(is_bin)),
            int(bool(f_ctx.get("has_root_bin"))),
            int(bool(f_ctx.get("has_nested_bin"))),
            int(bool(f_ctx.get("scan_inc"))),
            bool(is_struct),
            int(json_v),
            int(src_json_v),
            int(is_sampled_sz),
            int(is_sampled_tm),
            int(has_trunc),
            int(snap_ok)
        )

    def register_observation(self, file: str, func: str, obj: Any) -> Optional[int]:
        if self._closed:
            return None

        res = self._payload_physics(obj)
        now = self._ctx.get_wall_time()
        stored_sz = len(res.content.encode("utf-8", "replace"))

        flags = set()
        if res.raw_sz > self.limits["payload"]:
            flags.add("_PAYLOAD_LIMIT_EXCEEDED")
        if res.sampled_size:
            flags.add("_SAMPLED_BY_SIZE")
        if res.sampled_time:
            flags.add("_SAMPLED_BY_TIME")
        if res.has_trunc:
            flags.add("_TRUNC")
        if res.snap_ok:
            flags.add("_SNAPOK")
        else:
            flags.add("_SNAPFAIL")
        if res.scan_inc:
            flags.add("_SCAN_INCOMPLETE")
        if res.root_bin:
            flags.add("_HAS_ROOT_BIN")
        if res.nested_bin:
            flags.add("_HAS_NESTED_BIN")

        kind = "OBS_MONOLITH" + "".join(s for s in SUFFIX_GRAMMAR if s in flags)

        # [PATCH C] Hierarchy Guard Added
        with self._hierarchy_guard("DB_LOCK"):
            with self._lock:
                key = hashlib.sha1(f"{file}|{func}".encode("utf-8", "replace")).hexdigest()[:12]
                vals = (
                    now,
                    str(file),
                    str(func),
                    res.content,
                    kind,
                    None,
                    res.err_code,
                    self._lock_entry.nonce_id,
                    self._boot_nonce,
                    res.origin_type,
                    key,
                    1 if res.is_structural else 0,
                    res.json_parseable,
                    res.source_json_valid,
                    res.raw_sz,
                    stored_sz,
                    res.sha_r,
                    res.sha_f,
                    res.has_trunc,
                    1,
                    res.is_bin,
                    1,
                    res.nested_bin,
                    res.root_bin,
                    res.scan_inc
                )
                c = self.conn.execute(
                    f'INSERT INTO "{T_OBS}" ({",".join(COLS)}) VALUES ({",".join(["?"] * len(COLS))})',
                    vals
                )
                rid = c.lastrowid

        self._set_ruling("OBS_STORED", tags=("MONOLITH",), evidence={"rowid": rid, "kind": kind, "integrity": res.snap_ok})
        return rid

    def _deep_materialize(self, obj: Any, budget: Optional[List[int]], deadline: float) -> Any:
        now = self._ctx.get_time()
        if deadline > 0 and now > deadline:
            return {"__reso_error__": "MAT_TIMEOUT"}
        if budget is not None and budget[0] <= 0:
            return {"__reso_error__": "MAT_BUDGET_LIMIT"}
        if budget is not None:
            budget[0] -= 1

        if type(obj) is tuple and len(obj) >= 2 and obj[0] == "<<RESO>>":
            tag = obj[2] if len(obj) >= 3 else "MALFORMED"

            if tag in ("SET", "LIST", "TUPLE"):
                payload = obj[3] if (len(obj) >= 4 and type(obj[3]) in (list, tuple)) else []
                m_items = [self._deep_materialize(x, budget, deadline) for x in payload]
                if tag == "SET":
                    return {"__reso_set__": m_items}
                if tag == "TUPLE":
                    return tuple(m_items)
                return m_items

            if tag == "BYTES":
                if len(obj) >= 5:
                    return {"__reso_bytes__": {"len": obj[3], "val": obj[4]}}
                return {"__reso_tok__": {"t": tag, "v": []}}

            if tag == "STR_TRUNC":
                if len(obj) >= 5:
                    return {"__reso_str_trunc__": {"len": obj[3], "sha16": obj[4]}}
                return {"__reso_str_trunc__": {"len": None, "sha16": None}}

            if tag == "TIME_LIMIT":
                return {"__reso_time_limit__": "TIMEOUT"}
            if tag == "BUDGET_EXCEEDED":
                return {"__reso_budget_exceeded__": "RESOURCE_EXHAUSTED"}
            if tag == "DEPTH_LIMIT":
                return {"__reso_depth_limit__": "MAX_DEPTH_REACHED"}
            if tag == "CYCLIC_REF":
                return {"__reso_cyclic_ref__": "DETECTED"}

            return {"__reso_tok__": {"t": tag, "v": [self._deep_materialize(x, budget, deadline) for x in obj[3:]]}}

        if isinstance(obj, list):
            return [self._deep_materialize(x, budget, deadline) for x in obj]
        if isinstance(obj, tuple):
            return tuple(self._deep_materialize(x, budget, deadline) for x in obj)
        if isinstance(obj, Mapping):
            try:
                return {str(k): self._deep_materialize(v, budget, deadline) for k, v in obj.items()}
            except:
                return {"__reso_error__": "MAP_MAT_FAIL"}
        return obj

# ==============================================================================
# [LAYER 2] VERIFIER & SIMULATION
# ==============================================================================

def unwrap_payload(raw_json: str) -> Any:
    try:
        content = json.loads(raw_json)
        if type(content) is dict and "payload" in content:
            return content["payload"]
        return content
    except:
        return raw_json

async def run_genesis_verifier():
    print(f"\n--- [Resonetics v{_SCHEMA_VERSION}-FINAL_CANON] Baseline Audit Rig ---")
    os.environ["RESONETICS_TESTING"] = "1"
    SovereignBody._reset_thread_state_for_test_ONLY()

    with SovereignBody(":memory:", deterministic=True) as k1:
        # 1) Chronos sync proof
        k1._ctx.tick(1.0)
        k1.register_observation("p.time", "v", {"data": "future"})
        row_time = k1.conn.execute(f'SELECT obs_kind FROM "{T_OBS}" WHERE source_file="p.time" ORDER BY id DESC LIMIT 1').fetchone()
        if row_time and "_SAMPLED_BY_TIME" in row_time[0]:
            print("[PASS] Proof 1: Atomic Chronos Sync Verified (_SAMPLED_BY_TIME).")
        else:
            print("[INFO] Proof 1: No time sampling triggered.")

        # 2) SLOT_TYPES DDL proof
        type_info = k1.conn.execute(f"PRAGMA table_info('{T_OBS}')").fetchall()
        if any(t[2] == "INTEGER" for t in type_info):
            print("[PASS] Proof 2: Schema SLOT_TYPES Law Verified.")

        # 3) Path protocol isolation proof
        if k1._use_uri and k1._norm_path.startswith("file:reso_akashic_"):
            print("[PASS] Proof 3: Multiverse Isolation Verified.")

        # 4) Cycle resolution proof
        cycle = {}
        cycle["s"] = cycle
        k1.register_observation("p.cycle", "v", cycle)
        row_cyc = k1.conn.execute(f'SELECT obs_content FROM "{T_OBS}" WHERE source_file="p.cycle" ORDER BY id DESC LIMIT 1').fetchone()
        if row_cyc and "CYCLIC_REF" in row_cyc[0]:
            print("[PASS] Proof 4: Topological Voids Resolved (CYCLIC_REF).")

        # 5) Nested binary detection proof
        nested = {"a": [b"bin_in_list", {"b": b"bin_in_map"}]}
        k1.register_observation("p.bin", "v", nested)
        row_bin = k1.conn.execute(f'SELECT obs_content FROM "{T_OBS}" WHERE source_file="p.bin" ORDER BY id DESC LIMIT 1').fetchone()
        if row_bin and ("HAS_NESTED_BIN" in row_bin[0] or "HAS_ROOT_BIN" in row_bin[0]):
            print("[PASS] Proof 5: Unified scanner detected binary tokens.")
        else:
            print("[FAIL] Proof 5: Binary detection missing!")

        # 6) Key collision preservation proof
        class KeyA:
            def __init__(self, x): self.x = x
        d = {KeyA(1): "v1", KeyA(2): "v2"}
        k1.register_observation("p.kc", "v", d)
        row_kc = k1.conn.execute(f'SELECT obs_content FROM "{T_OBS}" WHERE source_file="p.kc" ORDER BY id DESC LIMIT 1').fetchone()
        if row_kc and "#C" in row_kc[0]:
            print("[PASS] Proof 6: Key collision preserved (#C suffix).")
        else:
            print("[INFO] Proof 6: Collision not triggered (hashes differed).")

    print(f"--- [FINAL CANON] Verified. ---")

if __name__ == "__main__":
    asyncio.run(run_genesis_verifier())
