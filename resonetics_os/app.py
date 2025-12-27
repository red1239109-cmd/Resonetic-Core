# ==============================================================================
# Project: Resonetics
# File:resonetics_os 
# Author: red1239109-cdm
#
# Copyright 2025 red1239109-cdm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os, json, time, hashlib, hmac, secrets, stat, threading, math, errno, ctypes, sys
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timezone
from contextlib import ExitStack, closing

# ===============================
# [Article 0] Constitution
# ===============================
class ResoneticsFatal(RuntimeError): pass
class ResoneticsHeadStale(RuntimeError):
    def __init__(self, msg, payload):
        super().__init__(msg)
        self.payload = payload

# ===============================
# Config
# ===============================
class Config:
    VERSION = "v57.2"
    GENESIS_HASH = "0" * 64

    try:
        _RAW = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _RAW = os.getcwd()

    BASE_DIR = os.path.realpath(os.path.abspath(_RAW))
    if os.name == "nt":
        BASE_DIR = os.path.normcase(os.path.normpath(BASE_DIR))

    @classmethod
    def _canonicalize(cls, path):
        rp = os.path.realpath(os.path.abspath(path))
        if os.name == "nt":
            rp = os.path.normcase(os.path.normpath(rp))
        return rp

    @classmethod
    def within_base(cls, path):
        try:
            return os.path.commonpath([cls.BASE_DIR, cls._canonicalize(path)]) == cls.BASE_DIR
        except Exception:
            return False

    @classmethod
    def get_lock_dir(cls):
        return os.path.join(cls.BASE_DIR, ".locks")

    @classmethod
    def get_graveyard_dir(cls):
        return os.path.join(cls.get_lock_dir(), ".graveyard")

    MAX_HEADER_SIZE = 20
    MAX_FRAME_SIZE = 256 * 1024
    PULSE_INTERVAL = 0.5
    DEV_MODE = os.environ.get("RESONETICS_DEV_MODE") == "1"
    LOCK_TTL = 300.0
    HEARTBEAT_INTERVAL = 15.0
    MAX_SKIP_BYTES = 4096
    COMMIT_INTERVAL = 5.0

    SIG_FIELDS_V1 = ["prev_hash","global_step","physics_step","episode","episode_step","type","status","msg","metrics","meta","time"]
    SIG_FIELDS_V2 = SIG_FIELDS_V1 + ["ver"]

    SNAP_FIELDS_V1 = ["global_step","chain_tip","anchor_hash","physics_step","metrics","timestamp","ver"]
    SNAP_FIELDS_V2 = SNAP_FIELDS_V1 + ["_key_fp","snap_ver","app_ver"]

Config.LEDGER_FILE = os.path.join(Config.BASE_DIR, f"resonetics_{Config.VERSION}.ledger")
Config.HEAD_FILE   = Config.LEDGER_FILE + ".head"
Config.KEY_FILE    = Config.LEDGER_FILE + ".key"
Config.SNAP_FILE   = Config.LEDGER_FILE + ".snap"

Config.CORE_SET = frozenset(
    map(Config._canonicalize, [Config.LEDGER_FILE, Config.HEAD_FILE, Config.KEY_FILE, Config.SNAP_FILE])
)

# ===============================
# Physics
# ===============================
class PhysicsConfig:
    S_BASE = 0.0001
    S_REBUILD_COST = 0.5
    S_HEAL_COST = 0.1
    S_FRICTION_K = 0.00002
    S_VACUUM_REDUCTION = 0.15
    S_COOLING_RATE = 0.005
    PHI_A = 1.25
    PENALTY_DEGRADED = 0.8
    PENALTY_LOCKS_K = 0.002
    HEAT_HIGH = 1.0
    HEAT_CRITICAL = 2.0

@dataclass
class PhysicsState:
    current_s: float = 0.0
    current_phi: float = 1.0
    def heat_level(self):
        if self.current_s >= PhysicsConfig.HEAT_CRITICAL: return "CRITICAL"
        if self.current_s >= PhysicsConfig.HEAT_HIGH: return "HIGH"
        return "NORMAL"

# ===============================
# IO Tracker / Locks
# ===============================
_IO_TRACKER = threading.local()

class TrackedRLock:
    def __init__(self, name, allow_in_io=False):
        self._lock = threading.RLock()
        self.name = name
        self.allow_in_io = allow_in_io
        self._tls = threading.local()

    def held_by_me(self):
        return getattr(self._tls, "count", 0) > 0

    def __enter__(self):
        depth = getattr(_IO_TRACKER, "depth", 0)
        if depth > 0 and not self.allow_in_io:
            raise ResoneticsFatal(f"ARTICLE_4_VIOLATION: {self.name} inside IO")
        self._lock.acquire()
        self._tls.count = getattr(self._tls, "count", 0) + 1
        return self

    def __exit__(self, et, ev, tb):
        self._tls.count -= 1
        self._lock.release()

# ===============================
# Global State (UI-agnostic)
# ===============================
def _gen_nonce():
    return hashlib.sha256(f"{os.getpid()}:{time.time()}:{secrets.token_hex(4)}".encode()).hexdigest()[:16]

@dataclass
class ProcessStateData:
    quarantine: bool = False
    running: bool = False
    last_pulse: float = 0.0
    integrity_degraded: bool = False
    policy_logged: bool = False
    last_error_code: str = "OK"
    last_warn_code: str = "None"
    first_warn_code: str = "None"
    process_nonce: str = field(default_factory=_gen_nonce)
    file_oaths: dict = field(default_factory=dict)
    snapshot_loaded: bool = False
    key_fingerprint: str = None
    boot_logs: deque = field(default_factory=lambda: deque(maxlen=300))
    _lock: TrackedRLock = field(default_factory=lambda: TrackedRLock("STATE"))
    _oath_lock: TrackedRLock = field(default_factory=lambda: TrackedRLock("OATH", allow_in_io=True))

GLOBAL_STATE = ProcessStateData()

# ===============================
# AtomicFileLock / Crypto / Agent
# ===============================
