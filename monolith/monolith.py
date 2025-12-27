# ==============================================================================
# Project: Resonetics
# File:monolith 
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
import os
import json
import time
import hashlib
import hmac
import secrets
import stat
import threading
import math
import errno
import ctypes
import sys
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timezone
from contextlib import ExitStack, closing

import streamlit as st

# ==============================================================================
# [Article 0] The Constitution of Resonetics
# ==============================================================================
# Invariant A: Quarantine implies (Running == False) AND (Snapshot_Loaded == False).
# Invariant B: HEAD(seq, hash) must always equal the last verified chain tip.
# Invariant C: CORE_SET Inodes (Identity) must never change during a session (POSIX).
#
# [Article 4] Concurrency Protocol (Lock Hierarchy)
# 1. AtomicFileLock (Disk IO) and GLOBAL_STATE._lock (Runtime State) must NEVER be held simultaneously.
#    (Exception: _oath_lock is an independent Identity Domain allowed under Disk IO).
# 2. GLOBAL_STATE._lock (State) and GLOBAL_STATE._oath_lock (Identity) are MUTUALLY EXCLUSIVE.
# 3. Sentencing (Quarantine) must occur OUTSIDE of any Disk Locks (via Exception Bubbling).
# ==============================================================================

# ==============================================================================
# [Article 1] Global Configuration (Inherited Oath v57.1)
# ==============================================================================
class Config:
    VERSION = "v57.1" # 👑 Milestone: The Sovereign of Inherited Oath
    GENESIS_HASH = "0" * 64
    
    try:
        _RAW = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _RAW = os.getcwd()
        
    BASE_DIR = os.path.realpath(os.path.abspath(_RAW))
    if os.name == "nt":
        BASE_DIR = os.path.normcase(os.path.normpath(BASE_DIR))
    
    @classmethod
    def _canonicalize(cls, path: str) -> str:
        rp = os.path.realpath(os.path.abspath(path))
        if os.name == "nt": rp = os.path.normcase(os.path.normpath(rp))
        return rp

    @classmethod
    def within_base(cls, path: str) -> bool:
        try:
            target = cls._canonicalize(path)
            return os.path.commonpath([cls.BASE_DIR, target]) == cls.BASE_DIR
        except (ValueError, OSError): return False

    @classmethod
    def get_lock_dir(cls): return os.path.join(cls.BASE_DIR, ".locks")
    @classmethod
    def get_graveyard_dir(cls): return os.path.join(cls.get_lock_dir(), ".graveyard")

    MAX_HEADER_SIZE, MAX_FRAME_SIZE = 20, 256 * 1024
    PULSE_INTERVAL, DEV_MODE = 0.5, os.environ.get("RESONETICS_DEV_MODE") == "1"
    MAX_FORENSIC_EVIDENCE = 5 
    MAX_SNAPSHOT_FAILURES = 3 
    
    MAX_SKIP_BYTES = 4096 
    COMMIT_INTERVAL = 5.0 
    
    SIG_FIELDS_V1 = ["prev_hash", "global_step", "physics_step", "episode", "episode_step", "type", "status", "msg", "metrics", "meta", "time"]
    SIG_FIELDS_V2 = SIG_FIELDS_V1 + ["ver"]
    
    SNAP_FIELDS_V1 = ["global_step", "chain_tip", "anchor_hash", "physics_step", "metrics", "timestamp", "ver"]
    SNAP_FIELDS_V2 = SNAP_FIELDS_V1 + ["_key_fp", "snap_ver", "app_ver"] 
    
    ABSOLUTE_FSYNC = False
    LOCK_TTL = 300.0 
    HEARTBEAT_INTERVAL = 15.0
    VACUUM_UNKNOWN_FACTOR = 2.0

Config.LEDGER_FILE = os.path.join(Config.BASE_DIR, f"resonetics_{Config.VERSION}.ledger")
Config.HEAD_FILE = Config.LEDGER_FILE + ".head"
Config.KEY_FILE = Config.LEDGER_FILE + ".key"
Config.SNAP_FILE = Config.LEDGER_FILE + ".snap"
Config.CORE_SET = frozenset({
    Config._canonicalize(Config.KEY_FILE),
    Config._canonicalize(Config.HEAD_FILE),
    Config._canonicalize(Config.LEDGER_FILE),
    Config._canonicalize(Config.SNAP_FILE)
})

LOCK_IMPL_TYPE = "None"
try:
    import fcntl; LOCK_IMPL_TYPE = "fcntl (POSIX)"
except ImportError:
    try:
        import msvcrt; LOCK_IMPL_TYPE = "msvcrt (Windows)"
    except ImportError: pass

IS_POSIX = "fcntl" in LOCK_IMPL_TYPE
IS_WINLOCK = "msvcrt" in LOCK_IMPL_TYPE

class ResoneticsFatal(RuntimeError):
    """Irrecoverable system error requiring quarantine."""
    pass

class ResoneticsHeadStale(RuntimeError):
    """Ledger committed but Head update failed. Payload attached."""
    def __init__(self, msg, payload):
        super().__init__(msg)
        self.payload = payload

class PhysicsConfig:
    S_BASE, S_REBUILD_COST, S_HEAL_COST = 0.0001, 0.5, 0.1
    S_FRICTION_K, S_VACUUM_REDUCTION = 0.00002, 0.15
    S_COOLING_RATE = 0.005 
    PHI_A, PENALTY_DEGRADED, PENALTY_LOCKS_K = 1.25, 0.8, 0.002
    HEAT_HIGH, HEAT_CRITICAL = 1.0, 2.0

@dataclass
class PhysicsState:
    current_s: float = 0.0
    current_phi: float = 1.0
    def heat_level(self) -> str:
        if self.current_s >= PhysicsConfig.HEAT_CRITICAL: return "CRITICAL"
        if self.current_s >= PhysicsConfig.HEAT_HIGH: return "HIGH"
        return "NORMAL"

_IO_TRACKER = threading.local()

class TrackedRLock:
    def __init__(self, name: str, allow_in_io: bool = False):
        self._lock = threading.RLock()
        self.name = name
        self.allow_in_io = allow_in_io
        self._tls = threading.local()

    def held_by_me(self) -> bool:
        return getattr(self._tls, "count", 0) > 0

    def __enter__(self):
        io_depth = getattr(_IO_TRACKER, "depth", 0)
        if io_depth > 0 and not self.allow_in_io:
            raise ResoneticsFatal(f"ARTICLE_4_1_VIOLATION: Attempted to grab {self.name} Lock inside Disk IO Context (depth={io_depth})")
            
        self._lock.acquire()
        self._tls.count = getattr(self._tls, "count", 0) + 1
        return self

    def __exit__(self, et, ev, tb):
        try:
            new_count = getattr(self._tls, "count", 0) - 1
            if Config.DEV_MODE and new_count < 0:
                self._tls.count = 0
                raise RuntimeError(f"TrackedRLock underflow: {self.name}")
            self._tls.count = new_count
        finally:
            self._lock.release()

    def acquire(self): 
        self.__enter__()
        return True
    
    def release(self):
        self.__exit__(None, None, None)

# ==============================================================================
# [Article 2] Sensing & Resilient State
# ==============================================================================
st.set_page_config(layout="wide", page_title=f"Resonetics {Config.VERSION}")

def _generate_unique_nonce():
    seed = f"{os.getpid()}:{time.time()}:{secrets.token_hex(4)}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]

@dataclass
class ProcessStateData:
    quarantine: bool = False
    running: bool = False
    last_pulse: float = 0.0
    integrity_degraded: bool = False
    policy_logged: bool = False
    
    no_file_retry_logged: set = field(default_factory=set)
    
    last_error_code: str = "OK"
    last_warn_code: str = "None"
    first_warn_code: str = "None" 
    
    process_nonce: str = field(default_factory=_generate_unique_nonce) 
    base_dir_anchor: tuple = None
    file_oaths: dict = field(default_factory=dict) 
    snapshot_fail_count: int = 0
    boot_logs: deque = field(default_factory=lambda: deque(maxlen=300), repr=False)
    
    _lock: TrackedRLock = field(default_factory=lambda: TrackedRLock("STATE"), repr=False)
    _oath_lock: TrackedRLock = field(default_factory=lambda: TrackedRLock("OATH", allow_in_io=True), repr=False)
    
    tail_heal_observed, head_rebuild_observed, vacuum_observed = False, False, False
    snapshot_loaded: bool = False
    last_audit_time, last_audit_result, last_audit_frames = 0.0, "-", "0/0"
    
    key_fingerprint: str = None 
    head_stale: bool = False
    last_commit_time: float = 0.0 

    def trigger_quarantine(self, reason: str):
        _guard_no_oath_lock(f"trigger_quarantine({reason})")
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"💀 {timestamp} QUARANTINE TRIGGERED: {reason}"
        
        with self._lock: 
            if self.quarantine: return 
            self.quarantine = True
            self.running = False
            self.snapshot_loaded = False
            self.integrity_degraded = True
            
            if self.last_error_code == "OK":
                self.last_error_code = reason[:40]
            
            self.boot_logs.append(log_entry)

if "GLOBAL_STATE" not in st.session_state:
    st.session_state.GLOBAL_STATE = ProcessStateData()

GLOBAL_STATE = st.session_state.GLOBAL_STATE
PROCESS_NONCE = GLOBAL_STATE.process_nonce

def _guard_no_state_lock(reason=""):
    if GLOBAL_STATE._lock.held_by_me():
        raise ResoneticsFatal(f"ARTICLE_4_1_VIOLATION: StateLock held during IO ({reason})")

def _guard_no_oath_lock(reason=""):
    if GLOBAL_STATE._oath_lock.held_by_me():
        raise ResoneticsFatal(f"ARTICLE_4_2_VIOLATION: OathLock held during State ops ({reason})")

def _guard_no_state_lock_during_oath(reason=""):
    if GLOBAL_STATE._lock.held_by_me():
        raise ResoneticsFatal(f"ARTICLE_4_2_VIOLATION: StateLock held during Oath ops ({reason})")

# ✅ Fix 5: Silent Record (Bypass STATE lock if in IO context)
def log_boot(msg, level="INFO"):
    if getattr(_IO_TRACKER, "depth", 0) > 0:
        # Cannot grab STATE lock here. Emit to stderr.
        if Config.DEV_MODE:
            print(f"[IO_SAFE_LOG] {level}: {msg}", file=sys.stderr)
        return

    _guard_no_oath_lock(f"log_boot({msg[:20]})")
    prefix_map = {"INFO": "🛡️ ", "WARN": "⚠️ ", "ERROR": "🚨 ", "FATAL": "💀 "}
    prefix = prefix_map.get(level, "📝 ")
    entry = f"{prefix}{datetime.now().strftime('%H:%M:%S')} {msg}"
    try:
        with GLOBAL_STATE._lock:
            GLOBAL_STATE.boot_logs.append(entry)
            if level in ("ERROR", "FATAL"):
                if GLOBAL_STATE.last_error_code == "OK": GLOBAL_STATE.last_error_code = msg[:40]
            elif level == "WARN":
                GLOBAL_STATE.last_warn_code = msg[:40]
                if GLOBAL_STATE.first_warn_code == "None": GLOBAL_STATE.first_warn_code = msg[:40]
    except Exception as e:
        if Config.DEV_MODE:
            print(f"[LOG_BOOT_FAIL] {type(e).__name__}: {e}", file=sys.stderr)

def _warn_if_io_depth(tag=""):
    d = getattr(_IO_TRACKER, "depth", 0)
    if d > 0:
        # log_boot handles IO check now, but explicit warning is good
        if Config.DEV_MODE:
            print(f"WARN: RERUN_WITH_IO_DEPTH:{d} ({tag})", file=sys.stderr)

def fsync_dir(path):
    if not IS_WINLOCK:
        canon_path = Config._canonicalize(path)
        dir_p = canon_path if os.path.isdir(canon_path) else os.path.dirname(canon_path)
        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"): flags |= os.O_DIRECTORY
            if hasattr(os, "O_CLOEXEC"): flags |= os.O_CLOEXEC
            fd = os.open(dir_p, flags)
            try: os.fsync(fd)
            finally: os.close(fd)
        except: pass

def base_dir_safety_check():
    logs_to_emit = []
    try:
        bd_raw, bd_canon = Config._RAW, Config.BASE_DIR
        raw_stat = os.lstat(bd_raw); canon_stat = os.lstat(bd_canon)
        is_symlink = stat.S_ISLNK(raw_stat.st_mode)
        is_world_writable = False; is_junction = False
        if os.name != "nt":
            if (canon_stat.st_mode & 0o002) != 0: is_world_writable = True
        else:
            attrs = getattr(canon_stat, "st_file_attributes", None)
            if attrs and (attrs & 0x400): is_junction = True
        
        if os.name == "nt": current_anchor = (Config.BASE_DIR, is_junction)
        else: current_anchor = (canon_stat.st_dev, canon_stat.st_ino)

        should_degrade = False
        if is_symlink: logs_to_emit.append(("BASE_DIR is symbolic.", "WARN")); should_degrade = True
        if is_world_writable: logs_to_emit.append(("BASE_DIR is world-writable.", "WARN")); should_degrade = True
        if is_junction: logs_to_emit.append(("BASE_DIR junction detected.", "WARN")); should_degrade = True

        with GLOBAL_STATE._lock:
            if GLOBAL_STATE.base_dir_anchor is None: GLOBAL_STATE.base_dir_anchor = current_anchor
            elif GLOBAL_STATE.base_dir_anchor != current_anchor:
                should_degrade = True
                logs_to_emit.append(("BASE_DIR Drift Detected!", "WARN"))
            if should_degrade: GLOBAL_STATE.integrity_degraded = True
            if IS_WINLOCK and not GLOBAL_STATE.policy_logged: GLOBAL_STATE.policy_logged = True
    except Exception as e: logs_to_emit.append((f"Safety fail: {e}", "WARN"))
    for msg, level in logs_to_emit: log_boot(msg, level)

def get_process_status(pid: int):
    if os.name == 'nt':
        try:
            kernel32 = ctypes.windll.kernel32
            if not getattr(kernel32, "_resonetics_typed", False):
                kernel32.SetLastError.argtypes = [ctypes.c_ulong]; kernel32.SetLastError.restype = None
                kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]; kernel32.OpenProcess.restype = ctypes.c_void_p
                kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]; kernel32.GetExitCodeProcess.restype = ctypes.c_bool 
                kernel32.CloseHandle.argtypes = [ctypes.c_void_p]; kernel32.CloseHandle.restype = ctypes.c_bool
                kernel32.GetLastError.argtypes = []; kernel32.GetLastError.restype = ctypes.c_ulong
                kernel32._resonetics_typed = True
            SYNCHRONIZE = 0x00100000; PROCESS_QUERY_LIMITED_INFORMATION = 0x1000; STILL_ACTIVE = 259
            kernel32.SetLastError(0)
            process = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, ctypes.c_bool(False), pid)
            if not process: return None 
            exit_code = ctypes.c_ulong(); success = kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code))
            if not kernel32.CloseHandle(process): pass
            if success: return exit_code.value == STILL_ACTIVE
            return None 
        except: return None
    else:
        try: os.kill(pid, 0); return True
        except ProcessLookupError: return False
        except PermissionError: return True
        except: return None

def _vacuum_safe_read(path):
    rp = Config._canonicalize(path); lock_dir = Config._canonicalize(Config.get_lock_dir())
    if os.path.commonpath([lock_dir, rp]) != lock_dir: raise RuntimeError("Vacuum Escape")
    if not Config.within_base(rp): raise RuntimeError("Vacuum Escape Base")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
    fd = os.open(rp, flags)
    try:
        stf = os.fstat(fd)
        if not stat.S_ISREG(stf.st_mode): raise RuntimeError("Not Regular")
        f = os.fdopen(fd, "rb"); fd = None 
        with f: return f.read()
    finally:
        if fd is not None:
            try: os.close(fd)
            except: pass

def vacuum_stale_locks(lock_dir: str) -> tuple:
    if not os.path.isdir(lock_dir) or not Config.within_base(lock_dir): return (0, 0)
    
    graveyard = Config.get_graveyard_dir()
    if not os.path.exists(graveyard):
        try: os.makedirs(graveyard, exist_ok=True)
        except: pass
        
    rem_locks, rem_bad = 0, 0
    now = time.time()
    for fn in os.listdir(lock_dir):
        if not fn.endswith(".lock"): continue
        fp = os.path.join(lock_dir, fn)
        ok_to_remove = False 
        try:
            age = now - os.path.getmtime(fp)
            if age > Config.LOCK_TTL:
                content = _vacuum_safe_read(fp).strip().decode(errors='ignore')
                if ":" in content and content.startswith("X:"):
                    parts = content.split(":", 2) 
                    if len(parts) >= 3:
                        pid_str, nonce = parts[1], parts[2]
                        try: pid = int(pid_str)
                        except: ok_to_remove = False 
                        else:
                            if pid == os.getpid(): ok_to_remove = False
                            else:
                                status = get_process_status(pid)
                                if status is True: ok_to_remove = False
                                elif status is False: ok_to_remove = True
                                else:
                                    if age > (Config.LOCK_TTL * Config.VACUUM_UNKNOWN_FACTOR): ok_to_remove = True
                                    else: ok_to_remove = False
                    else: ok_to_remove = False 
                else: 
                    if age > (Config.LOCK_TTL * Config.VACUUM_UNKNOWN_FACTOR * 2): ok_to_remove = True
                    else: ok_to_remove = False
        except Exception: ok_to_remove = False
        
        if ok_to_remove:
            try:
                dest_fn = f"{fn}.{int(time.time())}.{secrets.token_hex(4)}"
                dest = os.path.join(graveyard, dest_fn)
                os.replace(fp, dest)
                rem_locks += 1
            except: 
                try: os.remove(fp); rem_locks += 1
                except: pass
            
    try:
        base_dir = Config.BASE_DIR; bad_snaps = []
        for fn in os.listdir(base_dir):
            if fn.startswith(os.path.basename(Config.SNAP_FILE) + ".bad."): bad_snaps.append(os.path.join(base_dir, fn))
        if len(bad_snaps) > Config.MAX_FORENSIC_EVIDENCE:
            bad_snaps.sort(key=os.path.getmtime) 
            to_remove = bad_snaps[:len(bad_snaps) - Config.MAX_FORENSIC_EVIDENCE]
            for tr in to_remove:
                try: os.remove(tr); rem_bad += 1
                except: pass
    except: pass
    
    try:
        for fn in os.listdir(graveyard):
            fp = os.path.join(graveyard, fn)
            if time.time() - os.path.getmtime(fp) > (Config.LOCK_TTL * 10): 
                try: os.remove(fp)
                except: pass
    except: pass

    return (rem_locks, rem_bad)

# ==============================================================================
# [Article 3] The Citadel Engine (Isolated Core)
# ==============================================================================
class AtomicFileLock:
    def __init__(self, target_filename, exclusive=True, timeout=5.0, dev_mode=None):
        self.target_abs = Config._canonicalize(target_filename)
        if not Config.within_base(self.target_abs): raise RuntimeError(f"⛔ Escape: {target_filename}")
        self.LOCK_DIR = Config.get_lock_dir()
        self.dev_mode = dev_mode if dev_mode is not None else Config.DEV_MODE
        self.exclusive = exclusive
        self.timeout = timeout
        if not os.path.isdir(self.LOCK_DIR):
            try: os.makedirs(self.LOCK_DIR, exist_ok=True)
            except: pass
        if os.name == "nt" and os.path.isdir(self.LOCK_DIR):
            try:
                if getattr(os.lstat(self.LOCK_DIR), "st_file_attributes", 0) & 0x400: raise RuntimeError("⛔ LOCK_DIR Junction.")
            except OSError as e:
                if not self.dev_mode: raise RuntimeError("⛔ LOCK_DIR stat fail.") from e
        if IS_POSIX:
            st_dir = os.lstat(self.LOCK_DIR)
            if stat.S_ISLNK(st_dir.st_mode): raise RuntimeError("⛔ LOCK_DIR is symlink.")
            if not stat.S_ISDIR(st_dir.st_mode): raise RuntimeError("⛔ LOCK_DIR is not a directory.")
            try: os.chmod(self.LOCK_DIR, 0o700)
            except: pass
            st_dir = os.lstat(self.LOCK_DIR)
            if (st_dir.st_mode & 0o777) != 0o700: raise RuntimeError("⛔ Perms 0700 required.")
            if hasattr(os, "getuid") and st_dir.st_uid != os.getuid(): raise RuntimeError("⛔ LOCK_DIR Owner mismatch.")
        self.lock_file = os.path.join(self.LOCK_DIR, f"{hashlib.sha256(self.target_abs.encode()).hexdigest()[:16]}.lock")
        self.handle = None; self._lockmod = None; self._nonce = PROCESS_NONCE

    def touch(self):
        try:
            os.utime(self.lock_file, None)
            return True, None
        except Exception as e:
            if self.dev_mode or Config.DEV_MODE:
                return False, f"LOCK_TOUCH_FAIL: {os.path.basename(self.lock_file)} {type(e).__name__}:{str(e)[:80]}"
            return False, None

    def __enter__(self):
        if GLOBAL_STATE._lock.held_by_me():
             raise ResoneticsFatal("ARTICLE_4_1_VIOLATION: Attempted Disk IO while holding State Lock")
             
        try:
            if IS_POSIX:
                import fcntl; self._lockmod = fcntl
                flags = os.O_RDWR | os.O_CREAT
                if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
                if hasattr(os, "O_CLOEXEC"):  flags |= os.O_CLOEXEC
                fd = os.open(self.lock_file, flags, 0o600)
                try: os.fchmod(fd, 0o600)
                except: pass
                stf = os.fstat(fd)
                if (not stat.S_ISREG(stf.st_mode)) or (hasattr(os, "getuid") and stf.st_uid != os.getuid()): os.close(fd); raise RuntimeError("⛔ Lock Inode/Owner fail.")
                self.handle = os.fdopen(fd, "a+b")
            elif IS_WINLOCK:
                import msvcrt; self._lockmod = msvcrt
                try: self.handle = open(self.lock_file, "xb" if self.exclusive else "a+b")
                except: self.handle = open(self.lock_file, "a+b")
                st2 = os.fstat(self.handle.fileno())
                if not stat.S_ISREG(st2.st_mode): self.handle.close(); raise RuntimeError("⛔ Lock Deception.")
                self.handle.seek(0, os.SEEK_END)
                if self.handle.tell() == 0: self.handle.write(b"\0"); self.handle.flush()
            else:
                if not self.dev_mode: raise RuntimeError("⛔ No lock engine.")
                return self
            start_t = time.time()
            while True:
                try:
                    if IS_POSIX: self._lockmod.flock(self.handle.fileno(), (self._lockmod.LOCK_EX if self.exclusive else self._lockmod.LOCK_SH)|self._lockmod.LOCK_NB)
                    elif IS_WINLOCK: 
                        if self.exclusive: self.handle.seek(0); self._lockmod.locking(self.handle.fileno(), self._lockmod.LK_NBLCK, 1)
                        else: self.handle.seek(0); self._lockmod.locking(self.handle.fileno(), self._lockmod.LK_NBRLCK, 1)
                    if self.exclusive:
                        self.handle.seek(0); self.handle.truncate(); self.handle.write(f"X:{os.getpid()}:{self._nonce}".encode()); self.handle.flush()
                    else: self.touch()
                    break
                except (IOError, OSError):
                    if time.time() - start_t > self.timeout: raise TimeoutError(f"Lock timeout: {os.path.basename(self.target_abs)} ({os.path.basename(self.lock_file)})")
                    time.sleep(0.05)
            
            setattr(_IO_TRACKER, "depth", getattr(_IO_TRACKER, "depth", 0) + 1)
            return self
        except Exception as e:
            if self.handle: 
                try: self.handle.close()
                except: pass
            e_str = str(e)
            if "⛔" in e_str: raise ResoneticsFatal(f"LOCK_INTEGRITY: {e_str}") from e
            if isinstance(e, (TimeoutError, BlockingIOError)): raise
            raise

    def __exit__(self, et, ev, tb):
        try:
            if self.handle and getattr(self, "_lockmod", None):
                try:
                    if IS_POSIX: self._lockmod.flock(self.handle.fileno(), self._lockmod.LOCK_UN)
                    elif IS_WINLOCK: self.handle.seek(0); self._lockmod.locking(self.handle.fileno(), self._lockmod.LK_UNLCK, 1)
                finally:
                    try: self.handle.close()
                    except: pass
        finally:
            current = getattr(_IO_TRACKER, "depth", 0)
            if current > 0: setattr(_IO_TRACKER, "depth", current - 1)

class ResoneticCrypto:
    def __init__(self, key_path):
        self.secret = None
        with AtomicFileLock(key_path, exclusive=True):
            if os.path.exists(key_path):
                with self.secure_open(key_path, "rb") as f: self.secret = f.read()
            
            if (not self.secret) or (len(self.secret) != 32):
                if not Config.DEV_MODE:
                     raise ResoneticsFatal("⛔ KEY_MISSING_IN_PROD: Identity must be immutable.")
                
                self.secret = secrets.token_bytes(32)
                with self.secure_open(key_path, "xb", exclusive=True) as f: 
                    f.write(self.secret); f.flush(); os.fsync(f.fileno())
        
        self.fingerprint = hashlib.sha256(self.secret).hexdigest()[:12]
        with GLOBAL_STATE._lock:
            GLOBAL_STATE.key_fingerprint = self.fingerprint

    @staticmethod
    def secure_open(filepath, mode="rb", exclusive=False):
        _guard_no_state_lock(f"secure_open({os.path.basename(filepath)})")
        rp = Config._canonicalize(filepath)
        if not Config.within_base(rp): raise RuntimeError("⛔ Territorial Escape.")
        if not mode or mode[0] not in ("r", "w", "a", "x"): raise ValueError("Invalid mode.")
        if "b" not in mode: mode += "b"
        lock_dir = Config._canonicalize(Config.get_lock_dir())
        crit_prefixes = [p + "." for p in [Config.HEAD_FILE, Config.SNAP_FILE, Config.LEDGER_FILE]]
        is_temp_critical = rp.endswith(".tmp") and any(rp.startswith(p) for p in crit_prefixes)
        is_critical = ((rp in Config.CORE_SET) or (rp.endswith(".lock") and os.path.commonpath([lock_dir, rp]) == lock_dir) or is_temp_critical)
        if IS_POSIX:
            m_char, rw = mode[0], "+" in mode
            if m_char == "r": flags = os.O_RDWR if rw else os.O_RDONLY
            elif m_char == "w": flags = (os.O_RDWR if rw else os.O_WRONLY) | os.O_CREAT | os.O_TRUNC
            elif m_char == "a": flags = (os.O_RDWR if rw else os.O_WRONLY) | os.O_CREAT | os.O_APPEND
            elif m_char == "x": flags = (os.O_RDWR if rw else os.O_WRONLY) | os.O_CREAT | os.O_EXCL
            else: raise ValueError(f"Invalid mode: {mode}")
            if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
            if hasattr(os, "O_CLOEXEC"):  flags |= os.O_CLOEXEC
            fd = os.open(rp, flags, 0o600)
            try:
                if m_char in "wax": 
                    try: os.fchmod(fd, 0o600)
                    except: pass
                stf = os.fstat(fd)
                if not stat.S_ISREG(stf.st_mode): raise RuntimeError("⛔ Inode fail.")
                owner_match = (not hasattr(os, "getuid")) or (stf.st_uid == os.getuid())
                if (m_char in "wax" or is_critical) and not owner_match: raise RuntimeError("⛔ Owner mismatch.")
                if is_critical and rp in Config.CORE_SET:
                    new_oath = (stf.st_dev, stf.st_ino)
                    try:
                        with GLOBAL_STATE._oath_lock:
                            old_oath = GLOBAL_STATE.file_oaths.get(rp)
                            if old_oath and old_oath != new_oath: raise ResoneticsFatal(f"⛔ Identity Oath Broken for {os.path.basename(rp)}")
                            elif not old_oath: GLOBAL_STATE.file_oaths[rp] = new_oath
                    except NameError: pass
                return os.fdopen(fd, mode)
            except:
                try: os.close(fd)
                except: pass
                raise
        else:
            if is_critical and os.path.exists(rp):
                attrs = getattr(os.lstat(rp), "st_file_attributes", None)
                if (attrs and (attrs & 0x400)) or os.path.islink(rp): raise ResoneticsFatal("⛔ Windows Integrity Guard.")
            if exclusive and mode not in ("xb", "x+b"): raise ValueError("⛔ Mode Violation.")
            f = open(rp, mode)
            try:
                st2 = os.fstat(f.fileno())
                if is_critical and (not stat.S_ISREG(st2.st_mode)): f.close(); raise RuntimeError("⛔ Non-regular file.")
                return f
            except: 
                try: f.close()
                except: pass
                raise

    def _read_no_oath(self, filepath):
        rp = Config._canonicalize(filepath)
        if not Config.within_base(rp): raise ResoneticsFatal(f"ESCAPE_READ_NO_OATH: {os.path.basename(rp)}")
        if rp not in Config.CORE_SET: raise ResoneticsFatal(f"SCOPE_VIOLATION_NO_OATH: {os.path.basename(rp)}")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):  flags |= os.O_CLOEXEC
        fd = os.open(rp, flags)
        try:
            stf = os.fstat(fd)
            if not stat.S_ISREG(stf.st_mode): raise ResoneticsFatal(f"NOT_REGULAR_NO_OATH: {os.path.basename(rp)}")
            if IS_POSIX and stf.st_uid != os.getuid(): raise ResoneticsFatal(f"OWNER_MISMATCH_NO_OATH: {os.path.basename(rp)}")
            f = os.fdopen(fd, "rb"); fd = None 
            with f: return f.read()
        finally:
            if fd is not None:
                try: os.close(fd)
                except: pass

    def _renew_oath(self, filepath):
        if not IS_POSIX: return 
        rp = Config._canonicalize(filepath)
        if rp not in Config.CORE_SET: return
        _guard_no_state_lock_during_oath(f"_renew_oath({os.path.basename(filepath)})")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
        fd = -1
        try:
            fd = os.open(rp, flags); st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode): return
            if st.st_uid != os.getuid(): raise ResoneticsFatal(f"OWNER_MISMATCH_RENEW_OATH: {os.path.basename(rp)}")
            with GLOBAL_STATE._oath_lock: GLOBAL_STATE.file_oaths[rp] = (st.st_dev, st.st_ino)
        except ResoneticsFatal: raise
        except: pass
        finally:
            if fd != -1:
                try: os.close(fd)
                except: pass

    def canonical_str(self, entry: dict, fields=None) -> str:
        def _deep(v, depth=0):
            if depth > 6: return "__DEPTH_CUT__"
            if v is None or isinstance(v, (bool, int)): return v
            if isinstance(v, float): return round(v, 6)
            if isinstance(v, str): return v[:500] + ("__TRUNC__" if len(v) > 500 else "")
            if isinstance(v, dict):
                items = sorted(list(v.items()), key=lambda x: str(x[0]))
                p = {str(k): _deep(val, depth+1) for k, val in items[:50]}
                if len(items) > 50: p["__TRUNC__"] = True
                return p
            if isinstance(v, (list, tuple)):
                items = list(v)
                p = [_deep(x, depth+1) for x in items[:50]]
                if len(items) > 50: p.append("__TRUNC__")
                return p
            return str(v)
        if not fields: raise ValueError("Canonicalization requires explicit fields.")
        for k in fields:
            if k not in entry: raise ValueError(f"SCHEMA VIOLATION: Missing '{k}'")
        payload = {k: _deep(entry[k]) for k in fields}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def verify_entry(self, entry: dict):
        try:
            is_ledger = "prev_hash" in entry
            if is_ledger:
                fields = Config.SIG_FIELDS_V2 if "ver" in entry else Config.SIG_FIELDS_V1
            else:
                fields = Config.SNAP_FIELDS_V2 if "_key_fp" in entry else Config.SNAP_FIELDS_V1
                
            cstr = self.canonical_str(entry, fields)
            sig = hmac.new(self.secret, cstr.encode(), hashlib.sha256).hexdigest().lower()
            ok = hmac.compare_digest(sig, entry.get("sig", "").lower())
            return ok, ("OK" if ok else "SIG_MISMATCH")
        except Exception as e: return False, str(e)

    def commit_transaction(self, filepath, head_path, builder, expected=None):
        _guard_no_state_lock("commit_transaction")
        l1, l2 = AtomicFileLock(head_path), AtomicFileLock(filepath)
        pairs = sorted([(os.path.basename(l1.lock_file), l1), (os.path.basename(l2.lock_file), l2)], key=lambda x: x[0])
        try:
            with pairs[0][1], pairs[1][1]:
                anchor = (0, Config.GENESIS_HASH); valid = False
                if os.path.exists(head_path):
                    with self.secure_open(head_path, "rb") as f:
                        p = f.read().decode().strip().split()
                        if len(p) == 2 and p[0].isdigit() and len(p[1]) == 64: anchor = (int(p[0]), p[1].lower()); valid = True
                        else: raise RuntimeError("HEAD_NEEDS_REBUILD")
                elif expected: raise ResoneticsFatal("HEAD_NEEDS_REBUILD")
                if expected and (not valid or anchor[1] != expected): raise ResoneticsFatal("SPLIT_BRAIN" if valid else "HEAD_NEEDS_REBUILD")
                
                raw_payload = builder(anchor[0] + 1, anchor[1])
                if not isinstance(raw_payload, dict): raise ResoneticsFatal(f"BUILDER_VIOLATION: Expected dict, got {type(raw_payload)}")
                payload = dict(raw_payload); payload.setdefault("ver", "v2")
                try: cstr = self.canonical_str(payload, Config.SIG_FIELDS_V2)
                except ValueError as e: raise ResoneticsFatal(f"SCHEMA_VIOLATION: {str(e)[:80]}") from e
                payload["sig"] = hmac.new(self.secret, cstr.encode(), hashlib.sha256).hexdigest().lower()
                payload["hash"] = hashlib.sha256((cstr + "|sig:" + payload["sig"]).encode()).hexdigest().lower()
                
                with self.secure_open(filepath, "ab") as f:
                    d = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
                    f.write(f"{len(d)}:".encode() + d + b"\n"); f.flush(); os.fsync(f.fileno())
                    if Config.ABSOLUTE_FSYNC: fsync_dir(filepath)
                tmp = f"{head_path}.{secrets.token_hex(4)}.tmp"
                with self.secure_open(tmp, "xb", exclusive=True) as f:
                    f.write(f"{payload['global_step']} {payload['hash']}\n".encode()); f.flush(); os.fsync(f.fileno())
                try:
                    os.replace(tmp, head_path)
                    
                    verified = False
                    for _ in range(3):
                        try:
                            raw = self._read_no_oath(head_path).decode().strip().split()
                            if len(raw)==2 and raw[0]==str(payload["global_step"]) and raw[1].lower()==payload["hash"].lower():
                                verified = True; break
                        except: pass
                        time.sleep(0.03)
                    
                    if not verified: raise ResoneticsFatal("HEAD_VERIFY_FAIL_AFTER_REPLACE")
                    self._renew_oath(head_path)
                    
                except OSError as e:
                    try: os.remove(tmp)
                    except: pass
                    if expected: raise ResoneticsHeadStale(f"HEAD_STALE_AFTER_LEDGER_APPEND: {e}", payload)
                    else: raise
                fsync_dir(head_path); return payload
        except Exception: raise

    def force_rebuild_head(self, head_path, seq, last_hash):
        with AtomicFileLock(head_path, exclusive=True):
            tmp = f"{head_path}.{secrets.token_hex(4)}.tmp"
            with self.secure_open(tmp, "xb", exclusive=True) as f:
                f.write(f"{seq} {last_hash.lower()}\n".encode()); f.flush(); os.fsync(f.fileno())
            try:
                os.replace(tmp, head_path)
                raw = self._read_no_oath(head_path).decode().strip().split()
                if not (len(raw)==2 and raw[0]==str(seq) and raw[1].lower()==last_hash.lower()): raise ResoneticsFatal("HEAD_REBUILD_VERIFY_FAIL")
                self._renew_oath(head_path)
            except OSError as e:
                try: os.remove(tmp)
                except: pass
                raise ResoneticsFatal(f"HEAD_REBUILD_FAIL: {e}") from e
            fsync_dir(head_path)

    def truncate_ledger_at(self, filepath, offset):
        if not os.path.exists(filepath): return
        with AtomicFileLock(filepath, exclusive=True):
            with self.secure_open(filepath, "r+b") as f:
                f.seek(offset); f.truncate(); f.flush(); os.fsync(f.fileno())
        if Config.ABSOLUTE_FSYNC: fsync_dir(filepath)

    def _purge_snapshot_locked(self, snap_path, reason):
        if not os.path.exists(snap_path): return True, f"PURGE_SNAP: Already gone ({reason})"
        try:
            bad_path = f"{snap_path}.bad.{int(time.time())}.{secrets.token_hex(2)}"
            os.replace(snap_path, bad_path)
            return True, f"PURGE_SNAP: Moved to {os.path.basename(bad_path)} ({reason})"
        except OSError as e:
            if e.errno == errno.EXDEV: return False, f"PURGE_SNAP_EXDEV: Cannot move across devices ({reason})"
            if e.errno == errno.ENOENT: return True, f"PURGE_SNAP: Already gone (Race) ({reason})"
            if e.errno in (errno.EACCES, errno.EPERM): return False, f"PURGE_SNAP_DEFERRED: {e} ({reason})"
        except Exception as e: return False, f"PURGE_SNAP_UNKNOWN: {e} ({reason})"
        try: os.remove(snap_path); return True, f"PURGE_SNAP: Removed (Move Failed) ({reason})"
        except Exception as e: return False, f"PURGE_SNAP_FAIL: {e} ({reason})"

    def purge_snapshot(self, snap_path, reason):
        _guard_no_state_lock("purge_snapshot")
        with AtomicFileLock(snap_path, exclusive=True):
            return self._purge_snapshot_locked(snap_path, reason)

    def save_snapshot(self, snap_path, state, anchor_hash):
        _guard_no_state_lock("save_snapshot")
        
        with GLOBAL_STATE._lock: fp = GLOBAL_STATE.key_fingerprint
        
        ok = False
        with AtomicFileLock(snap_path, exclusive=True):
            payload = dict(state)
            payload["anchor_hash"] = anchor_hash
            payload["chain_tip"] = anchor_hash
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            if fp: 
                payload["_key_fp"] = fp
                payload["snap_ver"] = "v2" 
                payload["app_ver"] = Config.VERSION 
                # ✅ Fix 4: Inherited Oath (Fill 'ver' for V1 compatibility)
                payload["ver"] = "v2"
                fields = Config.SNAP_FIELDS_V2
            else:
                payload["ver"] = "v2"
                fields = Config.SNAP_FIELDS_V1

            cstr = self.canonical_str(payload, fields)
            
            payload["sig"] = hmac.new(self.secret, cstr.encode(), hashlib.sha256).hexdigest().lower()
            tmp = f"{snap_path}.{secrets.token_hex(4)}.tmp"
            with self.secure_open(tmp, "xb", exclusive=True) as f:
                f.write(json.dumps(payload, ensure_ascii=False).encode()); f.flush(); os.fsync(f.fileno())
            try:
                os.replace(tmp, snap_path)
                raw = self._read_no_oath(snap_path)
                data = json.loads(raw.decode("utf-8"))
                
                ver_ok, _ = self.verify_entry(data)
                if not ver_ok: self._purge_snapshot_locked(snap_path, "SNAP_VERIFY_FAIL_AFTER_REPLACE"); raise ResoneticsFatal("SNAP_VERIFY_FAIL_AFTER_REPLACE")
                self._renew_oath(snap_path); ok = True
            except OSError as e:
                try: os.remove(tmp)
                except: pass
                raise OSError(f"SNAP_REPLACE_FAIL: {e}") from e
            except ResoneticsFatal:
                try: os.remove(tmp)
                except: pass
                raise
        if ok: fsync_dir(snap_path)
        return ok

    def load_snapshot(self, snap_path):
        _guard_no_state_lock("load_snapshot")
        if not os.path.exists(snap_path): return None, None
        try:
            with AtomicFileLock(snap_path, exclusive=False):
                raw = self._read_no_oath(snap_path)
                data = json.loads(raw.decode("utf-8"))
                
                ok, _ = self.verify_entry(data)
                if not ok: return None, "SIG_MISMATCH"
                else: self._renew_oath(snap_path); return data, None
        except (json.JSONDecodeError, UnicodeDecodeError) as e: return None, f"JSON_FAIL:{e}"
        except ResoneticsFatal: raise 
        except Exception: raise

    @staticmethod
    def stream_frames(filepath, head_path=None, lock_read=False, strict=True):
        if not os.path.exists(filepath):
            logs_to_emit = []
            try:
                fname = os.path.relpath(Config._canonicalize(filepath), Config.BASE_DIR)
            except: fname = os.path.basename(filepath)
            
            with GLOBAL_STATE._lock:
                if fname not in GLOBAL_STATE.no_file_retry_logged:
                    GLOBAL_STATE.no_file_retry_logged.add(fname)
                    logs_to_emit.append((f"NO_FILE observed: {fname}", "WARN"))
            
            for msg, level in logs_to_emit: log_boot(msg, level)
            time.sleep(0.1)
            if not os.path.exists(filepath): yield None, ("EOF", "NO_FILE"); return

        try:
            with ExitStack() as stack:
                active_locks = []
                if lock_read:
                    locks = [AtomicFileLock(filepath, False)]
                    if head_path: locks.append(AtomicFileLock(head_path, False))
                    locks.sort(key=lambda x: os.path.basename(x.lock_file))
                    for l in locks: stack.enter_context(l); active_locks.append(l)
                
                last_heartbeat = time.time()
                with ResoneticCrypto.secure_open(filepath, "rb") as f:
                    while True:
                        if lock_read and time.time() - last_heartbeat > Config.HEARTBEAT_INTERVAL:
                            for l in active_locks: 
                                ok, msg = l.touch()
                                if not ok: yield None, ("WARN", msg)
                            last_heartbeat = time.time()
                        
                        skipped = 0
                        p_start = f.tell()
                        while True: 
                            p = f.tell(); ch = f.read(1)
                            if not ch: yield None, ("EOF", "OK"); return
                            if ch not in b"\n\r\t ": f.seek(p); break
                            
                            skipped += 1
                            if skipped > Config.MAX_SKIP_BYTES:
                                yield None, ("FATAL", f"EXCESSIVE_WHITESPACE@{p_start}")
                                return

                        header_start = f.tell(); lb = b""
                        while True:
                            c = f.read(1)
                            if not c: yield None, ("FATAL", f"TAIL_TRUNCATED@{header_start}" if not strict else f"TRUNCATED@{header_start}"); return
                            if c == b":": break
                            if not (b'0' <= c <= b'9') or len(lb) >= Config.MAX_HEADER_SIZE: yield None, ("FATAL", f"MALFORMED_HEADER@{header_start}"); return
                            lb += c
                        if not lb: yield None, ("FATAL", f"EMPTY_HEADER@{header_start}"); return
                        length = int(lb)
                        if length > Config.MAX_FRAME_SIZE: yield None, ("FATAL", f"SIZE_FAIL@{header_start}"); return
                        payload = f.read(length)
                        if len(payload) != length: yield None, ("FATAL", f"TAIL_TRUNCATED@{header_start}" if not strict else f"TRUNCATED@{header_start}"); return 
                        try: yield json.loads(payload.decode("utf-8")), None
                        except: yield None, ("FATAL", f"TAIL_JSON_FAIL@{header_start}" if not strict else f"JSON_FAIL@{header_start}"); return
        except ResoneticsFatal: raise
        except Exception as e:
            eno = getattr(e, "errno", None)
            yield None, ("FATAL", f"READ_FAIL:{errno.errorcode.get(eno, type(e).__name__) if eno else type(e).__name__}")

class ResoneticAgent:
    def __init__(self):
        self._physics_lock = threading.Lock()
        self.crypto = ResoneticCrypto(Config.KEY_FILE)
        self.global_step, self.hash_chain = 0, Config.GENESIS_HASH
        self.physics_step = 0
        self.ui_buffer = deque(maxlen=500); self.physics = PhysicsState(); self._buf_lock = threading.Lock()
        self.snapshot_ignored = False 
        self._secure_boot()

    def _clean_oath(self, filepath):
        _guard_no_state_lock_during_oath(f"_clean_oath({os.path.basename(filepath)})")
        try:
            rp = Config._canonicalize(filepath)
            with GLOBAL_STATE._oath_lock:
                existed = rp in GLOBAL_STATE.file_oaths
                if existed: del GLOBAL_STATE.file_oaths[rp]
            if existed and Config.DEV_MODE: log_boot(f"OATH_REVOKED: {os.path.basename(rp)}", "INFO")
        except Exception as e:
            if Config.DEV_MODE: log_boot(f"OATH_REVOKE_FAIL: {e}", "WARN")

    def _after_snapshot_purge_success(self, reason):
        if Config.DEV_MODE and GLOBAL_STATE._lock.held_by_me():
             raise ResoneticsFatal(f"ARTICLE_4_2_VIOLATION: _after_snapshot_purge_success called under StateLock ({reason})")

        self._clean_oath(Config.SNAP_FILE)
        with GLOBAL_STATE._lock:
            GLOBAL_STATE.snapshot_loaded = False
            GLOBAL_STATE.integrity_degraded = True
            GLOBAL_STATE.snapshot_fail_count = 0
        if Config.DEV_MODE: log_boot(f"SNAP_PURGED_STATE_RESET: {reason}", "INFO")

    def _secure_boot(self):
        while True:
            events = []
            fatal_error = None
            restart_scan = False
            heal_needed, heal_offset = False, 0
            purge_needed_reason = None
            
            with GLOBAL_STATE._lock: 
                if GLOBAL_STATE.quarantine: return
                GLOBAL_STATE.snapshot_loaded = False
                GLOBAL_STATE.head_stale = False # Reset flag on boot
            
            log_boot(f"Boot Sequence {Config.VERSION}..."); base_dir_safety_check()
            rem_locks, rem_bad = vacuum_stale_locks(Config.get_lock_dir())
            if rem_locks > 0 or rem_bad > 0: 
                log_boot(f"Auto-Hygienic: Cleaned {rem_locks} locks, {rem_bad} evidence.", "INFO")
            
            if not os.path.exists(Config.LEDGER_FILE):
                try: 
                    with self.crypto.secure_open(Config.LEDGER_FILE, "xb", exclusive=True) as f: f.write(b"")
                except: pass
            
            snap = None
            if not self.snapshot_ignored:
                try:
                    snap, purge_reason = self.crypto.load_snapshot(Config.SNAP_FILE)
                    if purge_reason:
                        success, msg = self.crypto.purge_snapshot(Config.SNAP_FILE, purge_reason)
                        if success: self._after_snapshot_purge_success(purge_reason)
                        log_boot(msg, "INFO" if success else "WARN")
                        snap = None
                except ResoneticsFatal as e: 
                    GLOBAL_STATE.trigger_quarantine(str(e)); log_boot(f"FATAL: Snapshot Oath Broken ({e})", "FATAL"); return
                except Exception as e:
                     with GLOBAL_STATE._lock: GLOBAL_STATE.integrity_degraded = True
                     log_boot(f"SNAPSHOT_READ_FAIL: {e}", "WARN")

            if snap:
                anchor_h = snap.get("anchor_hash", snap["chain_tip"])
                self.global_step = snap["global_step"]
                self.hash_chain = anchor_h
                self.physics_step = snap.get("physics_step", 0)
                m = snap.get("metrics", {})
                with self._physics_lock:
                    self.physics.current_s = float(m.get("entropy", 0.0))
                    self.physics.current_phi = float(m.get("stability", 1.0))
                exp_p, prev_g = self.hash_chain, self.global_step
                log_boot(f"Snapshot Loaded: Step {self.global_step}", "INFO")
                with GLOBAL_STATE._lock: GLOBAL_STATE.snapshot_loaded = True
                
                with GLOBAL_STATE._lock:
                    current_fp = GLOBAL_STATE.key_fingerprint
                if current_fp and snap.get("_key_fp") and snap["_key_fp"] != current_fp:
                    GLOBAL_STATE.trigger_quarantine(f"KEY_DRIFT: Snap {snap['_key_fp']} != Curr {current_fp}")
                    return
            else:
                exp_p, prev_g = Config.GENESIS_HASH, 0
            
            with self._buf_lock: self.ui_buffer.clear()
            
            with closing(self.crypto.stream_frames(Config.LEDGER_FILE, head_path=Config.HEAD_FILE, lock_read=True, strict=False)) as frames:
                try:
                    for entry, err_data in frames:
                        err_type, err_msg = err_data if err_data else (None, None)

                        if err_type == "EOF": break
                        if err_type == "WARN":
                                events.append((err_msg, "WARN")); continue
                        
                        if err_type == "FATAL" and err_msg and ("TAIL_TRUNCATED" in err_msg or "TAIL_JSON_FAIL" in err_msg):
                                try: 
                                    heal_offset = int(err_msg.split("@")[1])
                                    heal_needed = True
                                    events.append((f"Healable Wound: {err_msg}", "WARN"))
                                    break 
                                except: pass 

                        if err_type == "FATAL":
                            events.append((f"BOOT CRITICAL: {err_msg}", "FATAL"))
                            fatal_error = str(err_msg)
                            break
                        
                        e_step = entry["global_step"]
                        if snap and e_step == snap["global_step"]:
                            anchor_h = snap.get("anchor_hash", snap["chain_tip"])
                            if entry["hash"].lower() != anchor_h:
                                events.append((f"MISMATCH G{e_step}: Snap vs Ledger.", "WARN"))
                                purge_needed_reason = "SNAP_VS_LEDGER_MISMATCH"
                                restart_scan = True; break 
                        
                        if snap and e_step <= snap["global_step"]: continue
                        ok, msg = self.crypto.verify_entry(entry)
                        if not ok or e_step != prev_g + 1 or entry["prev_hash"].lower() != exp_p:
                            events.append((f"Chain Break at G{e_step}: {msg}", "FATAL"))
                            fatal_error = f"Chain Break: {msg}"
                            break
                        
                        self.global_step, self.hash_chain = e_step, entry["hash"].lower()
                        self.physics_step = int(entry.get("physics_step", self.physics_step))
                        exp_p, prev_g = self.hash_chain, self.global_step
                        with self._buf_lock: self.ui_buffer.append(entry)
                except ResoneticsFatal as e:
                    fatal_error = str(e)
                    events.append((f"FATAL: {e}", "FATAL"))
            
            for msg, level in events: log_boot(msg, level)
            
            if fatal_error:
                GLOBAL_STATE.trigger_quarantine(fatal_error)
                return

            if purge_needed_reason:
                success, msg = self.crypto.purge_snapshot(Config.SNAP_FILE, purge_needed_reason)
                if success:
                    self._after_snapshot_purge_success(purge_needed_reason)
                    log_boot(f"Deferred Purge: {msg}", "INFO")
                else:
                    log_boot(f"Deferred Purge Failed: {msg}", "WARN")
                    with GLOBAL_STATE._lock:
                        GLOBAL_STATE.integrity_degraded = True
                        GLOBAL_STATE.snapshot_loaded = False
                self.snapshot_ignored = True
                continue

            if restart_scan:
                with GLOBAL_STATE._lock:
                     GLOBAL_STATE.integrity_degraded = True
                     if snap: GLOBAL_STATE.snapshot_loaded = False
                self.snapshot_ignored = True
                continue 
            
            if heal_needed and heal_offset > 0:
                log_boot(f"Executing Surgical Truncation at {heal_offset}...", "INFO")
                self.crypto.truncate_ledger_at(Config.LEDGER_FILE, heal_offset)
                log_boot("HEAL_APPLIED. Rebooting...", "WARN")
                with GLOBAL_STATE._lock: GLOBAL_STATE.tail_heal_observed = True; GLOBAL_STATE.integrity_degraded = True
                continue
            
            if self.ui_buffer:
                m = (self.ui_buffer[-1].get("metrics", {}) or {})
                with self._physics_lock:
                    self.physics.current_s = float(m.get("entropy", 0.0))
                    self.physics.current_phi = float(m.get("stability", 1.0))
            
            need_rebuild = False
            if os.path.exists(Config.HEAD_FILE):
                try:
                    with self.crypto.secure_open(Config.HEAD_FILE, "rb") as f:
                        parts = f.read().decode().strip().split()
                    if len(parts) == 2 and parts[0].isdigit():
                        if int(parts[0]) != self.global_step or parts[1].lower() != self.hash_chain: need_rebuild = True
                    else: need_rebuild = True
                except ResoneticsFatal as e: GLOBAL_STATE.trigger_quarantine(str(e)); log_boot(f"FATAL: {e}", "FATAL"); return
                except Exception: need_rebuild = True
            else: need_rebuild = True

            if need_rebuild:
                try:
                    self.crypto.force_rebuild_head(Config.HEAD_FILE, self.global_step, self.hash_chain)
                    with GLOBAL_STATE._lock: GLOBAL_STATE.head_rebuild_observed = True
                    log_boot("HEAD_REBUILT to match Ledger.", "INFO")
                except ResoneticsFatal as e: GLOBAL_STATE.trigger_quarantine(str(e)); log_boot(f"FATAL: {e}", "FATAL"); return
                except Exception as e: GLOBAL_STATE.trigger_quarantine(f"HEAD_REBUILD_UNKNOWN: {e}"); return

            log_boot(f"Kernel Online. Step {self.global_step}", "INFO"); break

    def get_snapshot_state(self):
        with self._physics_lock:
            s = self.physics.current_s
            phi = self.physics.current_phi
            
        return {
            "global_step": self.global_step,
            "physics_step": self.physics_step,
            "metrics": {
                "entropy": float(s),
                "stability": float(phi)
            }
        }

    def create_snapshot(self):
        with GLOBAL_STATE._lock:
            if GLOBAL_STATE.quarantine: return False
        try:
            state = self.get_snapshot_state()
            if self.crypto.save_snapshot(Config.SNAP_FILE, state, self.hash_chain):
                _, purge_reason = self.crypto.load_snapshot(Config.SNAP_FILE)
                if purge_reason:
                    success, msg = self.crypto.purge_snapshot(Config.SNAP_FILE, purge_reason)
                    if success: self._after_snapshot_purge_success(purge_reason)
                    else: 
                        with GLOBAL_STATE._lock: GLOBAL_STATE.snapshot_loaded = False; GLOBAL_STATE.integrity_degraded = True
                    log_boot(f"SNAP_POSTCHECK_FAIL: {msg}", "WARN")
                else:
                     with GLOBAL_STATE._lock: GLOBAL_STATE.snapshot_loaded = True
                     self.snapshot_ignored = False
                     return True
            return False
        except ResoneticsFatal as e: GLOBAL_STATE.trigger_quarantine(str(e)); log_boot(f"FATAL: {e}", "FATAL"); return False
        except Exception as e:
            with GLOBAL_STATE._lock: GLOBAL_STATE.integrity_degraded = True
            log_boot(f"SNAPSHOT FAIL: {e}", "WARN"); return False

    def verify_ledger_now(self, deep=False):
        snap = None
        logs_to_emit = []
        result = None
        
        if not deep and not self.snapshot_ignored:
            try:
                snap, purge_reason = self.crypto.load_snapshot(Config.SNAP_FILE)
                if purge_reason:
                     with GLOBAL_STATE._lock: GLOBAL_STATE.integrity_degraded = True
                     logs_to_emit.append((f"SNAP_FAIL:{purge_reason}", "WARN"))
                     result = (False, (0, 0), f"SNAP_FAIL:{purge_reason}", Config.GENESIS_HASH)

            except ResoneticsFatal as e: raise 
            except Exception as e:
                 with GLOBAL_STATE._lock: GLOBAL_STATE.integrity_degraded = True

        if result:
            for m, l in logs_to_emit: log_boot(m, l)
            return result

        start_g = snap["global_step"] if snap else 0
        anchor = None
        if snap: anchor = snap.get("anchor_hash") or snap.get("chain_tip")
        exp_prev_hash = anchor if snap else Config.GENESIS_HASH
        prev_g = start_g
        total_scanned = 0
        total_verified = 0
        anchored = False if snap else True
        
        degrade_reason = None
        fatal_reason = None
        
        with closing(self.crypto.stream_frames(Config.LEDGER_FILE, head_path=Config.HEAD_FILE, lock_read=True, strict=False)) as frames:
            try:
                for entry, err_data in frames:
                    err_type, err_msg = err_data if err_data else (None, None)

                    if err_type == "EOF": break
                    if err_type == "WARN": logs_to_emit.append((err_msg, "WARN")); continue
                    
                    if err_type == "FATAL":
                         if err_msg and ("TAIL_TRUNCATED" in err_msg or "TAIL_JSON_FAIL" in err_msg):
                             result = (True, (total_verified, total_scanned), f"HEALABLE:{err_msg}", exp_prev_hash)
                         else:
                             result = (False, (total_verified, total_scanned), f"ERR:{err_msg}", exp_prev_hash)
                         break

                    total_scanned += 1
                    e_step = entry["global_step"]
                    
                    if snap and e_step == start_g:
                        if entry["hash"].lower() != anchor:
                            result = (False, (total_verified, total_scanned), "SNAP_ANCHOR_MISMATCH", exp_prev_hash)
                            break
                        anchored = True; exp_prev_hash, prev_g = entry["hash"].lower(), e_step; continue
                    
                    if e_step <= start_g: continue
                    
                    ok, msg = self.crypto.verify_entry(entry)
                    if not ok: 
                        result = (False, (total_verified, total_scanned), f"SIG_FAIL:{msg}", exp_prev_hash)
                        break
                    if e_step != prev_g + 1 or entry["prev_hash"].lower() != exp_prev_hash:
                        result = (False, (total_verified, total_scanned), f"CHAIN_BREAK@G{e_step}", exp_prev_hash)
                        break
                    
                    total_verified += 1
                    exp_prev_hash, prev_g = entry["hash"].lower(), e_step
            except ResoneticsFatal as e:
                raise

        # ✅ Fix 2: Living Anchor (Check if anchor was actually found)
        if snap and not anchored and result is None:
             result = (False, (total_verified, total_scanned), "SNAP_ANCHOR_NOT_FOUND", exp_prev_hash)

        if degrade_reason:
            with GLOBAL_STATE._lock: GLOBAL_STATE.integrity_degraded = True
            log_boot(degrade_reason, "WARN")
            
        for m, l in logs_to_emit: log_boot(m, l)
        
        return result if result else (False, (0,0), "UNKNOWN_STATE", Config.GENESIS_HASH)

    def tick(self):
        with GLOBAL_STATE._lock:
            if GLOBAL_STATE.quarantine:
                GLOBAL_STATE.running = False; GLOBAL_STATE.snapshot_loaded = False; return
            if not GLOBAL_STATE.running: return
            
            now = time.time()
            if now - GLOBAL_STATE.last_commit_time < Config.COMMIT_INTERVAL:
                 GLOBAL_STATE.last_pulse = now 
                 return

            if now - GLOBAL_STATE.last_pulse < Config.PULSE_INTERVAL: return
            GLOBAL_STATE.last_pulse = now
            vacuum = GLOBAL_STATE.vacuum_observed; tail_heal = GLOBAL_STATE.tail_heal_observed; head_rebuild = GLOBAL_STATE.head_rebuild_observed; degraded = GLOBAL_STATE.integrity_degraded
            GLOBAL_STATE.vacuum_observed = False; GLOBAL_STATE.tail_heal_observed = False; GLOBAL_STATE.head_rebuild_observed = False

        try:
            lock_dir = Config.get_lock_dir()
            lock_res = len([f for f in os.listdir(lock_dir) if f.endswith(".lock")]) if os.path.isdir(lock_dir) else 0
        except: lock_res = 0

        with self._physics_lock:
            dS = PhysicsConfig.S_BASE + (lock_res * PhysicsConfig.S_FRICTION_K)
            if vacuum: self.physics.current_s *= (1.0 - PhysicsConfig.S_VACUUM_REDUCTION)
            if tail_heal: dS += PhysicsConfig.S_HEAL_COST
            if head_rebuild: dS += PhysicsConfig.S_REBUILD_COST
            self.physics.current_s += dS
            self.physics.current_s *= (1.0 - PhysicsConfig.S_COOLING_RATE)
            penalty = (lock_res * PhysicsConfig.PENALTY_LOCKS_K) + (PhysicsConfig.PENALTY_DEGRADED if degraded else 0.0)
            self.physics.current_phi = max(0.0, min(1.0, math.exp(-PhysicsConfig.PHI_A * penalty)))
            metrics = {"entropy": round(self.physics.current_s, 6), "stability": round(self.physics.current_phi, 6), "heat_level": self.physics.heat_level()}

        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            meta = {}; meta.setdefault("_platform", {})["integrity"] = "DEGRADED" if degraded else "OK"
            entry_builder = lambda ns, ph: {"prev_hash": ph, "global_step": ns, "physics_step": self.physics_step + 1, "episode": 0, "episode_step": 0, "type": "PHYSICS_TICK", "status": "OK", "msg": "Inherited Oath pulse", "metrics": metrics, "meta": meta, "time": ts}
            entry = self.crypto.commit_transaction(Config.LEDGER_FILE, Config.HEAD_FILE, entry_builder, expected=self.hash_chain)
            self.global_step, self.hash_chain = entry["global_step"], entry["hash"].lower()
            self.physics_step = int(entry["physics_step"])
            with self._buf_lock: self.ui_buffer.append(entry)
            
            with GLOBAL_STATE._lock: GLOBAL_STATE.last_commit_time = time.time()

        except ResoneticsHeadStale as e:
            log_boot(f"HEAD_STALE_DETECTED: {e}. Attempting Active Recovery...", "WARN")
            try:
                self.crypto.force_rebuild_head(Config.HEAD_FILE, e.payload["global_step"], e.payload["hash"])
                self.global_step = e.payload["global_step"]
                self.hash_chain = e.payload["hash"].lower()
                self.physics_step = int(e.payload["physics_step"])
                with self._buf_lock: self.ui_buffer.append(e.payload)
                with GLOBAL_STATE._lock: GLOBAL_STATE.head_rebuild_observed = True
                log_boot("Active Recovery Successful: Memory synchronized with Ledger.", "INFO")
            except Exception as heal_err:
                GLOBAL_STATE.trigger_quarantine(f"HEAD_STALE_HEAL_FAIL: {heal_err}")
            return 
            
        except ResoneticsFatal as e: GLOBAL_STATE.trigger_quarantine(str(e)); log_boot(f"FATAL: {e}", "FATAL")
        except (OSError, Exception) as e:
            log_boot(f"COMMIT FAIL: {e}", "WARN")
            with GLOBAL_STATE._lock:
                GLOBAL_STATE.integrity_degraded = True

# ✅ Fix 3: Decoupled Core (Using session_state only)
if "RES_AGENT" not in st.session_state:
    st.session_state.RES_AGENT = ResoneticAgent()

agent = st.session_state.RES_AGENT

# ✅ UI Ritual: Visualized Inheritance
st.sidebar.title(f"💎 Resonetics {Config.VERSION}")
if GLOBAL_STATE.quarantine: 
    st.sidebar.error(f"🚨 QUARANTINE: {GLOBAL_STATE.last_error_code}")
elif GLOBAL_STATE.integrity_degraded: 
    warn_msg = GLOBAL_STATE.last_warn_code
    if GLOBAL_STATE.first_warn_code != "None" and GLOBAL_STATE.first_warn_code != GLOBAL_STATE.last_warn_code: warn_msg = f"{GLOBAL_STATE.first_warn_code} ... {GLOBAL_STATE.last_warn_code}"
    if warn_msg == "None": warn_msg = "Unknown Integrity Issue"
    st.sidebar.warning(f"🛡️ DEGRADED: {warn_msg}")
    if GLOBAL_STATE.head_stale: st.sidebar.info("🔧 Head Stale: Rebuild Pending")

with st.sidebar.expander("📜 System Logs", expanded=False):
    st.text_area("Recent Logs", "\n".join(list(GLOBAL_STATE.boot_logs)[-20:]), height=200)

with st.sidebar.expander("🛠️ Forensic & Snapshot"):
    if st.button("Fast Audit"):
        _warn_if_io_depth("Fast Audit Button")
        try:
            ok, (ver, scan), r, _ = agent.verify_ledger_now(deep=False)
            with GLOBAL_STATE._lock: GLOBAL_STATE.last_audit_time, GLOBAL_STATE.last_audit_result, GLOBAL_STATE.last_audit_frames = time.time(), f"FAST:{r}", f"{ver}/{scan}"
            st.rerun()
        except ResoneticsFatal as e: GLOBAL_STATE.trigger_quarantine(str(e)); log_boot(f"FATAL: {e}", "FATAL"); st.rerun()

    if st.button("Deep Audit"):
        _warn_if_io_depth("Deep Audit Button")
        try:
            ok, (ver, scan), r, _ = agent.verify_ledger_now(deep=True)
            with GLOBAL_STATE._lock: GLOBAL_STATE.last_audit_time, GLOBAL_STATE.last_audit_result, GLOBAL_STATE.last_audit_frames = time.time(), f"DEEP:{r}", f"{ver}/{scan}"
            st.rerun()
        except ResoneticsFatal as e: GLOBAL_STATE.trigger_quarantine(str(e)); log_boot(f"FATAL: {e}", "FATAL"); st.rerun()

    st.write(f"**Result:** {GLOBAL_STATE.last_audit_result}")
    st.caption(f"Verified/Scanned: {GLOBAL_STATE.last_audit_frames}")
    
    if st.button("Create Snapshot"):
        if agent.create_snapshot(): st.success(f"Snapshot Crystallized @ G{agent.global_step}")
        else: st.error("Snapshot Failed")
    
    if st.button("Vacuum Locks"):
        rem_locks, rem_bad = vacuum_stale_locks(Config.get_lock_dir())
        with GLOBAL_STATE._lock: GLOBAL_STATE.vacuum_observed = True
        log_boot(f"Vacuum: {rem_locks} locks, {rem_bad} evidence removed.", "INFO")
        st.rerun()

if st.sidebar.button("Start/Stop Pulse", disabled=GLOBAL_STATE.quarantine):
    with GLOBAL_STATE._lock: GLOBAL_STATE.running = not GLOBAL_STATE.running
    st.rerun()

@st.fragment(run_every=Config.PULSE_INTERVAL)
def pulse_monitor():
    with GLOBAL_STATE._lock:
        is_quarantined = GLOBAL_STATE.quarantine; is_running = GLOBAL_STATE.running; is_snapshot = GLOBAL_STATE.snapshot_loaded
    if is_quarantined: st.error("SYSTEM QUARANTINED"); st.markdown(f"### 🔥 Heat: CRITICAL | Φ=0.0"); return
    if is_running: agent.tick()
    with agent._physics_lock: heat = agent.physics.heat_level(); phi = round(agent.physics.current_phi, 4)
    st.markdown(f"### 🔥 Heat: {heat} | Φ={phi}")
    st.write(f"🌌 **Step:** {agent.global_step} | **Tip:** {agent.hash_chain[:12]}")
    if is_snapshot: st.caption("✨ Snapshot Active")

pulse_monitor()

