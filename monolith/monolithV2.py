# ==============================================================================
# Project: Resonetics
# File: monolithV2 (patched)
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
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timezone
from contextlib import contextmanager, ExitStack
from typing import Optional

import streamlit as st

# [Article 0] Constitution: Executable, Unified Lock Order, Unified Identity, No IO-State Lock, Resilient Deep Scan.
# [Article 0.1] OATH Policy: OATH keys derive from physical IDs (POSIX) or verified handle paths (Win).
# [Article 0.2] UI Rule: No UI-State Lock. Interact with UI outside of STATE lock.
# [Article 0.3] Intent Policy: OATH monitoring is mandatory for Core Files (Dual Promotion) and explicit for others.
# [Article 0.4] Governance Policy: All verification must follow a structured PolicySpec.

@dataclass(frozen=True)
class PolicySpec:
    name: str
    strict_stream: bool
    allow_extra: bool
    lock: bool
    budget: Optional[float]
    nb: bool

    @classmethod
    def from_dict(cls, name: str, d: dict):
        return cls(
            name=name,
            strict_stream=bool(d.get("strict_stream", False)),
            allow_extra=bool(d.get("allow_extra", False)),
            lock=bool(d.get("lock", False)),
            budget=d.get("budget", None),
            nb=bool(d.get("nb", False)),
        )

class Config:
    APP_VERSION, DATA_VERSION = "v168.0", "v2_core"
    GENESIS_HASH = "0" * 64
    try: _RAW = os.path.dirname(os.path.abspath(__file__))
    except NameError: _RAW = os.getcwd()
    BASE_DIR_PATH = Path(_RAW).resolve(); BASE_DIR = str(BASE_DIR_PATH)

    FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF

    @classmethod
    def _canonicalize(cls, p):
        return str(Path(p).absolute())

    @classmethod
    def _norm_key(cls, p, already_canon=False):
        c = p if already_canon else cls._canonicalize(p)
        return c.casefold() if os.name == 'nt' else c

    @classmethod
    def within_base(cls, p):
        try:
            t = Path(p).absolute()
            b = cls.BASE_DIR_PATH
            if os.name == 'nt':
                t_s, b_s = str(t).rstrip("\\/").casefold(), str(b).rstrip("\\/").casefold()
                return (t_s == b_s) or t_s.startswith(b_s + "\\")
            if sys.version_info >= (3,9): return t.is_relative_to(b)
            return t.parts[:len(b.parts)] == b.parts
        except: return False

    @classmethod
    def get_lock_dir(cls): return str(cls.BASE_DIR_PATH / ".locks")
    @classmethod
    def get_temp_dir(cls): return str(cls.BASE_DIR_PATH / ".tmp")

    MAX_HEADER_SIZE, MAX_FRAME_SIZE = 20, 256 * 1024
    MAX_SKIP_BYTES, MAX_SKIPS_ALLOWED = 4096, 100
    PULSE_INTERVAL, COMMIT_INTERVAL = 1.0, 10.0
    DEEP_SCAN_BUDGET, FAST_SCAN_BUDGET = 2.0, 0.3
    BOOT_SCAN_BUDGET, MAX_WARN_KEYS = 30.0, 1024
    _UNSET = -1.0

    HAS_NOFOLLOW = hasattr(os, "O_NOFOLLOW")
    DEV_MODE = os.environ.get("RESONETICS_DEV_MODE")=="1"

    SIG_FIELDS_V2 = ["prev_hash", "global_step", "physics_step", "episode", "episode_step", "type", "status", "msg", "metrics", "meta", "time", "ver"]
    SNAP_FIELDS_V2_1 = ["global_step", "chain_tip", "anchor_hash", "physics_step", "metrics", "timestamp", "ver", "_key_fp", "snap_ver", "app_ver", "ledger_offset"]
    SNAP_FIELDS_V2_0 = ["global_step", "chain_tip", "anchor_hash", "physics_step", "metrics", "timestamp", "ver", "_key_fp", "snap_ver", "app_ver"]

    POLICIES = {
        "AUDIT_FAST": {"strict_stream": False, "allow_extra": True, "lock": False, "budget": FAST_SCAN_BUDGET, "nb": False},
        "AUDIT_DEEP": {"strict_stream": False, "allow_extra": True, "lock": True, "budget": DEEP_SCAN_BUDGET, "nb": False},
        "BOOT":       {"strict_stream": True, "allow_extra": True, "lock": True, "budget": BOOT_SCAN_BUDGET, "nb": False}
    }

def _gp(p): return Path(Config._canonicalize(p))
_B = Config.BASE_DIR_PATH / f"resonetics_{Config.DATA_VERSION}.ledger"
Config.LEDGER_FILE_P, Config.HEAD_FILE_P = _gp(str(_B)), _gp(str(_B) + ".head")
Config.KEY_FILE_P, Config.SNAP_FILE_P = _gp(str(_B) + ".key"), _gp(str(_B) + ".snap")
Config.CORE_SET = frozenset({Config.KEY_FILE_P, Config.HEAD_FILE_P, Config.LEDGER_FILE_P, Config.SNAP_FILE_P})

IS_POSIX, IS_WINLOCK = os.name=="posix", os.name=="nt"

if IS_WINLOCK:
    import msvcrt
    from ctypes import wintypes
    _K32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _GetFinalPathNameByHandleW = _K32.GetFinalPathNameByHandleW
    _GetFinalPathNameByHandleW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    _GetFinalPathNameByHandleW.restype  = wintypes.DWORD
    _GetFileAttributesW = _K32.GetFileAttributesW
    _GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
    _GetFileAttributesW.restype = wintypes.DWORD

    def _win_final_path_from_fd(fd: int) -> str:
        h = msvcrt.get_osfhandle(fd)
        if h in (0, -1): raise OSError("BAD_HANDLE")
        n = _GetFinalPathNameByHandleW(wintypes.HANDLE(h), None, 0, 0)
        if n == 0: raise OSError(ctypes.get_last_error())
        buf = ctypes.create_unicode_buffer(n + 2)
        if _GetFinalPathNameByHandleW(wintypes.HANDLE(h), buf, len(buf), 0) == 0: raise OSError(ctypes.get_last_error())
        p = buf.value
        if p.startswith("\\\\?\\UNC\\"): return "\\" + p[7:]
        if p.startswith("\\\\?\\"): return p[4:]
        return p

def _gen_cid(fd, st_obj):
    if IS_POSIX: return f"posix:{st_obj.st_dev}:{st_obj.st_ino}"
    try:
        curr = os.lseek(fd, 0, 1); os.lseek(fd, 0, 0)
        h = hashlib.sha256(os.read(fd, 4096)).hexdigest()[:16]
        os.lseek(fd, curr, 0)
        return f"win:{st_obj.st_size}:{h}"
    except: return f"win:{st_obj.st_size}:err"

class ResoneticsFatal(RuntimeError): pass
class ResoneticsHeadStale(ResoneticsFatal):
    def __init__(self, c, p, o): super().__init__(c); self.code, self.payload, self.new_offset = c, p, o

@dataclass
class PhysicsState:
    current_s: float = 0.0; current_phi: float = 1.0
    def heat_level(self): return "CRITICAL" if self.current_s >= 2.0 else "HIGH" if self.current_s >= 1.0 else "NORMAL"

_IO_TRACKER = threading.local()
@contextmanager
def io_context():
    setattr(_IO_TRACKER, "d", getattr(_IO_TRACKER, "d", 0)+1)
    try: yield
    finally: setattr(_IO_TRACKER, "d", max(0, getattr(_IO_TRACKER, "d", 0)-1))

class TrackedRLock:
    def __init__(self, n, io=False): self._l, self.n, self.io, self._t = threading.RLock(), n, io, threading.local()
    def held_by_me(self): return getattr(self._t, "c", 0) > 0
    def __enter__(self):
        if getattr(_IO_TRACKER, "d", 0)>0 and not self.io: raise ResoneticsFatal(f"VIOLATION: Lock {self.n} in IO")
        self._l.acquire(); self._t.c = getattr(self._t, "c", 0)+1; return self
    def __exit__(self, *a):
        try: self._t.c = max(0, getattr(self._t, "c", 0)-1)
        finally: self._l.release()

@dataclass
class ProcessStateData:
    quarantine: bool=False; running: bool=False; boot_completed: bool=False; root_cause: str="OK"
    snapshot_loaded: bool=False; file_oaths: dict=field(default_factory=dict)
    boot_logs: deque=field(default_factory=lambda: deque(maxlen=300), repr=False)
    _lock: TrackedRLock=field(default_factory=lambda: TrackedRLock("STATE"), repr=False)
    _oath_lock: TrackedRLock=field(default_factory=lambda: TrackedRLock("OATH", True), repr=False)
    last_audit_time, last_audit_result, last_audit_frames = 0.0, "-", "0/0"
    audit_resume_off, audit_last_verified_step = 0, 0
    audit_next_prev_hash: str = Config.GENESIS_HASH
    key_fingerprint, ledger_offset, last_commit_time = None, 0, 0.0
    def trigger_quarantine(self, r):
        if getattr(_IO_TRACKER, "d", 0) > 0:
            if not hasattr(_IO_TRACKER, "q"): _IO_TRACKER.q = deque(maxlen=8)
            _IO_TRACKER.q.append(r); return
        with self._lock:
            if self.quarantine: return
            self.quarantine, self.running = True, False
            if self.root_cause == "OK": self.root_cause = r[:40]

if "GLOBAL_STATE" not in st.session_state: st.session_state.GLOBAL_STATE = ProcessStateData()
GLOBAL_STATE = st.session_state.GLOBAL_STATE
_WARN_LOCK = threading.Lock()

# ==============================================================================
# PATCH 1: warn_once LRU (avoid "clear everything" policy)
# - Keeps O(1) membership via set + deque for eviction
# - Still protected by _WARN_LOCK
# ==============================================================================
def _warn_cache_init():
    if "_WR_SET" not in st.session_state: st.session_state._WR_SET = set()
    if "_WR_Q" not in st.session_state: st.session_state._WR_Q = deque(maxlen=Config.MAX_WARN_KEYS)

def warn_once(key, msg):
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    with _WARN_LOCK:
        _warn_cache_init()
        s = st.session_state._WR_SET
        q = st.session_state._WR_Q
        if h in s: 
            return
        # Evict oldest if deque at capacity (maxlen handles pop, but we must sync set)
        if len(q) >= q.maxlen:
            old = q.popleft()
            s.discard(old)
        q.append(h)
        s.add(h)
    log_boot(msg, "WARN")

def flush_pending_logs():
    if getattr(_IO_TRACKER, "d", 0) > 0 or getattr(_IO_TRACKER, "f", False): return 0
    setattr(_IO_TRACKER, "f", True)
    try:
        q_deque = getattr(_IO_TRACKER, "q", None)
        if q_deque:
            with GLOBAL_STATE._lock:
                while q_deque:
                    r = q_deque.popleft()
                    if not GLOBAL_STATE.quarantine:
                        GLOBAL_STATE.quarantine, GLOBAL_STATE.running = True, False
                        if GLOBAL_STATE.root_cause == "OK": GLOBAL_STATE.root_cause = r[:40]
                    ts = datetime.now().strftime('%H:%M:%S'); GLOBAL_STATE.boot_logs.append(f"[FATAL] {ts} 💀 QUARANTINE: {r}")
            return -1
        buf = getattr(_IO_TRACKER, "p", []); n = len(buf)
        if not buf: return 0
        with GLOBAL_STATE._lock:
            for e, l, r in buf:
                GLOBAL_STATE.boot_logs.append(e)
                if l in ("ERROR","FATAL") and GLOBAL_STATE.root_cause=="OK": GLOBAL_STATE.root_cause = r[:40]
        buf.clear(); return n
    finally: setattr(_IO_TRACKER, "f", False)

def log_boot(m, l="INFO"):
    e = f"[{l}] {datetime.now().strftime('%H:%M:%S')} {m}"
    if getattr(_IO_TRACKER, "d", 0)>0:
        if not hasattr(_IO_TRACKER, "p"): _IO_TRACKER.p = []
        _IO_TRACKER.p.append((e, l, m)); return
    try:
        with GLOBAL_STATE._lock:
            GLOBAL_STATE.boot_logs.append(e)
            if l in ("ERROR","FATAL") and GLOBAL_STATE.root_cause=="OK": GLOBAL_STATE.root_cause = m[:40]
    except: pass

def _guard_no_state_lock(r=""):
    gs = globals().get("GLOBAL_STATE", None)
    if gs and gs._lock.held_by_me(): raise ResoneticsFatal(f"VIOLATION: Lock STATE during IO ({r})")

# ---- Core Helpers ----
# ==============================================================================
# PATCH 2: _secure_dir_open fd leak hardening
# - Ensures next_fd is closed if any exception occurs after opening it
# - Keeps behavior identical otherwise
# ==============================================================================
def _secure_dir_open(path):
    if not IS_POSIX: return None
    if not Config.within_base(path): raise ResoneticsFatal("WALK_OUTSIDE_BASE")
    nf, cx = getattr(os, "O_NOFOLLOW", 0), getattr(os, "O_CLOEXEC", 0)
    if not Config.HAS_NOFOLLOW and not Config.DEV_MODE: raise ResoneticsFatal("OS_INSECURE_POLICY")

    cur_fd = os.open(str(Config.BASE_DIR_PATH), os.O_RDONLY | os.O_DIRECTORY | cx)
    rel_path = os.path.relpath(path, str(Config.BASE_DIR_PATH))
    if rel_path == ".": return cur_fd

    try:
        for part in Path(rel_path).parts:
            if part in ("", ".", ".."):
                raise ResoneticsFatal(f"PATH_COMPONENT_VIOLATION:{part}")

            next_fd = None
            try:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | nf | cx, dir_fd=cur_fd)
                f_st = os.fstat(next_fd)
                l_st = os.stat(part, dir_fd=cur_fd, follow_symlinks=False)
                if (f_st.st_dev, f_st.st_ino) != (l_st.st_dev, l_st.st_ino):
                    raise ResoneticsFatal(f"CHAIN_MIRROR_DETECTION:{part}")

                os.close(cur_fd)
                cur_fd = next_fd
                next_fd = None
            finally:
                # If we opened next_fd but failed before promoting it to cur_fd, close it.
                if next_fd is not None:
                    try: os.close(next_fd)
                    except: pass

        return cur_fd
    except Exception as e:
        if cur_fd != -1:
            try: os.close(cur_fd)
            except: pass
        raise ResoneticsFatal(f"CHAIN_WALK_FAIL:{e}")

def _audit_handle_identity(fd, display_name, logical_path=None, force_oath=False):
    """[Cerberus Vigilance] Fix 3 & 4🚨: Identity Auditor with Dual Promotion and Safe Attr Check."""
    if IS_WINLOCK:
        final_p = _win_final_path_from_fd(fd)
        final_canon = Config._canonicalize(final_p)
        target_key = Config._norm_key(final_canon, already_canon=True)

        # Fix 3🚨: Robust attribute check on final path
        attr = _GetFileAttributesW(final_canon)
        if attr == Config.INVALID_FILE_ATTRIBUTES:
            # Fallback to handle-raw path if canon failed
            attr = _GetFileAttributesW(final_p)

        if attr == Config.INVALID_FILE_ATTRIBUTES or (attr & Config.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ResoneticsFatal(f"WIN_HANDLE_DIRTY:{display_name}")

        if not Config.within_base(final_canon): raise ResoneticsFatal(f"BOUNDARY_ESC:{display_name}")
    else:
        st_obj = os.fstat(fd)
        target_key = f"posix:{st_obj.st_dev}:{st_obj.st_ino}"

    core_keys = st.session_state.get('GLOBAL_CORE_KEYS', frozenset())

    # Fix 4🚨: Dual-Map Promotion (Check both physical ID and normalized logical path)
    logical_key = Config._norm_key(logical_path) if logical_path else None
    effective_force = force_oath or (target_key in core_keys) or (logical_key in core_keys)

    if effective_force:
        cid = _gen_cid(fd, os.fstat(fd))
        with GLOBAL_STATE._oath_lock:
            if GLOBAL_STATE.file_oaths.get(target_key, cid) != cid: raise ResoneticsFatal(f"ID_SHIFT:{display_name}")
            GLOBAL_STATE.file_oaths[target_key] = cid
    return target_key

def base_dir_safety_check():
    """[Security Core] Sealed Boundary Boot Audit."""
    _guard_no_state_lock("safety_check")
    curr_base_canon = str(Config.BASE_DIR_PATH)
    if "SESSION_BASE_DIR" not in st.session_state: st.session_state.SESSION_BASE_DIR = curr_base_canon
    elif st.session_state.SESSION_BASE_DIR != curr_base_canon: GLOBAL_STATE.trigger_quarantine("DIR_IDENTITY_DRIFT"); return

    l, all_core_keys = [], set()
    try:
        with io_context():
            for cp in Config.CORE_SET:
                abs_key = Config._norm_key(str(cp), already_canon=True); all_core_keys.add(abs_key)
                if not os.path.exists(cp): continue
                try:
                    if IS_WINLOCK:
                        with open(cp, "rb") as f:
                            _audit_handle_identity(f.fileno(), str(cp), logical_path=str(cp), force_oath=True)
                            final_canon = Config._canonicalize(_win_final_path_from_fd(f.fileno()))
                            h_key = Config._norm_key(final_canon, already_canon=True); all_core_keys.add(h_key)
                            cid = _gen_cid(f.fileno(), os.fstat(f.fileno()))
                            with GLOBAL_STATE._oath_lock: GLOBAL_STATE.file_oaths[h_key] = cid
                    else:
                        dfd = _secure_dir_open(str(cp.parent))
                        try:
                            fname = cp.name
                            fd = os.open(fname, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=dfd)
                            try:
                                _audit_handle_identity(fd, str(cp), logical_path=str(cp), force_oath=True)
                                st_f, st_l = os.fstat(fd), os.stat(fname, dir_fd=dfd, follow_symlinks=False)
                                if (st_f.st_dev, st_f.st_ino) != (st_l.st_dev, st_l.st_ino): l.append((f"MIRROR:{cp.name}", "FATAL"))
                                h_key = f"posix:{st_f.st_dev}:{st_f.st_ino}"; all_core_keys.add(h_key)
                                cid = _gen_cid(fd, st_f)
                                with GLOBAL_STATE._oath_lock: GLOBAL_STATE.file_oaths[h_key] = cid
                            finally: os.close(fd)
                        finally: os.close(dfd)
                except OSError: l.append((f"OPEN_FAIL:{cp.name}", "FATAL"))
            st.session_state.GLOBAL_CORE_KEYS = frozenset(all_core_keys)
    except Exception as e: l.append((f"SAFETY_EXC:{e}", "WARN"))
    for m, lv in l: log_boot(m, lv)
    if any(v=="FATAL" for _,v in l): GLOBAL_STATE.trigger_quarantine("SECURITY_VIOLATION")
    else:
        with GLOBAL_STATE._lock:
            if not GLOBAL_STATE.boot_completed and not GLOBAL_STATE.quarantine:
                GLOBAL_STATE.running, GLOBAL_STATE.boot_completed = True, True

class AtomicFileLock:
    def __init__(self, t, ex=True, to=5.0):
        self.t_abs = Config._canonicalize(t)
        if not Config.within_base(self.t_abs): raise ResoneticsFatal("LOCK_ESCAPE_ATTEMPT")
        self.ex, self.to, self.fd, self.cm = ex, to, None, None
        self.lp = os.path.join(Config.get_lock_dir(), f"{hashlib.sha256(self.t_abs.encode()).hexdigest()[:16]}.lock")
    @property
    def lock_path(self): return self.lp
    def __enter__(self):
        _guard_no_state_lock(f"lock:{os.path.basename(self.t_abs)}")
        self.cm = io_context(); self.cm.__enter__()
        try:
            os.makedirs(Config.get_lock_dir(), 0o700, exist_ok=True); st_t = time.time()
            while True:
                try:
                    if IS_POSIX:
                        nf = getattr(os, "O_NOFOLLOW", 0); dfd = None
                        try:
                            dfd = _secure_dir_open(Config.get_lock_dir()); fname = os.path.basename(self.lp)
                            fd = os.open(fname, os.O_RDWR|os.O_CREAT|getattr(os,"O_CLOEXEC",0)|nf, 0o600, dir_fd=dfd)
                            f_st, l_st = os.fstat(fd), os.stat(fname, dir_fd=dfd, follow_symlinks=False)
                            if (f_st.st_dev, f_st.st_ino) != (l_st.st_dev, l_st.st_ino): os.close(fd); raise ResoneticsFatal("LOCK_MIRROR")
                            self.fd = os.fdopen(fd, "a+b")
                        finally:
                            if dfd is not None: os.close(dfd)
                        import fcntl; fcntl.flock(self.fd.fileno(), (fcntl.LOCK_EX if self.ex else fcntl.LOCK_SH)|fcntl.LOCK_NB)
                    elif IS_WINLOCK:
                        self.fd = open(self.lp, "a+b")
                        import msvcrt; msvcrt.locking(self.fd.fileno(), msvcrt.LK_NBLCK if self.ex else msvcrt.LK_NBRLCK, 1)
                    break
                except (IOError, OSError) as e:
                    if time.time()-st_t > self.to: raise TimeoutError(f"LOCK_TIMEOUT:{e}")
                    time.sleep(0.05)
            return self
        except:
            if self.fd: self.fd.close()
            self.cm.__exit__(None, None, None); raise
    def __exit__(self, *a):
        try:
            if self.fd:
                if IS_POSIX: import fcntl; fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
                elif IS_WINLOCK: import msvcrt; msvcrt.locking(self.fd.fileno(), msvcrt.LK_UNLCK, 1)
                self.fd.close()
        finally: self.cm.__exit__(*a)

class ResoneticCrypto:
    def __init__(self, kp):
        self.s = None
        with AtomicFileLock(str(kp), True):
            try:
                with self.secure_open(str(kp), "rb", oath=True) as f: self.s = f.read()
            except (FileNotFoundError, OSError): pass
            if not self.s or len(self.s)!=32:
                self.s = secrets.token_bytes(32)
                with self.secure_open(str(kp), "wb", oath=True) as f: f.write(self.s)
        self.fp = hashlib.sha256(self.s).hexdigest()[:12]
        with GLOBAL_STATE._lock: GLOBAL_STATE.key_fingerprint = self.fp

    def _is_hash64(self, x): return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x.lower())

    def _resync(self, p, off):
        try:
            with self.secure_open(p, "rb") as f:
                f.seek(0, 2); base = max(0, off-128); f.seek(base); b = f.read(65536)
                if not b: return off, False
                i, tries = b.rfind(b"\n"), 1024
                while i >= 0 and tries > 0:
                    tries -= 1; j = i+1; k = j
                    while k<len(b) and b[k:k+1].isdigit(): k+=1
                    if k>j and k<len(b) and b[k:k+1]==b":" and b[k+1:k+2]==b"{": return base+j, True
                    i = b.rfind(b"\n", 0, max(0, i-1))
            return off, False
        except: return off, False

    def ver_ent(self, e, flds, allow_extra=False):
        try:
            if not isinstance(e, dict): return False, "TYPE"
            if not all(k in e for k in flds): return False, "MISSING"
            if not allow_extra and (set(e.keys()) - set(flds) - {"sig", "hash"}): return False, "EXTRA"
            p = {k: e[k] for k in flds}
            cs = json.dumps(p, sort_keys=True, separators=(",",":"), ensure_ascii=False)
            sig = hmac.new(self.s, cs.encode(), hashlib.sha256).hexdigest().lower()
            if not hmac.compare_digest(sig, e.get("sig","").lower()): return False, "SIG"
            if "prev_hash" in e:
                h = hashlib.sha256((cs+"|sig:"+sig).encode()).hexdigest().lower()
                if not hmac.compare_digest(h, e.get("hash","").lower()): return False, "HASH"
            return True, "OK"
        except: return False, "EXC"

    def commit(self, fp, hp, bldr, exp=None):
        _guard_no_state_lock("com"); l1, l2 = AtomicFileLock(hp), AtomicFileLock(fp)
        ps = sorted([(l1.lock_path, l1), (l2.lock_path, l2)], key=lambda x:x[0])
        try:
            with ps[0][1], ps[1][1]:
                anc = (0, Config.GENESIS_HASH)
                try:
                    with self.secure_open(hp, "rb", oath=True) as f:
                        p = f.read().decode().split()
                        if len(p) >= 2 and p[0].isdigit() and self._is_hash64(p[1]): anc = (int(p[0]), p[1].lower())
                except: pass
                if exp is not None and anc[1]!=exp: raise ResoneticsHeadStale("SYNC_LOST", {"global_step":anc[0], "hash":anc[1]}, os.path.getsize(fp))
                pl = bldr(anc[0]+1, anc[1])
                cs = json.dumps({k: pl[k] for k in Config.SIG_FIELDS_V2}, sort_keys=True, separators=(",",":"), ensure_ascii=False)
                pl["sig"] = hmac.new(self.s, cs.encode(), hashlib.sha256).hexdigest().lower()
                pl["hash"] = hashlib.sha256((cs+"|sig:"+pl["sig"]).encode()).hexdigest().lower()
                with self.secure_open(fp, "ab", oath=True) as f:
                    d = json.dumps(pl, separators=(",",":"), ensure_ascii=False).encode()
                    f.write(f"{len(d)}:".encode()+d+b"\n"); f.flush(); os.fsync(f.fileno()); no = f.tell()
                with self.secure_open(f"{hp}.tmp", "wb", oath=False) as f:
                    f.write(f"{pl['global_step']} {pl['hash']}\n".encode()); f.flush(); os.fsync(f.fileno())
                os.replace(f"{hp}.tmp", hp); return pl, no
        except: raise

    def rebuild(self, lp, hp, seq, lh):
        ok, st_res, res, tip, off = self.verify(lp, hp=None, deep=True, vh=False, nb=True)
        if not ok or tip!=lh or st_res["vc"]!=seq: log_boot("REBUILD_FAIL", "FATAL"); raise ResoneticsFatal("EVID_MISMATCH")
        with AtomicFileLock(hp, True):
            with self.secure_open(f"{hp}.tmp", "wb", oath=False) as f:
                f.write(f"{seq} {lh.lower()}\n".encode()); f.flush(); os.fsync(f.fileno())
            os.replace(f"{hp}.tmp", hp)

    def snap(self, sp, st_dict, ah, ct):
        _guard_no_state_lock("snap")
        pl = {**st_dict, "anchor_hash": ah, "chain_tip": ct, "timestamp": datetime.now(timezone.utc).isoformat(),
              "_key_fp": self.fp, "snap_ver": "v2.1", "app_ver": Config.APP_VERSION, "ledger_offset": GLOBAL_STATE.ledger_offset}
        with AtomicFileLock(sp, True):
            cs = json.dumps({k: pl[k] for k in Config.SNAP_FIELDS_V2_1}, sort_keys=True, separators=(",",":"), ensure_ascii=False)
            pl["sig"] = hmac.new(self.s, cs.encode(), hashlib.sha256).hexdigest().lower()
            with self.secure_open(f"{sp}.tmp", "wb", oath=False) as f: f.write(json.dumps(pl, ensure_ascii=False).encode())
            os.replace(f"{sp}.tmp", sp); return True

    def load(self, sp):
        _guard_no_state_lock("load")
        try:
            with AtomicFileLock(sp, False):
                with self.secure_open(sp, "rb", oath=True) as f: d = json.loads(f.read().decode("utf-8"))
                for fs in [Config.SNAP_FIELDS_V2_1, Config.SNAP_FIELDS_V2_0]:
                    ok, _ = self.ver_ent(d, fs, allow_extra=(fs==Config.SNAP_FIELDS_V2_0))
                    if ok: return d, None
                return None, "SCHEMA"
        except Exception as e: return None, str(e)

    def verify(self, lp, hp=None, deep=False, off=0, eh=None, es=0, vh=True, nb=False,
               strict_stream=False, lock=None, budget=Config._UNSET, allow_extra=False,
               policy: Optional[PolicySpec] = None):
        if policy is not None:
            strict_stream, allow_extra, nb = policy.strict_stream, policy.allow_extra, policy.nb
            lock = policy.lock if lock is None else lock
            budget = policy.budget if policy.budget is not None else Config._UNSET
        is_resume, h_seq, h_hash, ever_head_unreadable = (off > 0), None, None, False
        use_budget = (Config.DEEP_SCAN_BUDGET if deep else Config.FAST_SCAN_BUDGET) if budget == Config._UNSET else budget
        if nb: use_budget = None
        for attempt in range(2):
            resume_immediate_fatal, base_off = False, off if attempt == 0 else 0
            t_eh, t_es = str(eh if attempt == 0 else Config.GENESIS_HASH).strip().lower(), (es if attempt == 0 else 0)
            target_lo, target_ok = self._resync(lp, base_off); lo = target_lo if target_ok else 0
            ep, pg, ctip, l_good, vf, s_cnt = t_eh, t_es, t_eh, lo, 0, 0
            with self.stream(p=lp, off=lo, hp=hp, lock=(lock if lock is not None else deep), strict=strict_stream, budget=use_budget) as fr:
                if vh and hp and os.path.exists(hp):
                    for _ in range(3):
                        try:
                            with self.secure_open(hp, "rb", oath=True) as f:
                                p = f.read().decode().split()
                                if len(p) >= 2 and p[0].isdigit() and self._is_hash64(p[1]): h_seq, h_hash = int(p[0]), p[1].lower(); break
                        except: pass
                        time.sleep(0.02)
                    else:
                        if deep: ever_head_unreadable = True; warn_once("HEAD_UNREAD", "HEAD_UNREADABLE")
                        else: return False, {"vc":pg, "vf":0}, "HEAD_READ_FAIL", ctip, l_good
                for ent, err, nx in fr:
                    if err:
                        if attempt==0 and is_resume and vf==0 and err.get("code")=="FATAL": resume_immediate_fatal = True; break
                        if err.get("code")=="SKIP": s_cnt += 1; l_good = nx; continue
                        if err.get("code")=="EOF": break
                        if err.get("code")=="BUDGET":
                            return True, {"vc":pg, "vf":vf, "sk":s_cnt, "ct":ctip, "resume_off":l_good, "resume_prev_hash":t_eh, "next_prev_hash": ep, "resume_base_step":pg}, "YIELD", ctip, l_good
                        return False, {"vc":pg, "vf":vf}, err.get("msg","ERR"), ctip, l_good
                    ok_e, why = self.ver_ent(ent, Config.SIG_FIELDS_V2, allow_extra=allow_extra); vf += 1
                    if not ok_e: return False, {"vc":pg, "vf":vf}, f"SIG:{why}", ctip, l_good
                    l_good, ftip, est = nx, ent["hash"].lower(), int(ent["global_step"])
                    if (est==pg+1 and ent["prev_hash"].lower()==ep): pg, ep, ctip = est, ftip, ftip
                    else: return False, {"vc":pg, "vf":vf}, "CHAIN_BREAK", ctip, l_good
                    if not deep and h_seq is not None and pg>=h_seq: break
            if resume_immediate_fatal and attempt == 0:
                if not deep: return True, {"vc":pg, "vf":vf, "resume_off":off, "resume_prev_hash":t_eh}, "NEEDS_DEEP", ctip, off
                continue
            break
        hr = ever_head_unreadable or (not vh or h_seq is None) or (pg == h_seq and ctip == h_hash)
        if vh and hp and not ever_head_unreadable and h_seq is not None and not hr:
            if h_seq > pg: return True, {"vc":pg, "vf":vf, "sk":s_cnt, "ct":ctip, "resume_off":l_good, "resume_prev_hash":t_eh, "next_prev_hash": ep, "head_seq":h_seq, "head_hash":h_hash}, "OK_HEAD_MOVED_FORWARD", ctip, l_good
            return False, {"vc":pg, "vf":vf}, "HEAD_MIS", ctip, l_good
        return True, {"vc":pg, "vf":vf, "sk":s_cnt, "ct":ctip, "resume_off":l_good, "resume_prev_hash":t_eh, "next_prev_hash": ep, "head_reached":hr}, "OK", ctip, l_good

    @staticmethod
    @contextmanager
    def secure_open(p, m="rb", oath=False):
        """[Security Core] Fix 1🚨: Semantic-aware modification policy for Win."""
        if not p: raise ValueError("OPEN_TARGET_NONE")
        if 'GLOBAL_CORE_KEYS' not in st.session_state: base_dir_safety_check()

        # Fix 1.1🚨: Correctly define temp requirement (Only for Overwrite/Creation)
        is_overwrite = ("w" in m or "x" in m)
        can_modify = any(ch in m for ch in ("w", "x", "a", "+"))
        target_rs = Config._canonicalize(str(p)); parent_dir = os.path.dirname(target_rs)

        if not (Config.within_base(target_rs) and Config.within_base(parent_dir)): raise ResoneticsFatal("ESCAPE_ATTEMPT")

        with io_context():
            nf, cx = getattr(os, "O_NOFOLLOW", 0), getattr(os, "O_CLOEXEC", 0)
            fd, fobj, temp_path = None, None, None

            # Fix 1.2🚨: Use temp file ONLY for Overwrite modes to preserve Append/Update semantics
            if IS_WINLOCK and is_overwrite:
                tmp_dir = Config.get_temp_dir(); os.makedirs(tmp_dir, 0o700, exist_ok=True)
                ta = _GetFileAttributesW(str(tmp_dir))
                if ta == Config.INVALID_FILE_ATTRIBUTES or (ta & Config.FILE_ATTRIBUTE_REPARSE_POINT): raise ResoneticsFatal("WIN_TMP_DIR_DIRTY")
                temp_path = os.path.join(tmp_dir, f"at_{secrets.token_hex(4)}.tmp"); actual_op_path = temp_path
            else: actual_op_path = target_rs

            if IS_POSIX:
                dfd = _secure_dir_open(parent_dir)
                try:
                    if m.startswith("r"): flags = os.O_RDWR if "+" in m else os.O_RDONLY
                    elif "x" in m: flags = (os.O_RDWR if "+" in m else os.O_WRONLY) | (os.O_CREAT | os.O_EXCL)
                    else:
                        flags = (os.O_RDWR if "+" in m else os.O_WRONLY) | os.O_CREAT
                        if "w" in m: flags |= os.O_TRUNC
                        elif "a" in m: flags |= os.O_APPEND

                    fname = os.path.basename(target_rs)
                    fd = os.open(fname, flags | nf | cx, 0o600, dir_fd=dfd)
                    # Promotion handled within identity auditor
                    _audit_handle_identity(fd, os.path.join(parent_dir, fname), logical_path=target_rs, force_oath=oath)
                    fobj = os.fdopen(fd, m); fd = None; yield fobj
                finally:
                    if fobj: fobj.close()
                    if fd is not None: os.close(fd)
                    if dfd is not None: os.close(dfd)
            else:
                # Pre-open string check
                if can_modify:
                    pa_attrs = _GetFileAttributesW(str(parent_dir))
                    if pa_attrs == Config.INVALID_FILE_ATTRIBUTES or (pa_attrs & Config.FILE_ATTRIBUTE_REPARSE_POINT): raise ResoneticsFatal("WIN_PARENT_DIRTY")
                    if os.path.exists(target_rs):
                        ta = _GetFileAttributesW(str(target_rs))
                        if ta == Config.INVALID_FILE_ATTRIBUTES or (ta & Config.FILE_ATTRIBUTE_REPARSE_POINT): raise ResoneticsFatal("WIN_FILE_REPARSE")

                fobj = open(actual_op_path, m)
                try:
                    _audit_handle_identity(fobj.fileno(), target_rs, logical_path=target_rs, force_oath=oath); yield fobj
                    if temp_path:
                        fobj.close(); fobj = None
                        pa_recheck = _GetFileAttributesW(str(parent_dir))
                        if pa_recheck == Config.INVALID_FILE_ATTRIBUTES or (pa_recheck & Config.FILE_ATTRIBUTE_REPARSE_POINT): raise ResoneticsFatal("WIN_TARGET_PARENT_DIRTY")
                        os.replace(temp_path, target_rs)
                        with open(target_rs, "rb") as final_f: _audit_handle_identity(final_f.fileno(), target_rs, logical_path=target_rs, force_oath=oath)
                finally:
                    if fobj: fobj.close()
                    if temp_path and os.path.exists(temp_path): os.remove(temp_path)

    @contextmanager
    def stream(self, p, off=0, hp=None, lock=False, strict=True, budget=None):
        def _g():
            deadline, skips, last_pos = (time.time() + budget) if budget else None, 0, off
            try:
                with ExitStack() as stack:
                    if lock:
                        targets = [p]
                        if hp and os.path.exists(hp): targets.append(hp)
                        for l in [AtomicFileLock(x, False) for x in targets]: stack.enter_context(l)
                    with self.secure_open(p, "rb", oath=True) as f:
                        if off>0: f.seek(off)
                        while True:
                            if deadline and time.time() > deadline: yield None, {"code": "BUDGET"}, f.tell(); return
                            if skips >= Config.MAX_SKIPS_ALLOWED: yield None, {"code": "FATAL", "msg": "SKIPS"}, f.tell(); return
                            ws_sk = 0
                            while (ch := f.read(1)) in b"\n\r\t ":
                                ws_sk += 1
                                if ws_sk > Config.MAX_SKIP_BYTES: yield None, {"code":"FATAL", "msg":"WS"}, f.tell(); return
                            if not ch: yield None, {"code":"EOF"}, f.tell(); return
                            f.seek(f.tell()-1); hst, lb = f.tell(), b""
                            while (c := f.read(1)) != b":":
                                if not c or len(lb) > Config.MAX_HEADER_SIZE: yield None, {"code":"FATAL", "msg":"HDR"}, hst; return
                                lb += c
                            try: length = int(lb)
                            except: yield None, {"code":"FATAL", "msg":"BAD_LEN"}, hst; return
                            if length > Config.MAX_FRAME_SIZE: yield None, {"code":"FATAL", "msg":"BIG"}, hst; return
                            payload = f.read(length)
                            if len(payload) != length or f.read(1) != b"\n": yield None, {"code":"FATAL", "msg":"FRM"}, hst; return
                            try: yield json.loads(payload.decode("utf-8")), None, f.tell()
                            except Exception as e:
                                if strict: yield None, {"code": "FATAL", "msg": f"PARSE:{e}"}, hst; return
                                next_off, found = self._resync(p, f.tell())
                                if found: skips += 1; yield None, {"code": "SKIP"}, next_off; f.seek(next_off)
                                else: return
            except FileNotFoundError: yield None, {"code":"EOF"}, off; return
            except Exception as e: yield None, {"code": "FATAL", "msg": str(e)}, last_pos
        g = _g()
        try: yield g
        finally:
            try: g.close()
            except Exception: pass

class ResoneticAgent:
    def __init__(self):
        self.c = ResoneticCrypto(Config.KEY_FILE_P); self.gs, self.hc, self.phs = 0, Config.GENESIS_HASH, 0
        self.ps = PhysicsState(); self.pl = threading.Lock(); self._boot()
    def _boot(self):
        base_dir_safety_check(); sn, _ = self.c.load(str(Config.SNAP_FILE_P))
        so, sh, ss = 0, Config.GENESIS_HASH, 0
        if sn:
            ss, sh, so = sn["global_step"], sn["chain_tip"], sn.get("ledger_offset", 0)
            self.ps.current_s = sn.get("metrics",{}).get("entropy", 0.0)
        with GLOBAL_STATE._lock: GLOBAL_STATE.snapshot_loaded = bool(sn)
        p = PolicySpec.from_dict("BOOT", Config.POLICIES["BOOT"])
        ok, st_res, res, tip, off = self.c.verify(
            str(Config.LEDGER_FILE_P), str(Config.HEAD_FILE_P), deep=True, off=so, eh=sh, es=ss, policy=p
        )
        self.gs, self.hc = st_res.get("vc", 0), tip
        with GLOBAL_STATE._lock: GLOBAL_STATE.ledger_offset = off; log_boot(f"ONLINE G{self.gs} ({res})")

    def tick(self):
        if not self.pl.acquire(blocking=False): return
        try:
            if GLOBAL_STATE.running and not GLOBAL_STATE.quarantine and (time.time()-GLOBAL_STATE.last_commit_time >= Config.COMMIT_INTERVAL):
                self.ps.current_s = max(0.0001, self.ps.current_s - 0.005)
                self.ps.current_phi = max(0.1, 1.0 - self.ps.current_s * 1.25); self.phs += 1
                try:
                    pl, no = self.c.commit(str(Config.LEDGER_FILE_P), str(Config.HEAD_FILE_P),
                        lambda n,p: {"prev_hash":p, "global_step":n, "physics_step":self.phs, "episode":0, "episode_step":0, "type":"PULSE", "status":"OK", "msg":"Pulse",
                                     "metrics":{"entropy":round(self.ps.current_s, 6), "stability":round(self.ps.current_phi, 6)},
                                     "meta":{}, "time":datetime.now(timezone.utc).isoformat(), "ver":"v2"}, self.hc)
                    self.gs, self.hc = pl["global_step"], pl["hash"]
                    with GLOBAL_STATE._lock: GLOBAL_STATE.last_commit_time, GLOBAL_STATE.ledger_offset = time.time(), no
                except ResoneticsHeadStale as e:
                    self.c.rebuild(str(Config.LEDGER_FILE_P), str(Config.HEAD_FILE_P), e.payload["global_step"], e.payload["hash"])
                    self.gs, self.hc = e.payload["global_step"], e.payload["hash"]
                    with GLOBAL_STATE._lock: GLOBAL_STATE.ledger_offset = e.new_offset
        finally: self.pl.release()

    def audit(self, deep=False):
        _guard_no_state_lock("audit")
        if flush_pending_logs() == -1: return False
        with GLOBAL_STATE._lock:
            s_off, s_hash, s_step = GLOBAL_STATE.audit_resume_off, GLOBAL_STATE.audit_next_prev_hash, GLOBAL_STATE.audit_last_verified_step
        try:
            cur_sz = os.path.getsize(str(Config.LEDGER_FILE_P))
            if cur_sz < s_off: s_off, s_hash, s_step = 0, Config.GENESIS_HASH, 0
        except: pass
        name = "AUDIT_DEEP" if deep else "AUDIT_FAST"; p = PolicySpec.from_dict(name, Config.POLICIES[name])
        ok, st_dict, res, tip, off = self.c.verify(
            str(Config.LEDGER_FILE_P), str(Config.HEAD_FILE_P), deep, off=s_off, eh=s_hash, es=s_step, policy=p
        )
        with GLOBAL_STATE._lock:
            GLOBAL_STATE.last_audit_time = time.time()
            GLOBAL_STATE.last_audit_frames = f"{st_dict.get('vc', 0)}/{st_dict.get('vf', 0)}"
            GLOBAL_STATE.audit_resume_off = st_dict.get("resume_off", off)
            GLOBAL_STATE.audit_next_prev_hash = st_dict.get("next_prev_hash", st_dict.get("resume_prev_hash", Config.GENESIS_HASH))
            GLOBAL_STATE.audit_last_verified_step = st_dict.get("vc", 0)
            GLOBAL_STATE.last_audit_result = f"{res} ({GLOBAL_STATE.last_audit_frames})"
        return (ok and res.startswith("OK"))

agent = st.session_state.get("RES_AGENT") or ResoneticAgent(); st.session_state.RES_AGENT = agent

# ---- UI ----
st.sidebar.markdown(f"**Version:** {Config.APP_VERSION}")
if not GLOBAL_STATE.quarantine:
    current_running = GLOBAL_STATE.running
    new_running = st.sidebar.toggle("Engine Active", value=current_running)
    if new_running != current_running:
        with GLOBAL_STATE._lock: GLOBAL_STATE.running = new_running
    if st.sidebar.button("Fast Audit"): agent.audit(False)
    if st.sidebar.button("Deep Audit"): agent.audit(True)
else:
    st.sidebar.error(f"QUARANTINED: {GLOBAL_STATE.root_cause}")

@st.fragment(run_every=Config.PULSE_INTERVAL)
def pulse_monitor():
    agent.tick(); flush_pending_logs()
    st.markdown(f"### 🔥 Step: {agent.gs} | Stability: {agent.ps.current_phi:.4f}")
    st.sidebar.info(f"Audit Status: {GLOBAL_STATE.last_audit_result}")
    if GLOBAL_STATE.snapshot_loaded: st.sidebar.success("Snapshot Restored")
pulse_monitor()

