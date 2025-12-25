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
import random
import atexit
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timezone
from contextlib import nullcontext

# ==============================================================================
# [Constitutional Order] Streamlit Setup & Global Authority
# ==============================================================================
import streamlit as st
st.set_page_config(layout="wide", page_title="Resonetics v36.4 Diamond Fortress")

@dataclass
class ProcessStateData:
    last_audit_time: float = 0.0
    last_audit_str: str = "Never"
    last_audit_result: str = "-"
    last_audit_msg: str = "-"
    quarantine: bool = False
    running: bool = False
    last_pulse: float = 0.0
    boot_logs: deque = field(default_factory=lambda: deque(maxlen=200), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

@st.cache_resource
def get_global_state(): return ProcessStateData()
GLOBAL_STATE = get_global_state()

def log_boot(msg, is_error=False):
    prefix = "🚨 " if is_error else "🛡️ "
    timestamp = datetime.now().strftime('%H:%M:%S')
    GLOBAL_STATE.boot_logs.append(f"{prefix}{timestamp} {msg}")

# ==============================================================================
# [System] Platform Sensing
# ==============================================================================
LOCK_IMPL_TYPE = "None (Unsafe)"
try:
    import fcntl
    LOCK_IMPL_TYPE = "fcntl (POSIX)"
except Exception:
    try:
        import msvcrt
        LOCK_IMPL_TYPE = "msvcrt (Windows)"
    except Exception: pass

IS_POSIX = LOCK_IMPL_TYPE.startswith("fcntl")
IS_WINLOCK = LOCK_IMPL_TYPE.startswith("msvcrt")
IS_NOLOCK = LOCK_IMPL_TYPE.startswith("None")

# ==============================================================================
# 1. Configuration (Diamond Fortress Standards)
# ==============================================================================
class Config:
    GENESIS_HASH = "0" * 64
    LEDGER_FILE = "resonetics_v36.ledger"
    HEAD_FILE = LEDGER_FILE + ".head"
    KEY_FILE = f"{LEDGER_FILE}.key"
    BASE_DIR = os.path.dirname(os.path.abspath(LEDGER_FILE))

    @classmethod
    def within_base(cls, path: str) -> bool:
        base = os.path.realpath(cls.BASE_DIR)
        rp = os.path.realpath(os.path.abspath(path))
        return rp == base or rp.startswith(base + os.sep)

    @classmethod
    def get_lock_dir(cls): return os.path.join(cls.BASE_DIR, ".locks")

    MAX_HEADER_SIZE = 20
    MAX_FRAME_SIZE = 256 * 1024 
    PULSE_INTERVAL = 0.5 
    DEV_MODE = os.environ.get("RESONETICS_DEV_MODE") == "1"
    SIG_FIELDS = ["prev_hash", "global_step", "physics_step", "episode", "episode_step", "type", "status", "msg", "metrics", "meta", "time"]

# ==============================================================================
# 2. Crypto & Atomic I/O Engine (The Diamond Fortress)
# ==============================================================================
class AtomicFileLock:
    def __init__(self, target_filename, exclusive=True, timeout=5.0, dev_mode=False):
        target_abs = os.path.realpath(os.path.abspath(target_filename))
        if not Config.within_base(target_abs): raise RuntimeError("⛔ Territory escape.")
        self.LOCK_DIR = Config.get_lock_dir()
        os.makedirs(self.LOCK_DIR, exist_ok=True)
        self.dev_mode = dev_mode

        if os.name != 'nt':
            try: os.chmod(self.LOCK_DIR, 0o700)
            except: pass
            st_dir = os.lstat(self.LOCK_DIR)
            if stat.S_ISLNK(st_dir.st_mode) or not stat.S_ISDIR(st_dir.st_mode):
                raise RuntimeError("⛔ SECURITY: Lock territory compromised.")
            # ✅ Patch: Owner validation for multi-user safety
            if st_dir.st_uid != os.getuid():
                raise RuntimeError("⛔ SECURITY: Territory owner mismatch.")
            if (st_dir.st_mode & 0o077) != 0:
                raise RuntimeError("⛔ SECURITY: Territory perms too open.")

        self.lock_file = os.path.join(self.LOCK_DIR, f"{hashlib.sha256(target_abs.encode()).hexdigest()[:16]}.lock")
        self.handle, self.exclusive, self.timeout = None, exclusive, timeout

    def __enter__(self):
        if IS_NOLOCK and not self.dev_mode: raise RuntimeError("⛔ FATAL: No lock support.")
        start_time = time.time()
        try:
            if os.name != 'nt':
                flags = os.O_RDWR | os.O_CREAT
                if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
                if hasattr(os, "O_CLOEXEC"): flags |= os.O_CLOEXEC
                fd = os.open(self.lock_file, flags, 0o600)
                stf = os.fstat(fd)
                if not stat.S_ISREG(stf.st_mode) or stf.st_uid != os.getuid():
                    os.close(fd); raise RuntimeError("⛔ SECURITY: Lock Inode violation.")
                self.handle = os.fdopen(fd, "a+b")
            else: 
                self.handle = open(self.lock_file, "a+b")
                st_info = os.lstat(self.lock_file)
                attrs = getattr(st_info, "st_file_attributes", 0)
                if stat.S_ISLNK(st_info.st_mode) or (attrs & 0x400):
                    self.handle.close(); raise RuntimeError("⛔ SECURITY: Windows Reparse violation.")

            while True:
                try:
                    if IS_POSIX:
                        import fcntl
                        fcntl.flock(self.handle.fileno(), (fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                        break
                    elif IS_WINLOCK:
                        import msvcrt
                        self.handle.seek(0); msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                except (IOError, OSError):
                    if time.time() - start_time > self.timeout: raise TimeoutError("Lock timeout.")
                    time.sleep(0.05 + random.uniform(0, 0.1))
            return self
        except Exception as e:
            if self.handle: self.handle.close()
            raise RuntimeError(f"⛔ LOCK FAIL: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle:
            try:
                if IS_POSIX:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                elif IS_WINLOCK:
                    import msvcrt
                    self.handle.seek(0); msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally: self.handle.close()

class ResoneticCrypto:
    EXCLUDE_CANON = {"sig", "hash"}
    def __init__(self, key_path):
        self.secret = None
        with AtomicFileLock(key_path, exclusive=True, dev_mode=Config.DEV_MODE):
            if os.path.exists(key_path):
                with self.secure_open(key_path, "rb") as f: self.secret = f.read()
            elif Config.DEV_MODE:
                self.secret = secrets.token_bytes(32)
                with self.secure_open(key_path, "wb", exclusive=True) as f:
                    f.write(self.secret); f.flush(); os.fsync(f.fileno())
                log_boot("Immutable root key secured.")
            else: raise RuntimeError("⛔ FATAL: Root key missing.")

    @staticmethod
    def secure_open(filepath, mode="rb", exclusive=False):
        if not Config.within_base(filepath): raise RuntimeError("⛔ Territorial escape.")
        filepath = os.path.realpath(os.path.abspath(filepath))
        if os.name != 'nt':
            flags = os.O_RDONLY if mode == "rb" else (os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else (os.O_APPEND if mode == "ab" else os.O_TRUNC)))
            if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
            if hasattr(os, "O_CLOEXEC"): flags |= os.O_CLOEXEC
            fd = os.open(filepath, flags, 0o600)
            stf = os.fstat(fd)
            if not stat.S_ISREG(stf.st_mode) or stf.st_uid != os.getuid():
                os.close(fd); raise RuntimeError("⛔ SECURITY: Inode violation.")
            return os.fdopen(fd, mode)
        else:
            f = open(filepath, mode)
            try:
                st_info = os.lstat(filepath)
                attrs = getattr(st_info, "st_file_attributes", 0)
                if stat.S_ISLNK(st_info.st_mode) or (attrs & 0x400):
                    f.close(); raise RuntimeError("⛔ SECURITY: Windows Reparse violation.")
                return f
            except: f.close(); raise

    def canonical_str(self, entry_like: dict) -> str:
        def _deep(v, depth=0):
            if depth > 6: return "__DEPTH__"
            if v is None or isinstance(v, (bool, int)): return v
            if isinstance(v, float): return round(v, 6)
            if isinstance(v, str): return v[:500]
            if isinstance(v, dict):
                items = sorted(list(v.items()), key=lambda x: str(x[0]))[:50]
                return {str(k): _deep(val, depth + 1) for k, val in items}
            if isinstance(v, (list, tuple)): return [_deep(x, depth + 1) for x in list(v)[:50]]
            return str(v)
        is_signed = "global_step" in entry_like
        payload = {}
        target = Config.SIG_FIELDS if is_signed else sorted(entry_like.keys())
        for k in target:
            if k in self.EXCLUDE_CANON: continue
            if is_signed and k not in entry_like: raise ValueError(f"SCHEMA ERROR: {k}")
            if k in entry_like: payload[k] = _deep(entry_like[k])
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def verify_entry(self, entry: dict):
        try:
            sig = hmac.new(self.secret, self.canonical_str(entry).encode(), hashlib.sha256).hexdigest()
            ok = hmac.compare_digest(sig, entry.get("sig", ""))
            return ok, ("OK" if ok else "SIG_MISMATCH")
        except Exception as e: return False, str(e)

    def commit_transaction(self, filepath, head_path, builder, expected=None):
        with AtomicFileLock(head_path, exclusive=True, dev_mode=Config.DEV_MODE):
            with AtomicFileLock(filepath, exclusive=True, dev_mode=Config.DEV_MODE):
                anchor_data = None
                if os.path.exists(head_path):
                    with self.secure_open(head_path, "rb") as f:
                        parts = f.read().decode().strip().split()
                        if len(parts) == 2:
                            seq_s, h = parts[0], parts[1].lower()
                            if seq_s.isdigit() and len(h) == 64 and all(c in "0123456789abcdef" for c in h):
                                anchor_data = (int(seq_s), h)
                            else: raise RuntimeError("⛔ HEAD_CORRUPT")
                
                prev_seq, prev_hash = anchor_data if anchor_data else (0, Config.GENESIS_HASH)
                if expected and prev_hash != expected: raise RuntimeError("SPLIT_BRAIN")
                
                payload = builder(prev_seq + 1, prev_hash)
                cstr = self.canonical_str(payload)
                payload["sig"] = hmac.new(self.secret, cstr.encode(), hashlib.sha256).hexdigest()
                payload["hash"] = hashlib.sha256((cstr + "|sig:" + payload["sig"]).encode()).hexdigest()
                
                data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                frame = f"{len(data)}:".encode('ascii') + data + b"\n"
                with self.secure_open(filepath, "ab") as f:
                    f.write(frame); f.flush(); os.fsync(f.fileno())
                
                nonce = secrets.token_hex(4); tmp = f"{head_path}.{nonce}.tmp"
                with self.secure_open(tmp, "wb", exclusive=True) as f:
                    f.write(f"{prev_seq + 1} {payload['hash']}\n".encode()); f.flush(); os.fsync(f.fileno())
                os.replace(tmp, head_path)
                return payload

    @staticmethod
    def stream_frames(filepath, lock_read=False, strict=True):
        if not os.path.exists(filepath): yield None, "NO_FILE"; return
        lock_ctx = AtomicFileLock(filepath, exclusive=False, dev_mode=Config.DEV_MODE) if lock_read else nullcontext()
        with lock_ctx:
            with ResoneticCrypto.secure_open(filepath, "rb") as f:
                while True:
                    while True: # Gap skip
                        pos = f.tell(); ch = f.read(1)
                        if not ch: return
                        if ch not in b"\n\r\t ": f.seek(pos); break
                    len_buf = b""
                    while True: # Header
                        char = f.read(1)
                        if not char or char == b":": break
                        if not b'0' <= char <= b'9': yield None, "MALFORMED_HEADER"; return
                        len_buf += char
                        if len(len_buf) > Config.MAX_HEADER_SIZE: yield None, "OVERFLOW"; return
                    if not len_buf: return
                    try: length = int(len_buf)
                    except: yield None, "MALFORMED_HEADER"; return
                    if length > Config.MAX_FRAME_SIZE: yield None, "SIZE_FAIL"; return
                    payload = f.read(length)
                    if len(payload) != length:
                        yield None, ("TRUNCATED" if strict else "TAIL_TRUNCATED"); return 
                    try: 
                        text = payload.decode("utf-8")
                        yield json.loads(text), None
                        while True: # Post-frame cleanup
                            pos = f.tell(); ch = f.read(1)
                            if not ch: break
                            if ch not in b"\n\r\t ": f.seek(pos); break
                    except: yield None, "JSON_FAIL"

    # ✅ Patch 1: Restored Missing Recovery Heart
    def force_rebuild_head(self, head_path, seq, last_hash):
        with AtomicFileLock(head_path, exclusive=True, dev_mode=Config.DEV_MODE):
            nonce = secrets.token_hex(4); tmp = f"{head_path}.{nonce}.tmp"
            with self.secure_open(tmp, "wb", exclusive=True) as f:
                f.write(f"{seq} {last_hash}\n".encode()); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, head_path)
            log_boot(f"Anchor Rebuilt from verified ledger tip: G:{seq} | {last_hash[:8]}")

# ==============================================================================
# 3. Agent Implementation
# ==============================================================================
class ResoneticAgent:
    def __init__(self):
        self.proc_state = GLOBAL_STATE
        self._data_lock = threading.RLock()
        self.crypto = ResoneticCrypto(Config.KEY_FILE)
        self.global_step, self.hash_chain = 0, Config.GENESIS_HASH
        self.physics_step = 0
        self.ui_buffer = deque(maxlen=500)
        self._secure_boot()

    def _secure_boot(self):
        log_boot("Initiating Diamond Fortress Restoration")
        if not os.path.exists(Config.LEDGER_FILE):
            with self.crypto.secure_open(Config.LEDGER_FILE, "wb", exclusive=True) as f: f.write(b"")
        
        expected_prev, prev_global, total = Config.GENESIS_HASH, 0, 0
        last_valid_entry = None
        stream = self.crypto.stream_frames(Config.LEDGER_FILE, lock_read=True, strict=False)
        
        for entry, error in stream:
            if error == "TAIL_TRUNCATED": log_boot("Open tail frame detected (Repairable)."); break
            if error: log_boot(f"BOOT ERR: {error}", True); break 
            ok, msg = self.crypto.verify_entry(entry)
            if not ok or entry["global_step"] != prev_global + 1 or entry["prev_hash"] != expected_prev:
                log_boot(f"INTEGRITY FAIL: {msg} @ G:{entry.get('global_step','?')}", True)
                with self.proc_state._lock: self.proc_state.quarantine = True
                return 
            last_valid_entry = entry
            expected_prev, prev_global, total = entry["hash"], entry["global_step"], total + 1
            self.ui_buffer.append(entry)
        
        if last_valid_entry:
            self.hash_chain, self.global_step = last_valid_entry["hash"], last_valid_entry["global_step"]
            self.physics_step = last_valid_entry.get("physics_step", 0)

        if os.path.exists(Config.HEAD_FILE):
            with self.crypto.secure_open(Config.HEAD_FILE, "rb") as f:
                parts = f.read().decode().strip().split()
                if len(parts) != 2 or int(parts[0]) != self.global_step or parts[1] != self.hash_chain:
                    log_boot("Anchor desync detected. Rebuilding head from last verified ledger tip.", True)
                    self.crypto.force_rebuild_head(Config.HEAD_FILE, self.global_step, self.hash_chain)
        else: self.crypto.force_rebuild_head(Config.HEAD_FILE, self.global_step, self.hash_chain)
        log_boot(f"Diamond Kernel Online. G:{self.global_step}")

    def tick(self):
        with self._data_lock:
            next_physics = self.physics_step + 1 
            ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            def build_payload(next_seq, prev_hash):
                return {"prev_hash": prev_hash, "global_step": next_seq, "physics_step": next_physics, "episode": 0, "episode_step": 0, "type": "STEP_COMMIT", "status": "OK", "msg": "Fortress pulse", "metrics": {}, "meta": {}, "time": ts}
            try:
                entry = self.crypto.commit_transaction(Config.LEDGER_FILE, Config.HEAD_FILE, build_payload, expected=self.hash_chain)
                self.global_step, self.hash_chain = entry["global_step"], entry["hash"]
                self.physics_step = next_physics 
                self.ui_buffer.append(entry)
            except Exception as e:
                log_boot(f"COMMIT FAIL: {e}", True)
                with self.proc_state._lock: self.proc_state.quarantine = True

@st.cache_resource
def get_agent(): return ResoneticAgent()
agent = get_agent()

# ==============================================================================
# 4. UI Section
# ==============================================================================
st.sidebar.title("💎 v36.4 Diamond Fortress")
if IS_WINLOCK:
    st.sidebar.warning("🛡️ STATUS: DEGRADED (Win TOCTOU)")

with st.sidebar.expander("🧾 Forensic Boot Log", expanded=False):
    for line in list(GLOBAL_STATE.boot_logs)[-20:][::-1]: st.caption(line)

@st.fragment(run_every=0.5)
def pulse_monitor():
    now = time.time()
    if GLOBAL_STATE.running and not GLOBAL_STATE.quarantine:
        do_tick = False
        with GLOBAL_STATE._lock:
            if now - GLOBAL_STATE.last_pulse >= 0.5:
                GLOBAL_STATE.last_pulse, do_tick = now, True
        if do_tick: agent.tick()
    st.write(f"### 🌌 Ledger Step: {agent.global_step}")
    st.json({"physics": agent.physics_step, "quarantine": GLOBAL_STATE.quarantine, "chain_tip": agent.hash_chain[:16]})

if st.sidebar.button("Start/Stop Pulse", disabled=GLOBAL_STATE.quarantine):
    with GLOBAL_STATE._lock: GLOBAL_STATE.running = not GLOBAL_STATE.running
    st.rerun()
pulse_monitor()

