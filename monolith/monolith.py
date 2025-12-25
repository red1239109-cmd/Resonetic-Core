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
st.set_page_config(layout="wide", page_title="Resonetics v34.7 Sealed")

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
def get_global_state():
    return ProcessStateData()

# [Fix 1] Static Global Reference to avoid cache re-entry overhead
GLOBAL_STATE = get_global_state()

def log_boot(msg, is_error=False):
    prefix = "🚨 " if is_error else "🛡️ "
    timestamp = datetime.now().strftime('%H:%M:%S')
    GLOBAL_STATE.boot_logs.append(f"{prefix}{timestamp} {msg}")

# ==============================================================================
# [System] Lock Implementation & Platform Sensing
# ==============================================================================
LOCK_IMPL_TYPE = "None (Unsafe)"
try:
    import fcntl
    LOCK_IMPL_TYPE = "fcntl (POSIX)"
except Exception:
    try:
        import msvcrt
        LOCK_IMPL_TYPE = "msvcrt (Windows)"
    except Exception:
        LOCK_IMPL_TYPE = "None (Unsafe)"

IS_POSIX = LOCK_IMPL_TYPE.startswith("fcntl")
IS_WINLOCK = LOCK_IMPL_TYPE.startswith("msvcrt")
IS_NOLOCK = LOCK_IMPL_TYPE.startswith("None")

if "lock_degraded" not in st.session_state:
    st.session_state.lock_degraded = IS_WINLOCK or IS_NOLOCK

# ==============================================================================
# 1. Configuration (Sealed Standards)
# ==============================================================================
class Config:
    GENESIS_HASH = "0" * 64
    LEDGER_FILE = "resonetics_sealed.ledger"
    HEAD_FILE = LEDGER_FILE + ".head"
    AUDIT_FILE = "resonetics_sealed.audit"
    AUDIT_HEAD_FILE = AUDIT_FILE + ".head"
    KEY_FILE = f"{LEDGER_FILE}.key"

    @classmethod
    def get_lock_dir(cls):
        return os.path.join(os.path.dirname(os.path.abspath(cls.LEDGER_FILE)), ".locks")

    MAX_HEADER_SIZE = 20
    MAX_FRAME_SIZE = 256 * 1024
    MAX_DICT_KEYS = 50
    MAX_LIST_ITEMS = 50
    MAX_STR_LEN = 500
    PULSE_INTERVAL = 0.5
    DEV_MODE = os.environ.get("RESONETICS_DEV_MODE") == "1"

    ALLOWED_EVENTS = {
        "SYS_BOOT", "EPISODE_OPEN", "EPISODE_RESET", "STEP_COMMIT",
        "FAILSAFE_HOLD", "SYS_RECOVERY",
        "SYS_QUARANTINE_ON", "SYS_QUARANTINE_OFF",
        "POLICY_DENY", "AUDIT_PASS", "AUDIT_FAIL"
    }
    QUARANTINE_ALLOW = {
        "SYS_BOOT", "SYS_QUARANTINE_ON", "SYS_QUARANTINE_OFF",
        "FAILSAFE_HOLD", "SYS_RECOVERY", "POLICY_DENY",
        "AUDIT_PASS", "AUDIT_FAIL"
    }
    SIG_FIELDS = ["prev_hash", "global_step", "physics_step", "episode", "episode_step",
                  "type", "status", "msg", "metrics", "meta", "time"]
    AUDIT_SIG_FIELDS = ["audit_seq", "prev_hash", "timestamp", "result", "msg", "meta"]

# ==============================================================================
# 2. Crypto & Atomic I/O Engine (Sealed Logic)
# ==============================================================================
class AtomicFileLock:
    def __init__(self, target_filename, exclusive=True, timeout=5.0, dev_mode=False):
        self.LOCK_DIR = Config.get_lock_dir()
        os.makedirs(self.LOCK_DIR, exist_ok=True)
        if os.name != 'nt':
            try:
                os.chmod(self.LOCK_DIR, 0o700)
            except Exception:
                pass

        # [Fix 5] territorial sovereignty enforcement
        st_dir = os.lstat(self.LOCK_DIR)
        if stat.S_ISLNK(st_dir.st_mode) or not stat.S_ISDIR(st_dir.st_mode):
            raise RuntimeError("⛔ SECURITY: Invalid lock territory.")
        if os.name != 'nt':
            if st_dir.st_uid != os.getuid():
                raise RuntimeError("⛔ SECURITY: Territory owner mismatch.")
            if (st_dir.st_mode & 0o077) != 0:
                raise RuntimeError("⛔ SECURITY: Territory perms too open.")

        target_abs = os.path.realpath(os.path.abspath(target_filename))
        lock_hash = hashlib.sha256(target_abs.encode()).hexdigest()[:16]
        self.lock_file = os.path.join(self.LOCK_DIR, f"{lock_hash}.lock")
        self.handle = None
        self.exclusive = True if IS_WINLOCK else exclusive
        self.timeout = timeout
        self.dev_mode = dev_mode

    def __enter__(self):
        if IS_NOLOCK:
            if not self.dev_mode:
                raise RuntimeError("⛔ FATAL: No lock capability.")
            return self

        start_time = time.time()
        try:
            if os.name != 'nt':
                flags = os.O_RDWR | os.O_CREAT
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                if hasattr(os, "O_CLOEXEC"):
                    flags |= os.O_CLOEXEC
                fd = os.open(self.lock_file, flags, 0o600)
                stf = os.fstat(fd)
                if not stat.S_ISREG(stf.st_mode) or stf.st_uid != os.getuid():
                    os.close(fd)
                    raise RuntimeError("⛔ SECURITY: Lock file violation.")
                self.handle = os.fdopen(fd, "a+b")
            else:
                self.handle = open(self.lock_file, "a+b")
        except Exception as e:
            raise RuntimeError(f"⛔ SECURITY: Lock open fail: {e}")

        while True:
            try:
                if IS_POSIX:
                    import fcntl
                    fcntl.flock(
                        self.handle.fileno(),
                        (fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
                    )
                    break
                elif IS_WINLOCK:
                    import msvcrt
                    if os.fstat(self.handle.fileno()).st_size == 0:
                        self.handle.write(b'\x00')
                        self.handle.flush()
                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
            except (IOError, OSError):
                if time.time() - start_time > self.timeout:
                    raise TimeoutError("Lock timeout.")
                time.sleep(0.05 + random.uniform(0, 0.1))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is None:
            return
        try:
            if IS_POSIX:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            elif IS_WINLOCK:
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self.handle.close()

class ResoneticCrypto:
    EXCLUDE_CANON = {"sig", "hash"}

    def __init__(self):
        if IS_NOLOCK and not Config.DEV_MODE:
            raise RuntimeError("⛔ FATAL: Prod locking disabled.")
        self.secret = None
        with AtomicFileLock(Config.KEY_FILE, exclusive=True, dev_mode=Config.DEV_MODE):
            if os.path.exists(Config.KEY_FILE):
                try:
                    with self.secure_open(Config.KEY_FILE, "rb") as f:
                        self.secret = f.read()
                except Exception as e:
                    raise RuntimeError(f"⛔ SECURITY: Key access fail: {e}")

            # [Fix 3] Production Kill-switch for missing keys
            if not self.secret:
                if Config.DEV_MODE:
                    self.secret = secrets.token_bytes(32)
                    with self.secure_open(Config.KEY_FILE, "wb", exclusive=True) as f:
                        f.write(self.secret)
                        f.flush()
                        os.fsync(f.fileno())
                    self._fsync_parent_dir(Config.KEY_FILE)
                else:
                    raise RuntimeError("⛔ FATAL: Root secret missing (Production).")

    @staticmethod
    def _fsync_parent_dir(file_path):
        dir_path = os.path.dirname(os.path.abspath(file_path)) or "."
        try:
            fd = os.open(dir_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            pass

    @staticmethod
    def secure_open(filepath, mode="rb", exclusive=False):
        if mode not in ("rb", "wb", "ab"):
            raise ValueError(f"⛔ Mode violation: {mode}")

        filepath = os.path.realpath(filepath)
        if os.name != 'nt':
            flags = os.O_RDONLY if mode == "rb" else (
                os.O_WRONLY | os.O_CREAT |
                (os.O_EXCL if exclusive else (os.O_APPEND if mode == "ab" else os.O_TRUNC))
            )
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            fd = os.open(filepath, flags, 0o600)
            stf = os.fstat(fd)
            if not stat.S_ISREG(stf.st_mode) or stf.st_uid != os.getuid():
                os.close(fd)
                raise RuntimeError("⛔ SECURITY: Inode violation.")
            return os.fdopen(fd, mode)
        else:
            f = open(filepath, mode)
            try:
                st_info = os.lstat(filepath)
                attrs = getattr(st_info, "st_file_attributes", 0)
                if stat.S_ISLNK(st_info.st_mode) or (attrs & 0x400):
                    f.close()
                    raise RuntimeError("⛔ SECURITY: Reparse violation.")
                return f
            except Exception:
                f.close()
                raise

    def canonical_str(self, entry_like: dict) -> str:
        def _deep(v, depth=0):
            if depth > 6:
                return "__DEPTH__"
            if v is None:
                return None
            if isinstance(v, (bool, int)):
                return v
            if isinstance(v, float):
                if not (v == v) or v in (float("inf"), float("-inf")):
                    return "__NONFINITE__"
                return round(v, 6)
            if isinstance(v, str):
                return v[:Config.MAX_STR_LEN]
            if isinstance(v, dict):
                items = sorted(list(v.items()), key=lambda x: str(x[0]))[:Config.MAX_DICT_KEYS]
                return {str(k): _deep(val, depth + 1) for k, val in items}
            if isinstance(v, (list, tuple)):
                return [_deep(x, depth + 1) for x in list(v)[:Config.MAX_LIST_ITEMS]]
            return str(v)

        target = (
            Config.SIG_FIELDS if "global_step" in entry_like else
            (Config.AUDIT_SIG_FIELDS if "audit_seq" in entry_like else sorted(entry_like.keys()))
        )
        payload = {k: _deep(entry_like[k]) for k in target if k in entry_like and k not in self.EXCLUDE_CANON}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def verify_entry(self, entry: dict):
        try:
            cstr = self.canonical_str(entry)
            sig = hmac.new(self.secret, cstr.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, entry.get("sig", "")):
                return False, "SIG"
            if "hash" in entry:
                h = hashlib.sha256((cstr + "|sig:" + sig).encode()).hexdigest()
                if h != entry["hash"]:
                    return False, "HASH"
            return True, "OK"
        except Exception as e:
            return False, str(e)

    def commit_transaction(self, filepath, head_path, builder, expected=None):
        with AtomicFileLock(head_path, exclusive=True, dev_mode=Config.DEV_MODE):
            with AtomicFileLock(filepath, exclusive=True, dev_mode=Config.DEV_MODE):
                anchor_data = None
                if os.path.exists(head_path):
                    with self.secure_open(head_path, "rb") as f:
                        parts = f.read().decode().strip().split()
                        # [Fix 2] Strict Hex and Length Validation
                        if len(parts) == 2:
                            h = parts[1].lower()
                            if len(h) == 64 and all(c in "0123456789abcdef" for c in h):
                                anchor_data = (int(parts[0]), h)

                prev_seq, prev_hash = anchor_data if anchor_data else (0, Config.GENESIS_HASH)
                if anchor_data is None and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    raise RuntimeError("SPLIT_BRAIN: History exists without head.")
                if expected and prev_hash != expected:
                    raise RuntimeError("SPLIT_BRAIN: Hash divergence.")

                payload = builder(prev_seq + 1, prev_hash)
                cstr = self.canonical_str(payload)
                payload["sig"] = hmac.new(self.secret, cstr.encode(), hashlib.sha256).hexdigest()
                payload["hash"] = hashlib.sha256((cstr + "|sig:" + payload["sig"]).encode()).hexdigest()

                data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                frame = f"{len(data)}:".encode("ascii") + data + b"\n"
                with self.secure_open(filepath, "ab") as f:
                    f.write(frame)
                    f.flush()
                    os.fsync(f.fileno())
                self._fsync_parent_dir(filepath)

                nonce = secrets.token_hex(4)
                tmp = f"{head_path}.{nonce}.tmp"
                with self.secure_open(tmp, "wb", exclusive=True) as f:
                    f.write(f"{prev_seq + 1} {payload['hash']}\n".encode())
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, head_path)
                self._fsync_parent_dir(head_path)
                return payload

    @staticmethod
    def stream_frames(filepath, lock_read=False, strict=True):
        if not os.path.exists(filepath):
            yield None, "NO_FILE"
            return

        lock_ctx = AtomicFileLock(filepath, exclusive=False, dev_mode=Config.DEV_MODE) if lock_read else nullcontext()
        with lock_ctx:
            with ResoneticCrypto.secure_open(filepath, "rb") as f:
                while True:
                    # skip whitespace/newlines
                    while True:
                        pos = f.tell()
                        ch = f.read(1)
                        if not ch:
                            return
                        if ch not in b"\n\r\t ":
                            f.seek(pos)
                            break

                    len_buf = b""
                    char_retries = 3
                    while True:
                        try:
                            char = f.read(1)
                            if not char:
                                if not len_buf:
                                    return
                                yield None, "MALFORMED_HEADER"
                                return
                            if char == b":":
                                break
                            if not b"0" <= char <= b"9":
                                yield None, "MALFORMED_HEADER"
                                return
                            len_buf += char
                            if len(len_buf) > Config.MAX_HEADER_SIZE:
                                yield None, "OVERFLOW"
                                return
                        except Exception:
                            if char_retries > 0:
                                char_retries -= 1
                                time.sleep(0.01)
                                continue
                            yield None, "IO"
                            return

                    try:
                        length = int(len_buf)
                    except Exception:
                        yield None, "MALFORMED_HEADER"
                        return

                    if length > Config.MAX_FRAME_SIZE:
                        yield None, "SIZE"
                        return

                    payload_bytes = f.read(length)
                    if len(payload_bytes) != length:
                        yield None, "TRUNCATED"
                        return

                    try:
                        yield json.loads(payload_bytes.decode("utf-8")), None
                        # consume trailing whitespace
                        while True:
                            pos = f.tell()
                            ch = f.read(1)
                            if not ch:
                                return
                            if ch not in b"\n\r\t ":
                                f.seek(pos)
                                break
                    except Exception:
                        yield None, "CORRUPT"
                        if strict:
                            return

# ==============================================================================
# 3. Agent Implementation (The Sealed Pulse)
# ==============================================================================
class ResoneticAgent:
    def __init__(self):
        self.proc_state = GLOBAL_STATE  # Constant reference
        ledger_base = os.path.dirname(os.path.abspath(Config.LEDGER_FILE))
        self._instance_lock = AtomicFileLock(
            os.path.join(ledger_base, "app_instance"),
            exclusive=True,
            dev_mode=Config.DEV_MODE,
        )
        try:
            self._instance_lock.__enter__()
            atexit.register(self._release_instance_lock)
            self.crypto = ResoneticCrypto()
        except Exception as e:
            st.error(f"⛔ FATAL: Instance locked. {e}")
            st.stop()

        self._data_lock = threading.RLock()
        self.global_step, self.hash_chain = 0, Config.GENESIS_HASH
        self.physics_step, self.episode_id, self.episode_step = 0, 0, 0
        self.ui_buffer = deque(maxlen=500)
        self._secure_boot()

    def _release_instance_lock(self):
        try:
            self._instance_lock.__exit__(None, None, None)
        except Exception:
            pass

    def _secure_boot(self):
        log_boot("Booting Sealed Kernel...")
        if not os.path.exists(Config.LEDGER_FILE):
            try:
                with ResoneticCrypto.secure_open(Config.LEDGER_FILE, "wb", exclusive=True) as f:
                    f.write(b"")
                ResoneticCrypto._fsync_parent_dir(Config.LEDGER_FILE)
            except Exception:
                pass

        stream = ResoneticCrypto.stream_frames(Config.LEDGER_FILE, lock_read=True, strict=True)
        expected_prev, prev_global, total = Config.GENESIS_HASH, 0, 0
        for entry, error in stream:
            if error:
                log_boot(f"BOOT FAIL: {error}", is_error=True)
                break
            ok, _ = self.crypto.verify_entry(entry)
            if (not ok) or entry["global_step"] != prev_global + 1 or entry["prev_hash"] != expected_prev:
                log_boot(f"INTEGRITY FAIL @ G:{entry.get('global_step','?')}", is_error=True)
                break
            self.ui_buffer.append(entry)
            expected_prev, prev_global, total = entry["hash"], entry["global_step"], total + 1

        if total > 0:
            last = self.ui_buffer[-1]
            self.hash_chain, self.global_step = last["hash"], last["global_step"]
            self.physics_step = last.get("physics_step", 0)
            self.episode_id = last.get("episode", 0)
            self.episode_step = last.get("episode_step", 0)
            log_boot(f"Restored Tip G:{self.global_step}")

    def tick(self, metrics=None, meta=None):
        self._log_event("STEP_COMMIT", "OK", "Sealed pulse recorded.", metrics, meta)

    def _log_event(self, event_type, status, detail, metrics=None, meta=None):
        if event_type not in Config.ALLOWED_EVENTS:
            log_boot(f"POLICY_DENY: {event_type}", is_error=True)
            return
        if self.proc_state.quarantine and event_type not in Config.QUARANTINE_ALLOW:
            return

        with self._data_lock:
            if event_type == "EPISODE_OPEN":
                self.episode_id += 1
                self.episode_step = 0
            else:
                self.episode_step += 1
            self.physics_step += 1

            ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            def build_payload(next_seq, prev_hash):
                return {
                    "prev_hash": prev_hash,
                    "global_step": next_seq,
                    "physics_step": self.physics_step,
                    "episode": self.episode_id,
                    "episode_step": self.episode_step,
                    "type": event_type,
                    "status": status,
                    "msg": detail,
                    "metrics": metrics or {},
                    "meta": meta or {},
                    "time": ts,
                }

            try:
                entry = self.crypto.commit_transaction(
                    Config.LEDGER_FILE,
                    Config.HEAD_FILE,
                    build_payload,
                    expected=self.hash_chain,
                )
                self.global_step, self.hash_chain = entry["global_step"], entry["hash"]
                self.ui_buffer.append(entry)
            except Exception as re:
                log_boot(f"COMMIT FAIL: {re}", is_error=True)
                with self.proc_state._lock:
                    self.proc_state.quarantine = True

    def perform_self_audit(self):
        start_ts = time.time()
        stream = ResoneticCrypto.stream_frames(Config.LEDGER_FILE, lock_read=True, strict=False)
        expected_prev, prev_global, count = Config.GENESIS_HASH, 0, 0
        result, fail_reason = "PASS", None

        try:
            for entry, error in stream:
                if error:
                    result = "FAIL"
                    fail_reason = error
                    break
                ok, _ = self.crypto.verify_entry(entry)
                if (not ok) or entry["global_step"] != prev_global + 1 or entry["prev_hash"] != expected_prev:
                    result = "FAIL"
                    fail_reason = "VERIFY_FAIL"
                    break
                expected_prev, prev_global, count = entry["hash"], entry["global_step"], count + 1
        except Exception as e:
            result = "FAIL"
            fail_reason = str(e)

        duration_ms = int((time.time() - start_ts) * 1000)
        self._log_audit_verdict(
            result,
            fail_reason if result == "FAIL" else f"Verified {count} blocks",
            {"ms": duration_ms, "count": count},
        )
        return (result != "FAIL"), (fail_reason or f"Verified {count} blocks")

    def _log_audit_verdict(self, result, msg, meta):
        def build_payload(next_seq, prev_hash):
            return {
                "audit_seq": next_seq,
                "prev_hash": prev_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "result": result,
                "msg": msg,
                "meta": meta,
            }

        try:
            self.crypto.commit_transaction(Config.AUDIT_FILE, Config.AUDIT_HEAD_FILE, build_payload)
        except Exception as e:
            log_boot(f"JUDICIAL ERROR: {e}", is_error=True)

# ==============================================================================
# 4. Agent Singleton & UI
# ==============================================================================
@st.cache_resource
def get_agent():
    return ResoneticAgent()

agent = get_agent()
proc_state = agent.proc_state

st.sidebar.title("💎 v34.7 Sealed Monolith")
if st.session_state.lock_degraded:
    st.sidebar.warning("🛡️ SECURITY MODE: DEGRADED")

with st.sidebar.expander("🧾 Global Forensic Log", expanded=False):
    for line in list(proc_state.boot_logs)[-20:][::-1]:
        st.caption(line)

if proc_state.quarantine:
    st.sidebar.error("🚨 QUARANTINE: ACTIVE")
    if st.sidebar.button("🔓 Lift Quarantine"):
        ok, msg = agent.perform_self_audit()
        if ok:
            with proc_state._lock:
                proc_state.quarantine = False
            st.rerun()

st.sidebar.write(f"Ledger Tip: G:{agent.global_step} | {agent.hash_chain[:8]}")
if st.sidebar.button("🛡️ Run Global Audit", disabled=proc_state.running):
    ok, msg = agent.perform_self_audit()
    st.sidebar.write(msg)

if st.sidebar.button("Start/Stop Pulse", disabled=proc_state.quarantine):
    with proc_state._lock:
        proc_state.running = not proc_state.running

# Intelligent Global Pulse
if proc_state.running and not proc_state.quarantine:
    now = time.time()
    elapsed = now - proc_state.last_pulse
    if elapsed >= Config.PULSE_INTERVAL:
        with proc_state._lock:
            proc_state.last_pulse = now
        agent.tick()
        st.rerun()
    else:
        time.sleep(max(0.01, Config.PULSE_INTERVAL - elapsed))
        st.rerun()

c1, c2 = st.columns([2, 1])
with c1:
    st.write("### 🌌 Decision Monitoring")
    st.json({"step": agent.global_step, "quarantine": proc_state.quarantine, "running": proc_state.running})

with c2:
    st.write("### 📼 Verified Ledger")
    for log in list(agent.ui_buffer)[-5:][::-1]:
        st.caption(f"{log['type']} | G:{log['global_step']}")
