#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: godel_guardrail_enterprise_v10_2.py  (v10.2)
# Status: Production-Ready (FastAPI + Prometheus + K8s Health + Defense-in-Depth)
#
# v10.2 changes (Option #1):
# - HMAC audit signing + hash chain (tamper-evident)
# - ReplayTrap (simple replay/automation detector)
# - Rule versioning + rule_hash included in audit + responses
# - Audit JSONL file output (encrypted optional)
# ==============================================================================

import asyncio
import hashlib
import hmac
import time
import json
import logging
import re
import math
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
from datetime import datetime

# --- Third-party dependencies ---
# pip install fastapi uvicorn[standard] prometheus-client cryptography pydantic psutil
from cryptography.fernet import Fernet
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

# ==============================================================================
# 0. Infrastructure & Secrets
# ==============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("GodelGuard")

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

# Fernet encryption key (for optional encrypted audit-at-rest)
GODEL_KEY = os.getenv("GODEL_KEY")
if not GODEL_KEY:
    key = Fernet.generate_key()
    logger.warning("🚨 GODEL_KEY missing! Using EPHEMERAL key for this session (encrypted audits won't be decryptable after restart).")
    cipher = Fernet(key)
else:
    try:
        cipher = Fernet(GODEL_KEY.encode())
    except Exception as e:
        logger.fatal(f"Invalid GODEL_KEY: {e}")
        sys.exit(1)

# Audit signing key (HMAC). Prefer stable env var in production.
# NOTE: If this changes, chain integrity verification across restarts breaks.
AUDIT_KEY = os.getenv("GODEL_AUDIT_HMAC_KEY")
if not AUDIT_KEY:
    # EPHEMERAL (works for tamper-evidence during process lifetime only)
    AUDIT_KEY_BYTES = os.urandom(32)
    logger.warning("🚨 GODEL_AUDIT_HMAC_KEY missing! Using EPHEMERAL HMAC key for this session (chain not verifiable across restarts).")
else:
    AUDIT_KEY_BYTES = AUDIT_KEY.encode("utf-8")

AUDIT_FILE = os.getenv("GODEL_AUDIT_FILE", "godel_audit.jsonl")

# ==============================================================================
# 1. Advanced Configuration & Metrics
# ==============================================================================

class GuardrailConfig(BaseModel):
    # Rule versioning
    rule_version: int = Field(default=1, ge=1)
    rule_updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    limits: Dict[str, float] = Field(default_factory=lambda: {"standard": 1.0, "premium": 3.0})
    patterns: List[str] = Field(default_factory=lambda: ["ignore previous", "system prompt"])
    entropy: float = Field(default=5.8, ge=0.0, le=8.0)
    tps_g: float = Field(default=100.0, gt=0.0)
    tps_u: float = Field(default=5.0, gt=0.0)

    # Failure policy
    fail_open: bool = Field(default=False)      # Fail-open on plugin errors

    # Audit policy
    audit_encrypt: bool = Field(default=True)   # Encrypt audit logs at rest
    audit_to_file: bool = Field(default=True)   # Write audit JSONL file
    audit_to_logger: bool = Field(default=True) # Also log via logger

    # ReplayTrap tuning
    replay_window_sec: int = Field(default=30, ge=1, le=3600)
    replay_max_hits: int = Field(default=3, ge=1, le=100)
    replay_cache_per_user: int = Field(default=64, ge=8, le=4096)

    @field_validator("patterns")
    @classmethod
    def validate_regex(cls, v: List[str]) -> List[str]:
        valid: List[str] = []
        for p in v:
            try:
                re.compile(p, re.IGNORECASE)
                valid.append(p)
            except re.error:
                logger.error(f"Invalid regex pattern skipped: {p}")
        return valid

config = GuardrailConfig()

def compute_rule_hash(cfg: GuardrailConfig) -> str:
    # Hash only "policy-driving" fields (deterministic)
    payload = {
        "rule_version": cfg.rule_version,
        "limits": dict(sorted(cfg.limits.items())),
        "patterns": list(cfg.patterns),
        "entropy": cfg.entropy,
        "tps_g": cfg.tps_g,
        "tps_u": cfg.tps_u,
        "fail_open": cfg.fail_open,
        "replay_window_sec": cfg.replay_window_sec,
        "replay_max_hits": cfg.replay_max_hits,
        "replay_cache_per_user": cfg.replay_cache_per_user,
    }
    cstr = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(cstr.encode("utf-8")).hexdigest()

def now_utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

class PrometheusMetrics:
    def __init__(self):
        self.requests = Counter("godel_requests_total", "Total requests", ["status", "plugin", "tier"])
        self.latency = Histogram("godel_latency_seconds", "Request duration", ["plugin"])
        self.active = Gauge("godel_active_requests", "In-flight requests")
        self.debt = Histogram("godel_security_debt", "Accumulated debt score")
        self.replay = Counter("godel_replay_hits_total", "Replay hits", ["tier"])

metrics = PrometheusMetrics()

# ==============================================================================
# 2. Core Security Logic
# ==============================================================================

@dataclass
class SecurityContext:
    request_id: str
    limit: float
    debt: float = 0.0
    rule_version: int = 1
    rule_hash: str = ""

class DefenseInDepth:
    SCRIPT_PATTERNS = [
        r"<script.*?>",
        r"javascript:",
        r"onload\s*=",
        r"onerror\s*=",
    ]

    @staticmethod
    def sanitize(prompt: str) -> Tuple[bool, str]:
        if prompt is None:
            return False, "Prompt is null"
        if len(prompt) > 10000:
            return False, "Input too long"
        for p in DefenseInDepth.SCRIPT_PATTERNS:
            if re.search(p, prompt, re.IGNORECASE):
                return False, "XSS/Script Pattern Detected"
        return True, "OK"

class SecurityPlugin(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def inspect(self, prompt: str, ctx: SecurityContext, user_id: str, tier: str) -> Tuple[bool, str]:
        raise NotImplementedError

class RegexTrap(SecurityPlugin):
    def __init__(self, patterns: List[str]):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def name(self) -> str:
        return "RegexTrap"

    async def inspect(self, prompt: str, ctx: SecurityContext, user_id: str, tier: str) -> Tuple[bool, str]:
        for p in self.patterns:
            if p.search(prompt):
                ctx.debt += 0.5
                if ctx.debt > ctx.limit:
                    return False, "Malicious Pattern"
        return True, "Pass"

class EntropyTrap(SecurityPlugin):
    def name(self) -> str:
        return "EntropyTrap"

    def _entropy(self, text: str) -> float:
        if not text:
            return 0.0
        prob = [text.count(c) / len(text) for c in set(text)]
        return float(-sum(p * math.log(p, 2) for p in prob))

    async def inspect(self, prompt: str, ctx: SecurityContext, user_id: str, tier: str) -> Tuple[bool, str]:
        if len(prompt) < 20:
            return True, "Pass"
        e = self._entropy(prompt)
        if e > config.entropy:
            ctx.debt += 1.0
            if ctx.debt > ctx.limit:
                return False, "High Entropy (Obfuscation)"
        return True, "Pass"

class ReplayTrap(SecurityPlugin):
    """
    Simple replay/automation detector:
    - Normalize prompt -> hash
    - Track (ts list) per user+hash in an LRU-ish map
    - If repeated many times in a short window, raise debt and possibly block
    """
    def __init__(self, window_sec: int, max_hits: int, per_user_cap: int):
        self.window_sec = int(window_sec)
        self.max_hits = int(max_hits)
        self.per_user_cap = int(per_user_cap)
        self._lock = asyncio.Lock()
        # user_id -> OrderedDict[prompt_hash -> list[timestamps]]
        self._store: Dict[str, "OrderedDict[str, List[float]]"] = defaultdict(OrderedDict)

    def name(self) -> str:
        return "ReplayTrap"

    def _normalize(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        # remove long hex-ish blobs (common automation noise)
        s = re.sub(r"\b[0-9a-f]{24,}\b", "<hex>", s)
        return s

    async def inspect(self, prompt: str, ctx: SecurityContext, user_id: str, tier: str) -> Tuple[bool, str]:
        norm = self._normalize(prompt)
        if not norm:
            return True, "Pass"

        ph = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        now = time.time()

        async with self._lock:
            od = self._store[str(user_id)]
            if ph not in od:
                od[ph] = []
            od.move_to_end(ph, last=True)
            od[ph].append(now)

            # prune timestamps outside window
            cutoff = now - self.window_sec
            od[ph] = [t for t in od[ph] if t >= cutoff]
            hits = len(od[ph])

            # enforce per-user cap
            while len(od) > self.per_user_cap:
                od.popitem(last=False)

        if hits >= self.max_hits:
            metrics.replay.labels(tier=tier).inc()
            ctx.debt += 1.0 + min(2.0, 0.25 * hits)
            if ctx.debt > ctx.limit:
                return False, f"Replay/Automation Suspected (hits={hits}/{self.window_sec}s)"
            return True, f"Replay noted (hits={hits})"

        return True, "Pass"

class TokenBucket:
    def __init__(self, rate: float, burst: float):
        self.tokens = float(burst)
        self.capacity = float(burst)
        self.rate = float(rate)
        self.last = time.time()
        self.lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self.lock:
            now = time.time()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

# ==============================================================================
# 3. Audit Chain (HMAC + hash chain)
# ==============================================================================

class AuditChain:
    def __init__(self, audit_file: str):
        self.audit_file = audit_file
        self._lock = asyncio.Lock()
        # In-memory chain head (process-lifetime)
        self._prev_hash = hashlib.sha256(b"genesis").hexdigest()

    def _canonical(self, d: dict) -> str:
        return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)

    def _sign_and_chain(self, data: dict) -> dict:
        # Do NOT include sig/hash in canonical
        payload = dict(data)
        payload["ph"] = self._prev_hash

        cstr = self._canonical(payload)
        sig = hmac.new(AUDIT_KEY_BYTES, cstr.encode("utf-8"), hashlib.sha256).hexdigest()
        h = hashlib.sha256((cstr + "|sig:" + sig).encode("utf-8")).hexdigest()

        payload["sig"] = sig
        payload["hash"] = h
        self._prev_hash = h
        return payload

    async def emit(self, data: dict, encrypt: bool, to_file: bool, to_logger: bool) -> None:
        async with self._lock:
            entry = self._sign_and_chain(data)
            line = self._canonical(entry)

            if encrypt:
                try:
                    enc = cipher.encrypt(line.encode("utf-8")).decode("utf-8")
                    out_line = enc
                    prefix = "AUDIT_ENC:"
                except Exception as e:
                    # If encryption fails, still write plaintext chain (better than losing evidence)
                    logger.warning(f"Audit encrypt failed -> fallback plaintext: {e}")
                    out_line = line
                    prefix = "AUDIT:"
            else:
                out_line = line
                prefix = "AUDIT:"

            if to_logger:
                logger.info(f"{prefix} {out_line}")

            if to_file:
                try:
                    with open(self.audit_file, "a", encoding="utf-8") as f:
                        f.write(out_line + "\n")
                except Exception as e:
                    logger.warning(f"Audit file write failed: {e}")

# ==============================================================================
# 4. Main Engine
# ==============================================================================

class GodelEngine:
    def __init__(self):
        self.global_limiter = TokenBucket(config.tps_g, config.tps_g * 2)
        self.user_limiters: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(config.tps_u, config.tps_u * 2)
        )
        self.plugins: List[SecurityPlugin] = self._build_plugins()
        self.start_time = datetime.now()
        self.running = True

        self.rule_hash = compute_rule_hash(config)
        self.audit = AuditChain(AUDIT_FILE)

    def _build_plugins(self) -> List[SecurityPlugin]:
        return [
            RegexTrap(config.patterns),
            EntropyTrap(),
            ReplayTrap(config.replay_window_sec, config.replay_max_hits, config.replay_cache_per_user),
        ]

    async def _audit_log(self, data: dict) -> None:
        await self.audit.emit(
            data=data,
            encrypt=bool(config.audit_encrypt),
            to_file=bool(config.audit_to_file),
            to_logger=bool(config.audit_to_logger),
        )

    async def _inspect_with_failopen(self, prompt: str, ctx: SecurityContext, user_id: str, tier: str) -> Tuple[bool, str, str]:
        for plugin in self.plugins:
            try:
                safe, reason = await plugin.inspect(prompt, ctx, user_id=user_id, tier=tier)
                if not safe:
                    return False, reason, plugin.name()
            except Exception as e:
                if config.fail_open:
                    logger.warning(f"Plugin {plugin.name()} failed-open: {e}")
                    continue
                raise
        return True, "OK", "none"

    async def scan(self, prompt: str, user_id: str, tier: str) -> Dict[str, Any]:
        if not self.running:
            raise HTTPException(status_code=503, detail="Service Shutting Down")

        metrics.active.inc()
        start = time.perf_counter()
        status = "allowed"
        plugin_name = "none"

        # Make ctx early so it can be audited even on early exits
        ctx = SecurityContext(
            request_id=hashlib.md5(f"{user_id}{time.time()}".encode("utf-8")).hexdigest(),
            limit=float(config.limits.get(tier, 1.0)),
            rule_version=int(config.rule_version),
            rule_hash=str(self.rule_hash),
        )

        try:
            # Layer 0: Defense-in-Depth sanitize
            valid, reason = DefenseInDepth.sanitize(prompt)
            if not valid:
                status, plugin_name = "blocked", "Sanitizer"
                await self._audit_log({
                    "ts": now_utc_iso(),
                    "rid": ctx.request_id,
                    "user": str(user_id),
                    "tier": tier,
                    "rule_version": ctx.rule_version,
                    "rule_hash": ctx.rule_hash,
                    "safe": False,
                    "code": "E400",
                    "reason": reason,
                    "debt": ctx.debt,
                    "plugin": plugin_name,
                })
                return {"safe": False, "code": "E400", "reason": reason, "rule_version": ctx.rule_version, "rule_hash": ctx.rule_hash}

            # Layer 1: Rate limiting
            if not await self.global_limiter.allow():
                status = "throttled"
                raise HTTPException(status_code=429, detail="System Busy")
            if not await self.user_limiters[str(user_id)].allow():
                status = "throttled"
                raise HTTPException(status_code=429, detail="User Rate Limit")

            # Layer 2: Plugins
            safe, reason, plugin_name = await self._inspect_with_failopen(prompt, ctx, user_id=user_id, tier=tier)
            if not safe:
                status = "blocked"
                await self._audit_log({
                    "ts": now_utc_iso(),
                    "rid": ctx.request_id,
                    "user": str(user_id),
                    "tier": tier,
                    "rule_version": ctx.rule_version,
                    "rule_hash": ctx.rule_hash,
                    "safe": False,
                    "code": "E403",
                    "reason": reason,
                    "debt": ctx.debt,
                    "plugin": plugin_name,
                })
                return {"safe": False, "code": "E403", "reason": reason, "rule_version": ctx.rule_version, "rule_hash": ctx.rule_hash}

            metrics.debt.observe(ctx.debt)
            await self._audit_log({
                "ts": now_utc_iso(),
                "rid": ctx.request_id,
                "user": str(user_id),
                "tier": tier,
                "rule_version": ctx.rule_version,
                "rule_hash": ctx.rule_hash,
                "safe": True,
                "code": "S200",
                "reason": "OK",
                "debt": ctx.debt,
                "plugin": plugin_name,
            })
            return {"safe": True, "code": "S200", "reason": "OK", "rule_version": ctx.rule_version, "rule_hash": ctx.rule_hash}

        finally:
            duration = time.perf_counter() - start
            metrics.latency.labels(plugin=plugin_name).observe(duration)
            metrics.requests.labels(status=status, plugin=plugin_name, tier=tier).inc()
            metrics.active.dec()

    async def reload_config(self, new_config: GuardrailConfig) -> None:
        global config
        config = new_config
        self.rule_hash = compute_rule_hash(config)
        self.plugins = self._build_plugins()
        logger.info(f"Configuration hot-reloaded | rule_version={config.rule_version} rule_hash={self.rule_hash}")

# ==============================================================================
# 5. FastAPI App
# ==============================================================================

app = FastAPI(title="Godel Guardrail Enterprise", version="10.2")
engine = GodelEngine()

class ScanRequest(BaseModel):
    prompt: str
    user_id: str
    tier: str = "standard"

class ScanResponse(BaseModel):
    safe: bool
    code: str
    reason: str
    rule_version: int
    rule_hash: str

class ConfigReloadRequest(BaseModel):
    rule_version: int = 1
    rule_updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    limits: Dict[str, float]
    patterns: List[str]
    entropy: float
    tps_g: float
    tps_u: float
    fail_open: bool = False

    audit_encrypt: bool = True
    audit_to_file: bool = True
    audit_to_logger: bool = True

    replay_window_sec: int = 30
    replay_max_hits: int = 3
    replay_cache_per_user: int = 64

@app.on_event("startup")
async def on_startup():
    logger.info("🚀 Godel Guardrail v10.2 Started (AuditChain+ReplayTrap+RuleHash)")

@app.on_event("shutdown")
async def on_shutdown():
    logger.warning("🛑 Shutting down...")
    engine.running = False
    await asyncio.sleep(2)  # drain time (small & practical)

@app.post("/scan", response_model=ScanResponse)
async def scan(req: ScanRequest):
    try:
        return await engine.scan(req.prompt, req.user_id, req.tier)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        raise HTTPException(status_code=503, detail="Internal Guardrail Failure")

@app.post("/reload")
async def reload(req: ConfigReloadRequest):
    """Hot-reload guardrail rules without restart"""
    try:
        new_config = GuardrailConfig(**req.model_dump())
        await engine.reload_config(new_config)
        return {
            "status": "reloaded",
            "rule_version": new_config.rule_version,
            "rule_hash": engine.rule_hash,
            "rule_updated_at": new_config.rule_updated_at,
        }
    except Exception as e:
        logger.exception(f"Reload failed: {e}")
        raise HTTPException(status_code=400, detail="Reload failed")

@app.get("/health")
async def health():
    """K8s Liveness/Readiness Probe"""
    try:
        import psutil
        mem = psutil.Process().memory_info()
        return {
            "status": "healthy" if engine.running else "shutting_down",
            "uptime": str(datetime.now() - engine.start_time),
            "memory_mb": mem.rss / 1024 / 1024,
            "rule_version": config.rule_version,
            "rule_hash": engine.rule_hash,
        }
    except Exception:
        return {
            "status": "healthy" if engine.running else "shutting_down",
            "uptime": str(datetime.now() - engine.start_time),
            "rule_version": config.rule_version,
            "rule_hash": engine.rule_hash,
        }

@app.get("/metrics")
async def get_metrics():
    """Prometheus scrape endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ==============================================================================
# 6. Local Developer Run Mode
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n🛡️ Starting Godel Guardrail Enterprise v10.2...")
    print("👉 Swagger UI: http://localhost:8000/docs")
    print("👉 Metrics:    http://localhost:8000/metrics")
    print("👉 Health:     http://localhost:8000/health")
    print("👉 Hot-Reload: POST /reload (no restart needed)")
    print(f"👉 Audit file: {AUDIT_FILE} (encrypted={config.audit_encrypt}, to_file={config.audit_to_file})")
    uvicorn.run("godel_guardrail_enterprise_v10_2:app", host="0.0.0.0", port=8000, log_level="info")
