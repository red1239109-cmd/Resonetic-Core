#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: godel_guardrail_enterprise_v10_1.py  (v10.1)
# Status: Production-Ready (FastAPI + Prometheus + K8s Health + Defense-in-Depth)
#
# Key fixes vs your pasted draft:
# - Fixed syntax error in _inspect_with_failopen() signature
# - Pydantic v2 compatible validators (field_validator)
# - Entrypoint alignment: module exposes `app = FastAPI(...)`
# - /health present for Docker/K8s healthcheck
# ==============================================================================

import asyncio
import hashlib
import time
import json
import logging
import re
import math
import os
import sys
from typing import Dict, List, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
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

GODEL_KEY = os.getenv("GODEL_KEY")
if not GODEL_KEY:
    key = Fernet.generate_key()
    logger.warning("🚨 GODEL_KEY missing! Using ephemeral key for this session.")
    cipher = Fernet(key)
else:
    try:
        cipher = Fernet(GODEL_KEY.encode())
    except Exception as e:
        logger.fatal(f"Invalid GODEL_KEY: {e}")
        sys.exit(1)

# ==============================================================================
# 1. Advanced Configuration & Metrics
# ==============================================================================

class GuardrailConfig(BaseModel):
    limits: Dict[str, float] = Field(default_factory=lambda: {"standard": 1.0, "premium": 3.0})
    patterns: List[str] = Field(default_factory=lambda: ["ignore previous", "system prompt"])
    entropy: float = Field(default=5.8, ge=0.0, le=8.0)
    tps_g: float = Field(default=100.0, gt=0.0)
    tps_u: float = Field(default=5.0, gt=0.0)
    fail_open: bool = Field(default=False)      # Fail-open on plugin errors
    audit_encrypt: bool = Field(default=True)   # Encrypt audit logs

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

class PrometheusMetrics:
    def __init__(self):
        self.requests = Counter("godel_requests_total", "Total requests", ["status", "plugin", "tier"])
        self.latency = Histogram("godel_latency_seconds", "Request duration", ["plugin"])
        self.active = Gauge("godel_active_requests", "In-flight requests")
        self.debt = Histogram("godel_security_debt", "Accumulated debt score")

metrics = PrometheusMetrics()

# ==============================================================================
# 2. Core Security Logic
# ==============================================================================

@dataclass
class SecurityContext:
    request_id: str
    limit: float
    debt: float = 0.0

class DefenseInDepth:
    SCRIPT_PATTERNS = [r"<script.*?>", r"javascript:", r"onload\s*=", r"onerror\s*="]

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
    async def inspect(self, prompt: str, ctx: SecurityContext) -> Tuple[bool, str]:
        raise NotImplementedError

class RegexTrap(SecurityPlugin):
    def __init__(self, patterns: List[str]):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def name(self) -> str:
        return "RegexTrap"

    async def inspect(self, prompt: str, ctx: SecurityContext) -> Tuple[bool, str]:
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

    async def inspect(self, prompt: str, ctx: SecurityContext) -> Tuple[bool, str]:
        if len(prompt) < 20:
            return True, "Pass"
        e = self._entropy(prompt)
        if e > config.entropy:
            ctx.debt += 1.0
            if ctx.debt > ctx.limit:
                return False, "High Entropy (Obfuscation)"
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
# 3. Main Engine
# ==============================================================================

class GodelEngine:
    def __init__(self):
        self.global_limiter = TokenBucket(config.tps_g, config.tps_g * 2)
        self.user_limiters: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(config.tps_u, config.tps_u * 2)
        )
        self.plugins: List[SecurityPlugin] = [RegexTrap(config.patterns), EntropyTrap()]
        self.start_time = datetime.now()
        self.running = True

    def _audit_log(self, data: dict) -> None:
        if config.audit_encrypt:
            try:
                payload = json.dumps(data, ensure_ascii=False, default=str)
                enc = cipher.encrypt(payload.encode()).decode()
                logger.info(f"AUDIT_ENC: {enc}")
            except Exception as e:
                logger.warning(f"Audit encrypt failed: {e}")
        else:
            logger.info(f"AUDIT: {data}")

    async def _inspect_with_failopen(self, prompt: str, ctx: SecurityContext) -> Tuple[bool, str, str]:
        for plugin in self.plugins:
            try:
                safe, reason = await plugin.inspect(prompt, ctx)
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

        try:
            # Layer 0: Defense-in-Depth sanitize
            valid, reason = DefenseInDepth.sanitize(prompt)
            if not valid:
                status, plugin_name = "blocked", "Sanitizer"
                return {"safe": False, "code": "E400", "reason": reason}

            # Layer 1: Rate limiting
            if not await self.global_limiter.allow():
                status = "throttled"
                raise HTTPException(status_code=429, detail="System Busy")
            if not await self.user_limiters[str(user_id)].allow():
                status = "throttled"
                raise HTTPException(status_code=429, detail="User Rate Limit")

            # Layer 2: Plugins
            ctx = SecurityContext(
                request_id=hashlib.md5(f"{user_id}{time.time()}".encode()).hexdigest(),
                limit=float(config.limits.get(tier, 1.0)),
            )

            safe, reason, plugin_name = await self._inspect_with_failopen(prompt, ctx)
            if not safe:
                status = "blocked"
                return {"safe": False, "code": "E403", "reason": reason}

            metrics.debt.observe(ctx.debt)
            self._audit_log({"user": user_id, "safe": True, "debt": ctx.debt})
            return {"safe": True, "code": "S200", "reason": "OK"}

        finally:
            duration = time.perf_counter() - start
            metrics.latency.labels(plugin=plugin_name).observe(duration)
            metrics.requests.labels(status=status, plugin=plugin_name, tier=tier).inc()
            metrics.active.dec()

    async def reload_config(self, new_config: GuardrailConfig) -> None:
        global config
        config = new_config
        self.plugins = [RegexTrap(config.patterns), EntropyTrap()]
        logger.info("Configuration hot-reloaded")

# ==============================================================================
# 4. FastAPI App
# ==============================================================================

app = FastAPI(title="Godel Guardrail Enterprise", version="10.1")
engine = GodelEngine()

class ScanRequest(BaseModel):
    prompt: str
    user_id: str
    tier: str = "standard"

class ScanResponse(BaseModel):
    safe: bool
    code: str
    reason: str

class ConfigReloadRequest(BaseModel):
    limits: Dict[str, float]
    patterns: List[str]
    entropy: float
    tps_g: float
    tps_u: float
    fail_open: bool = False
    audit_encrypt: bool = True

@app.on_event("startup")
async def on_startup():
    logger.info("🚀 Godel Guardrail v10.1 Started (Hot-Reload + Fail-Open + EncAudit)")

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
        return {"status": "reloaded"}
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
        }
    except Exception:
        return {
            "status": "healthy" if engine.running else "shutting_down",
            "uptime": str(datetime.now() - engine.start_time),
        }

@app.get("/metrics")
async def get_metrics():
    """Prometheus scrape endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ==============================================================================
# 5. Local Developer Run Mode
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n🛡️ Starting Godel Guardrail Enterprise v10.1...")
    print("👉 Swagger UI: http://localhost:8000/docs")
    print("👉 Metrics:    http://localhost:8000/metrics")
    print("👉 Health:     http://localhost:8000/health")
    print("👉 Hot-Reload: POST /reload (no restart needed)")
    uvicorn.run("godel_guardrail_enterprise_v10_1:app", host="0.0.0.0", port=8000, log_level="info")
