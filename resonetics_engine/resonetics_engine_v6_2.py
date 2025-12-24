#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: resonetics_engine_v6_2_fixed_single_trunk.py
# Project: Resonetics (Reasoning Engine)
# Version: 6.2 (Practical Gold - Stability/Deploy Patched)
#
# Patch goals (practical, not academic):
#  - Avoid "process inside multi-worker server" footguns
#  - Make caching O(1) with OrderedDict
#  - Fix TruthFlow init logic (no float==0 comparisons)
#  - Make SentenceTransformer device handling single-source-of-truth
#  - Provide safe defaults for GPU/CPU deployment
# ==============================================================================

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, AsyncGenerator, Any
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# ------------------------------------------------------------------ #
# 0. Optional deps
# ------------------------------------------------------------------ #
try:
    import orjson  # type: ignore
    HAS_ORJSON = True
except Exception:
    import json  # type: ignore
    HAS_ORJSON = False

try:
    from ripser import ripser  # type: ignore
    from gudhi.bottleneck_distance import bottleneck_distance  # type: ignore
    HAS_TDA = True
except Exception:
    HAS_TDA = False
    ripser = None
    bottleneck_distance = None


# ------------------------------------------------------------------ #
# 1. Infra
# ------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
log = logging.getLogger("resonetics-v6.2-fixed")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_GPU = torch.cuda.is_available()

# Prometheus metrics
INFERENCE_REQUESTS = Counter("inference_requests_total", "Total inference requests")
INFERENCE_ERRORS = Counter("inference_errors_total", "Total inference errors")
TDA_CALC_TIME = Histogram("tda_calc_seconds", "Time spent in TDA calculation")
CACHE_SIZE_BYTES = Gauge("cache_size_bytes", "Current size of TDA cache in bytes")
FALLBACK_COUNT = Counter("fallback_count_total", "Number of times fallback was used")
TDA_METHOD = Counter("tda_method_total", "TDA methods", ["method"])


@dataclass
class ResoneticConfig:
    system_dim: int = 128
    sbert_name: str = "all-MiniLM-L6-v2"

    # topology
    tda_workers: int = 2
    max_cache_mb: int = 256
    tda_window: int = 16
    tda_stride: int = 2

    # reasoning dynamics
    flow_rate: float = 0.15
    coherence_min: float = 0.25
    shock_threshold: float = 0.75

    # timeouts
    inference_timeout: float = 15.0  # global cap
    base_step_timeout: float = 1.0
    step_timeout_per_chars: float = 1.0 / 500.0

    # executor strategy
    # "thread" is safest inside web servers. "process" only when you know what you're doing.
    topo_executor: str = "thread"  # "thread" | "process" | "none"
    # hard guard: if server has multiple workers, never use process pool here
    forbid_process_in_multiworker: bool = True


# ------------------------------------------------------------------ #
# 2. Fast JSON (NDJSON)
# ------------------------------------------------------------------ #
def dumps_json(obj: Any) -> str:
    if HAS_ORJSON:
        return orjson.dumps(obj).decode("utf-8")  # type: ignore
    return json.dumps(obj, ensure_ascii=False)  # type: ignore


# ------------------------------------------------------------------ #
# 3. Sized LRU Cache (O(1) via OrderedDict)
# ------------------------------------------------------------------ #
class SizedLRUCache:
    def __init__(self, max_mb: int):
        self.max_bytes = int(max_mb) * 1024 * 1024
        self.curr_bytes = 0
        self.cache: "OrderedDict[str, float]" = OrderedDict()

    def _item_size(self, key: str, val: float) -> int:
        # rough sizing: key + float object
        return sys.getsizeof(key) + sys.getsizeof(val)

    def get(self, key: str) -> Optional[float]:
        v = self.cache.get(key)
        if v is None:
            return None
        # mark as most-recent
        self.cache.move_to_end(key, last=True)
        return v

    def put(self, key: str, value: float) -> None:
        # if exists, adjust size, then move-to-end
        if key in self.cache:
            old = self.cache[key]
            self.curr_bytes -= self._item_size(key, old)
            self.cache.move_to_end(key, last=True)

        item_size = self._item_size(key, value)

        # evict until fits
        while self.curr_bytes + item_size > self.max_bytes and len(self.cache) > 0:
            k_old, v_old = self.cache.popitem(last=False)  # oldest
            self.curr_bytes -= self._item_size(k_old, v_old)

        # if still doesn't fit (max_bytes too small), drop it
        if item_size > self.max_bytes:
            return

        self.cache[key] = value
        self.curr_bytes += item_size
        CACHE_SIZE_BYTES.set(self.curr_bytes)


# ------------------------------------------------------------------ #
# 4. TDA compute (CPU-bound)
# ------------------------------------------------------------------ #
def _compute_topology_task(x_bytes: bytes, y_bytes: bytes, cfg_dict: dict) -> Optional[float]:
    """
    Runs in worker thread/process. Must be top-level for process pickling.
    """
    if not HAS_TDA:
        return None
    try:
        dim = int(cfg_dict["system_dim"])
        window = int(cfg_dict["tda_window"])
        stride = int(cfg_dict["tda_stride"])

        x_np = np.frombuffer(x_bytes, dtype=np.float32).reshape((dim,)).flatten()
        y_np = np.frombuffer(y_bytes, dtype=np.float32).reshape((dim,)).flatten()

        if x_np.shape[0] < window * 2 or y_np.shape[0] < window * 2:
            return None

        def embed(sig: np.ndarray) -> np.ndarray:
            n = sig.shape[0]
            pts = [sig[i : i + window] for i in range(0, n - window + 1, stride)]
            return np.asarray(pts, dtype=np.float32)

        pc_x = embed(x_np)
        pc_y = embed(y_np)
        if pc_x.shape[0] < 5 or pc_y.shape[0] < 5:
            return None

        dgm_x = ripser(pc_x, maxdim=1)["dgms"][1]
        dgm_y = ripser(pc_y, maxdim=1)["dgms"][1]
        if dgm_x.size == 0:
            dgm_x = np.array([[0.0, 0.0]], dtype=np.float32)
        if dgm_y.size == 0:
            dgm_y = np.array([[0.0, 0.0]], dtype=np.float32)

        return float(bottleneck_distance(dgm_x, dgm_y))
    except Exception:
        return None


# ------------------------------------------------------------------ #
# 5. Hybrid Topology Manager (safe for web servers)
# ------------------------------------------------------------------ #
class HybridTopoManager:
    def __init__(self, cfg: ResoneticConfig):
        self.cfg = cfg
        self.cache = SizedLRUCache(cfg.max_cache_mb)

        self._executor = None
        self._executor_kind = "none"

        # safe default: threads (no fork/spawn drama inside uvicorn/gunicorn)
        kind = (cfg.topo_executor or "thread").lower().strip()
        if kind not in ("thread", "process", "none"):
            kind = "thread"

        # hard safety: if multi-worker server, forbid process pool here
        if cfg.forbid_process_in_multiworker and _env_multiworker():
            if kind == "process":
                log.warning("Multi-worker detected; forcing topo_executor=thread (process pool disabled).")
                kind = "thread"

        if kind == "thread":
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=max(1, int(cfg.tda_workers)))
            self._executor_kind = "thread"
        elif kind == "process":
            from concurrent.futures import ProcessPoolExecutor
            # NOTE: do this only with workers=1 at server level
            self._executor = ProcessPoolExecutor(max_workers=max(1, int(cfg.tda_workers)))
            self._executor_kind = "process"
        else:
            self._executor_kind = "none"

    def close(self):
        if self._executor is not None:
            log.info(f"Shutting down topo executor ({self._executor_kind})...")
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _hash(self, x_bytes: bytes, y_bytes: bytes) -> str:
        # separator prevents accidental collisions
        return hashlib.md5(x_bytes + b"|" + y_bytes).hexdigest()

    async def compute_hybrid(self, p_vec: torch.Tensor, c_vec: torch.Tensor) -> Tuple[float, str]:
        # Always compute on CPU bytes for caching stability
        p_np = p_vec.detach().to("cpu").numpy().flatten().astype(np.float32)
        c_np = c_vec.detach().to("cpu").numpy().flatten().astype(np.float32)
        p_bytes = p_np.tobytes()
        c_bytes = c_np.tobytes()
        key = self._hash(p_bytes, c_bytes)

        cached = self.cache.get(key)
        if cached is not None:
            TDA_METHOD.labels(method="CACHE").inc()
            return cached, "CACHE"

        # If no executor or no TDA libs, fallback
        if self._executor_kind == "none" or (not HAS_TDA):
            FALLBACK_COUNT.inc()
            TDA_METHOD.labels(method="FALLBACK").inc()
            result = _fallback_distance(p_np, c_np)
            self.cache.put(key, result)
            return result, "FALLBACK"

        loop = asyncio.get_running_loop()
        cfg_lite = {
            "system_dim": self.cfg.system_dim,
            "tda_window": self.cfg.tda_window,
            "tda_stride": self.cfg.tda_stride,
        }

        tda_dist: Optional[float] = None
        try:
            with TDA_CALC_TIME.time():
                tda_dist = await loop.run_in_executor(self._executor, _compute_topology_task, p_bytes, c_bytes, cfg_lite)
        except Exception:
            tda_dist = None

        if tda_dist is None:
            FALLBACK_COUNT.inc()
            TDA_METHOD.labels(method="FALLBACK").inc()
            result = _fallback_distance(p_np, c_np)
            self.cache.put(key, result)
            return result, "FALLBACK"

        TDA_METHOD.labels(method="TDA").inc()
        self.cache.put(key, float(tda_dist))
        return float(tda_dist), "TDA"


def _fallback_distance(x: np.ndarray, y: np.ndarray) -> float:
    norm_x = float(np.linalg.norm(x))
    norm_y = float(np.linalg.norm(y))
    cos_sim = float(np.dot(x, y) / (norm_x * norm_y + 1e-9))
    return (1.0 - cos_sim) * 2.0


def _env_multiworker() -> bool:
    """
    Rough detection: if user configured multiple server workers, avoid process pool inside.
    We can’t perfectly know uvicorn/gunicorn config here, but we can detect common envs.
    """
    # gunicorn
    if os.getenv("GUNICORN_CMD_ARGS"):
        args = os.getenv("GUNICORN_CMD_ARGS", "")
        if "--workers" in args:
            return True
    # uvicorn (common)
    if os.getenv("UVICORN_WORKERS"):
        try:
            return int(os.getenv("UVICORN_WORKERS", "1")) > 1
        except Exception:
            return True
    return False


# ------------------------------------------------------------------ #
# 6. Neural components
# ------------------------------------------------------------------ #
class SemanticCortex(nn.Module):
    """
    Single-source-of-truth device:
    - SentenceTransformer is created with explicit device.
    - We do NOT rely on nn.Module.to() to move it.
    """
    def __init__(self, cfg: ResoneticConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        log.info(f"Loading SBERT '{cfg.sbert_name}' on {self.device} ...")
        self.encoder = SentenceTransformer(cfg.sbert_name, device=str(self.device))

        emb_dim = int(self.encoder.get_sentence_embedding_dimension())
        self.projector = nn.Linear(emb_dim, cfg.system_dim)
        self.projector.requires_grad_(False)
        self.projector.to(self.device)
        self.eval()

    def forward(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            emb = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                device=str(self.device),
                show_progress_bar=False,
            )
            out = self.projector(emb)
            return F.normalize(out, dim=1)


class TruthFlow(nn.Module):
    def __init__(self, cfg: ResoneticConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.register_buffer("belief", torch.zeros(1, cfg.system_dim, device=self.device))
        self._initialized = False
        self.eval()

    def reset(self) -> None:
        with torch.no_grad():
            self.belief.zero_()
        self._initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            m = x.mean(0, keepdim=True)
            if not self._initialized:
                self.belief = m.clone()
                self._initialized = True
            else:
                self.belief = self.belief * (1.0 - self.cfg.flow_rate) + m * self.cfg.flow_rate
            return self.belief.clone()


class ContextAwareShockDetector(nn.Module):
    def __init__(self, cfg: ResoneticConfig, device: torch.device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.system_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.to(device)
        self.eval()

    def forward(self, p: torch.Tensor, c: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = torch.cat([p, c, b], dim=1)
            return torch.sigmoid(self.net(z)).squeeze(-1)


# ------------------------------------------------------------------ #
# 7. Engine
# ------------------------------------------------------------------ #
@dataclass
class Step:
    premise: str
    conclusion: str
    coherence: float
    shock: float
    method: str
    confidence: float


class Engine:
    def __init__(self, cfg: Optional[ResoneticConfig] = None):
        self.cfg = cfg or ResoneticConfig()

        # Safety: if GPU, strongly discourage multi-worker
        if IS_GPU and _env_multiworker():
            log.warning("GPU + multi-worker detected. Strongly recommend server workers=1 to avoid VRAM duplication.")

        self.topo = HybridTopoManager(self.cfg)
        self.cortex = SemanticCortex(self.cfg, device=DEVICE)
        self.flow = TruthFlow(self.cfg, device=DEVICE)
        self.shock = ContextAwareShockDetector(self.cfg, device=DEVICE)

        # phase scale as a parameter (inference-only here)
        self.phase_scale = nn.Parameter(torch.tensor(5.0, device=DEVICE), requires_grad=False)

        log.info(f"Engine ready on {DEVICE} (TDA={HAS_TDA}, executor={self.topo._executor_kind})")

    async def infer_stream(self, query: str, premises: List[str]) -> AsyncGenerator[Step, None]:
        self.flow.reset()

        q_vec = self.cortex([query])
        belief = self.flow(q_vec)
        all_vecs = self.cortex(premises)

        for i in range(len(premises) - 1):
            pv = all_vecs[i : i + 1]
            cv = all_vecs[i + 1 : i + 2]

            text_len = len(premises[i]) + len(premises[i + 1])
            step_timeout = float(self.cfg.base_step_timeout + (text_len * self.cfg.step_timeout_per_chars))

            try:
                raw, method = await asyncio.wait_for(self.topo.compute_hybrid(pv, cv), timeout=step_timeout)
            except asyncio.TimeoutError:
                FALLBACK_COUNT.inc()
                raw, method = 2.0, "TIMEOUT"
            except Exception:
                FALLBACK_COUNT.inc()
                raw, method = 2.0, "ERROR"

            # coherence mapping
            coherence = float(torch.sigmoid(-self.phase_scale * (torch.tensor(raw, device=DEVICE) - 0.5)).item())
            shock_val = float(self.shock(pv, cv, belief).item())

            # belief update (you can gate it philosophically here if you want)
            belief = self.flow(cv)

            base_conf = 1.0 if method in ("TDA", "CACHE") else 0.5
            if method in ("TIMEOUT", "ERROR"):
                base_conf = 0.1
            confidence = float(base_conf * coherence)

            yield Step(
                premise=premises[i],
                conclusion=premises[i + 1],
                coherence=coherence,
                shock=shock_val,
                method=method,
                confidence=confidence,
            )

            # Via Negativa-ish: stop is a result, not just a crash
            if coherence < self.cfg.coherence_min:
                yield Step("", "STOPPED: Low Coherence (Via Negativa)", 0.0, 0.0, method, 0.0)
                break
            if shock_val > self.cfg.shock_threshold:
                yield Step("", "STOPPED: High Shock (Refusal)", 0.0, 0.0, method, 0.0)
                break

    def close(self) -> None:
        self.topo.close()


# ------------------------------------------------------------------ #
# 8. FastAPI app
# ------------------------------------------------------------------ #
app = FastAPI(title="Resonetics Engine", version="6.2-Practical-Gold")
engine: Optional[Engine] = None


class InferRequest(BaseModel):
    query: str
    premises: List[str] = Field(..., max_items=20)

    @validator("premises")
    def validate_premises(cls, v):
        if any(len(p) > 1000 for p in v):
            raise ValueError("Premise too long")
        if len(v) < 2:
            raise ValueError("Need at least 2 premises")
        return v


@app.on_event("startup")
async def startup():
    global engine
    cfg = ResoneticConfig()

    # Auto-safety defaults:
    # - GPU: thread executor, workers=1 recommended
    # - CPU: thread is still safest; you can set topo_executor=process only with server workers=1
    if IS_GPU:
        cfg.topo_executor = "thread"

    engine = Engine(cfg)

    # warmup
    try:
        _ = engine.cortex(["Warm up"])
    except Exception:
        pass

    log.info("Startup complete.")


@app.on_event("shutdown")
async def shutdown():
    global engine
    if engine:
        engine.close()
    log.info("Shutdown complete.")


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "tda": bool(HAS_TDA)}


@app.get("/metrics")
async def metrics():
    return StreamingResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/infer_stream")
async def infer_stream(req: InferRequest):
    INFERENCE_REQUESTS.inc()
    if not engine:
        raise HTTPException(503, "Loading...")

    async def event_generator():
        try:
            async with asyncio.timeout(engine.cfg.inference_timeout):
                async for step in engine.infer_stream(req.query, req.premises):
                    yield dumps_json(asdict(step)) + "\n"
        except asyncio.TimeoutError:
            INFERENCE_ERRORS.inc()
            yield dumps_json({"error": "Global Timeout Reached"}) + "\n"
        except Exception as e:
            INFERENCE_ERRORS.inc()
            log.exception("Inference failed")
            yield dumps_json({"error": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


# ------------------------------------------------------------------ #
# 9. Local run
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import uvicorn

    # Practical worker rule:
    # - GPU: workers=1 (VRAM duplication + executor issues)
    # - CPU: you *can* increase workers, but then do NOT use process pool inside.
    w = 1 if IS_GPU else int(os.getenv("UVICORN_WORKERS", "1"))
    log.info(f"Starting server (workers={w}, device={DEVICE}, topo_executor=thread-safe default)...")
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=w, log_level="info")
