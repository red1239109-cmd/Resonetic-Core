#!/usr/bin/env python3
# ==============================================================================
# File: resonetics_engine_v6_2.py
# Project: Resonetics (Reasoning Engine)
# Version: 6.2 (Final Gold - Stability Patched)
# Author: red1239109-cmd
# Copyright (c) 2025 Resonetics Project
# License: AGPL-3.0
# ==============================================================================

from __future__ import annotations

import logging
import hashlib
import asyncio
import sys
import multiprocessing
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, AsyncGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

import numpy as np

# ------------------------------------------------------------------ #
# 0. Dependency & Safety Checks
# ------------------------------------------------------------------ #
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    import json
    HAS_ORJSON = False
    print("Warning: 'orjson' not found. Using standard json (slower).")

try:
    from ripser import ripser
    from gudhi.bottleneck_distance import bottleneck_distance
    HAS_TDA = True
except ImportError:
    print("Warning: TDA libs (ripser/gudhi) missing. TDA features will be disabled.")
    HAS_TDA = False
    ripser = None
    bottleneck_distance = None

# [Fix 1] Force 'spawn' method for PyTorch/CUDA compatibility on Windows/Linux
try:
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# ------------------------------------------------------------------ #
# 1. Infrastructure Setup
# ------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
log = logging.getLogger("resonetics-v6.2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prometheus Metrics
INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total inference requests')
INFERENCE_ERRORS = Counter('inference_errors_total', 'Total inference errors')
TDA_CALC_TIME = Histogram('tda_calc_seconds', 'Time spent in TDA calculation')
CACHE_SIZE_BYTES = Gauge('cache_size_bytes', 'Current size of TDA cache in bytes')
FALLBACK_COUNT = Counter('fallback_count_total', 'Number of times fallback was used')

@dataclass
class ResoneticConfig:
    system_dim: int = 128
    sbert_name: str = "all-MiniLM-L6-v2"
    tda_workers: int = 2
    max_cache_mb: int = 512
    flow_rate: float = 0.15
    coherence_min: float = 0.25
    shock_threshold: float = 0.75
    tda_window: int = 16
    tda_stride: int = 2
    inference_timeout: float = 15.0  # Global timeout cap

# ------------------------------------------------------------------ #
# 2. Memory-Safe Caching
# ------------------------------------------------------------------ #
class SizedLRUCache:
    def __init__(self, max_mb: int):
        self.max_bytes = max_mb * 1024 * 1024
        self.curr_bytes = 0
        self.cache: Dict[str, float] = {}
        self.order: List[str] = []

    def get(self, key: str) -> Optional[float]:
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: float):
        if key in self.cache:
            self.curr_bytes -= sys.getsizeof(key) + sys.getsizeof(self.cache[key])
            self.order.remove(key)
        
        item_size = sys.getsizeof(key) + sys.getsizeof(value)
        while self.curr_bytes + item_size > self.max_bytes and self.order:
            oldest = self.order.pop(0)
            self.curr_bytes -= sys.getsizeof(oldest) + sys.getsizeof(self.cache[oldest])
            del self.cache[oldest]
        
        self.cache[key] = value
        self.order.append(key)
        self.curr_bytes += item_size
        CACHE_SIZE_BYTES.set(self.curr_bytes)

# ------------------------------------------------------------------ #
# 3. Robust TDA Logic (CPU-Bound)
# ------------------------------------------------------------------ #
def _compute_topology_task(x_bytes: bytes, y_bytes: bytes, cfg_dict: dict) -> Optional[float]:
    if not HAS_TDA: return None
    try:
        shape = (cfg_dict['system_dim'],)
        x_np = np.frombuffer(x_bytes, dtype=np.float32).reshape(shape).flatten()
        y_np = np.frombuffer(y_bytes, dtype=np.float32).reshape(shape).flatten()
        window = cfg_dict['tda_window']
        stride = cfg_dict['tda_stride']
        
        if x_np.shape[0] < window * 2: return None

        def embed(sig):
            n = sig.shape[0]
            points = [sig[i : i + window] for i in range(0, n - window + 1, stride)]
            return np.array(points)

        pc_x = embed(x_np)
        pc_y = embed(y_np)
        if pc_x is None or pc_y is None or pc_x.shape[0] < 5 or pc_y.shape[0] < 5: return None
        
        dgm_x = ripser(pc_x, maxdim=1)["dgms"][1]
        dgm_y = ripser(pc_y, maxdim=1)["dgms"][1]
        
        if dgm_x.size == 0: dgm_x = np.array([[0.0, 0.0]])
        if dgm_y.size == 0: dgm_y = np.array([[0.0, 0.0]])
        return float(bottleneck_distance(dgm_x, dgm_y))
    except Exception as e:
        # [Fix 6] Do not swallow exceptions blindly in dev, but return None for safety
        return None

# ------------------------------------------------------------------ #
# 4. Hybrid Topology Manager
# ------------------------------------------------------------------ #
class HybridTopoManager:
    def __init__(self, cfg: ResoneticConfig):
        self.cfg = cfg
        self.executor = ProcessPoolExecutor(max_workers=cfg.tda_workers)
        self.cache = SizedLRUCache(cfg.max_cache_mb)

    # [Fix 2] Graceful Shutdown
    def close(self):
        log.info("Shutting down TDA executor pool...")
        self.executor.shutdown(wait=True, cancel_futures=True)

    # [Fix 5] Cache Key Collision Prevention (Added separator)
    def _hash(self, x_bytes, y_bytes):
        return hashlib.md5(x_bytes + b"|" + y_bytes).hexdigest()

    async def compute_hybrid(self, p_vec: torch.Tensor, c_vec: torch.Tensor) -> Tuple[float, str]:
        p_np = p_vec.cpu().numpy().flatten().astype(np.float32)
        c_np = c_vec.cpu().numpy().flatten().astype(np.float32)
        p_bytes, c_bytes = p_np.tobytes(), c_np.tobytes()
        key = self._hash(p_bytes, c_bytes)
        
        cached = self.cache.get(key)
        if cached is not None: return cached, "CACHE"
        
        loop = asyncio.get_running_loop()
        cfg_lite = {'system_dim': self.cfg.system_dim, 'tda_window': self.cfg.tda_window, 'tda_stride': self.cfg.tda_stride}
        
        tda_dist = None
        if HAS_TDA:
            with TDA_CALC_TIME.time():
                tda_dist = await loop.run_in_executor(self.executor, _compute_topology_task, p_bytes, c_bytes, cfg_lite)
        
        method = "TDA"
        if tda_dist is not None:
            result = tda_dist
        else:
            FALLBACK_COUNT.inc()
            method = "FALLBACK"
            norm_x = np.linalg.norm(p_np); norm_y = np.linalg.norm(c_np)
            cos_sim = np.dot(p_np, c_np) / (norm_x * norm_y + 1e-9)
            result = (1.0 - cos_sim) * 2.0 
        
        self.cache.put(key, result)
        return result, method

# ------------------------------------------------------------------ #
# 5. Optimized Neural Components
# ------------------------------------------------------------------ #
class SemanticCortex(nn.Module):
    def __init__(self, cfg: ResoneticConfig):
        super().__init__()
        log.info(f"Loading SBERT on {DEVICE}...")
        self.encoder = SentenceTransformer(cfg.sbert_name, device=str(DEVICE))
        # [Fix 3] Explicitly set requires_grad=False and eval mode
        self.projector = nn.Linear(self.encoder.get_sentence_embedding_dimension(), cfg.system_dim).to(DEVICE)
        self.projector.weight.requires_grad = False
        self.projector.bias.requires_grad = False
        self.eval() # Lock dropout/batchnorm
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            emb = self.encoder.encode(texts, convert_to_tensor=True, device=DEVICE, show_progress_bar=False)
        return F.normalize(self.projector(
