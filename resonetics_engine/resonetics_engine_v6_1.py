#!/usr/bin/env python3
# ==============================================================================
# File: resonetics_engine_v6_1.py
# Project: Resonetics (Reasoning Engine)
# Version: 6.1 (Performance Edition)
# Author: red1239109-cmd
# Copyright (c) 2025 Resonetics Project
#
# License: AGPL-3.0
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================

from __future__ import annotations

import logging
import hashlib
import asyncio
import sys
import multiprocessing
import math
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

# [Opt 1] Fast JSON Serialization
try:
    import orjson # pip install orjson
except ImportError:
    print("Error: pip install orjson")
    sys.exit(1)

try:
    from ripser import ripser
    from gudhi.bottleneck_distance import bottleneck_distance
except ImportError:
    print("Error: Missing TDA libs")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
log = logging.getLogger("resonetics-v6.1")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ... (Metrics & Config are same) ...
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
    inference_timeout: float = 15.0

# ... (SizedLRUCache same) ...
class SizedLRUCache:
    def __init__(self, max_mb: int):
        self.max_bytes = max_mb * 1024 * 1024
        self.curr_bytes = 0
        self.cache: Dict[str, float] = {}
        self.order: List[str] = []
    def get(self, key: str) -> Optional[float]:
        if key in self.cache:
            self.order.remove(key); self.order.append(key)
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

# ... (TDA Task same) ...
def _compute_topology_task(x_bytes: bytes, y_bytes: bytes, cfg_dict: dict) -> Optional[float]:
    try:
        shape = (cfg_dict['system_dim'],)
        x_np = np.frombuffer(x_bytes, dtype=np.float32).reshape(shape).flatten()
        y_np = np.frombuffer(y_bytes, dtype=np.float32).reshape(shape).flatten()
        window = cfg_dict['tda_window']; stride = cfg_dict['tda_stride']
        
        # [Opt 3] Early Exit for small vectors (Noise filter)
        if x_np.shape[0] < window * 2: return None

        def embed(sig):
            n = sig.shape[0]
            # Vectorized embedding (slightly faster than list comp)
            # But kept simple for safety here
            points = [sig[i : i + window] for i in range(0, n - window + 1, stride)]
            return np.array(points)

        pc_x = embed(x_np); pc_y = embed(y_np)
        if pc_x is None or pc_y is None or pc_x.shape[0] < 5 or pc_y.shape[0] < 5: return None
        
        dgm_x = ripser(pc_x, maxdim=1)["dgms"][1]
        dgm_y = ripser(pc_y, maxdim=1)["dgms"][1]
        
        if dgm_x.size == 0: dgm_x = np.array([[0.0, 0.0]])
        if dgm_y.size == 0: dgm_y = np.array([[0.0, 0.0]])
        return float(bottleneck_distance(dgm_x, dgm_y))
    except: return None

# ... (Hybrid Manager same) ...
class HybridTopoManager:
    def __init__(self, cfg: ResoneticConfig):
        self.cfg = cfg
        ctx = multiprocessing.get_context("spawn")
        self.executor = ProcessPoolExecutor(max_workers=cfg.tda_workers, mp_context=ctx)
        self.cache = SizedLRUCache(cfg.max_cache_mb)

    def _hash(self, x_bytes, y_bytes): return hashlib.md5(x_bytes + y_bytes).hexdigest()

    async def compute_hybrid(self, p_vec: torch.Tensor, c_vec: torch.Tensor) -> Tuple[float, str]:
        p_np = p_vec.cpu().numpy().flatten().astype(np.float32)
        c_np = c_vec.cpu().numpy().flatten().astype(np.float32)
        p_bytes, c_bytes = p_np.tobytes(), c_np.tobytes()
        key = self._hash(p_bytes, c_bytes)
        
        cached = self.cache.get(key)
        if cached is not None: return cached, "CACHE"
        
        loop = asyncio.get_running_loop()
        cfg_lite = {'system_dim': self.cfg.system_dim, 'tda_window': self.cfg.tda_window, 'tda_stride': self.cfg.tda_stride}
        
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
#  Optimized Neural Components
# ------------------------------------------------------------------ #
class SemanticCortex(nn.Module):
    def __init__(self, cfg: ResoneticConfig):
        super().__init__()
        # [Opt 1] GPU Enabled SBERT
        log.info(f"Loading SBERT on {DEVICE}...")
        self.encoder = SentenceTransformer(cfg.sbert_name, device=str(DEVICE))
        self.projector = nn.Linear(self.encoder.get_sentence_embedding_dimension(), cfg.system_dim).to(DEVICE)
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            # SBERT handles batching internally, usually faster on GPU
            emb = self.encoder.encode(texts, convert_to_tensor=True, device=DEVICE, show_progress_bar=False)
        return F.normalize(self.projector(emb), dim=1)

# ... (TruthFlow & ShockDetector same) ...
class TruthFlow(nn.Module):
    def __init__(self, cfg: ResoneticConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("belief", torch.zeros(1, cfg.system_dim))
    def reset(self): self.belief.zero_()
    def forward(self, x):
        if self.belief.sum() == 0: self.belief = x.mean(0, keepdim=True).clone()
        else: self.belief = self.belief * (1 - self.cfg.flow_rate) + x.mean(0, keepdim=True) * self.cfg.flow_rate
        return self.belief.clone()

class ContextAwareShockDetector(nn.Module):
    def __init__(self, cfg: ResoneticConfig):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.system_dim * 3, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(self, p, c, b): return torch.sigmoid(self.net(torch.cat([p, c, b], dim=1))).squeeze(-1)

# ... (Step & Result same) ...
@dataclass
class Step:
    premise: str; conclusion: str; coherence: float; shock: float; method: str; confidence: float
@dataclass
class Result:
    query: str; steps: List[Step]; final_conclusion: Optional[str] = None; stopped: str = "OK"

class Engine:
    def __init__(self):
        self.cfg = ResoneticConfig()
        self.topo = HybridTopoManager(self.cfg)
        self.cortex = SemanticCortex(self.cfg).to(DEVICE)
        self.flow = TruthFlow(self.cfg).to(DEVICE)
        self.shock = ContextAwareShockDetector(self.cfg).to(DEVICE)
        self.phase_scale = nn.Parameter(torch.tensor(5.0)).to(DEVICE)
        log.info(f"Engine Ready on {DEVICE}")

    async def infer_stream(self, query: str, premises: List[str]) -> AsyncGenerator[Step, None]:
        self.flow.reset()
        
        # [Opt 1] Batch Encode initial query
        q_vec = self.cortex([query])
        belief = self.flow(q_vec)

        # [Opt 1] Batch Encode all premises at once (Huge Speedup)
        # Instead of encoding one by one in loop, encode all first
        all_texts = premises
        all_vecs = self.cortex(all_texts) # (N, Dim)

        for i in range(len(premises) - 1):
            pv = all_vecs[i:i+1]
            cv = all_vecs[i+1:i+2]
            
            text_len = len(premises[i]) + len(premises[i+1])
            timeout = 1.0 + (text_len / 500.0)
            
            try:
                raw, method = await asyncio.wait_for(self.topo.compute_hybrid(pv, cv), timeout=timeout)
            except asyncio.TimeoutError:
                raw, method = 2.0, "TIMEOUT"
                FALLBACK_COUNT.inc()

            coherence = float(torch.sigmoid(-self.phase_scale * (raw - 0.5)).item())
            shock_val = self.shock(pv, cv, belief).item()
            belief = self.flow(cv)
            
            base_conf = 1.0 if method == "TDA" or method == "CACHE" else 0.5
            if method == "TIMEOUT": base_conf = 0.1
            confidence = base_conf * coherence
            
            step = Step(premises[i], premises[i+1], coherence, shock_val, method, confidence)
            yield step
            
            if coherence < self.cfg.coherence_min: 
                yield Step("", "STOPPED: Low Coherence", 0, 0, method, 0)
                break
            if shock_val > self.cfg.shock_threshold:
                yield Step("", "STOPPED: High Shock", 0, 0, method, 0)
                break

app = FastAPI(title="Resonetics-v6.1-Perf", version="6.1")
engine: Optional[Engine] = None

class InferRequest(BaseModel):
    query: str
    premises: List[str] = Field(..., max_items=20)
    @validator('premises')
    def validate_len(cls, v):
        if any(len(p) > 1000 for p in v): raise ValueError("Premise too long")
        return v
    class Config:
        schema_extra = {"example": {"query": "Is AI dangerous?", "premises": ["AI learns.", "Data is biased.", "AI is biased."]}}

@app.on_event("startup")
async def startup():
    global engine
    engine = Engine()

@app.get("/metrics")
async def metrics(): return StreamingResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/infer_stream")
async def infer_stream(req: InferRequest):
    INFERENCE_REQUESTS.inc()
    if not engine: raise HTTPException(503, "Loading...")
    
    async def event_generator():
        try:
            async for step in engine.infer_stream(req.query, req.premises):
                # [Opt 2] orjson for super-fast serialization
                # orjson dumps returns bytes, so we decode to str for SSE
                yield orjson.dumps(asdict(step)).decode('utf-8') + "\n"
        except Exception as e:
            INFERENCE_ERRORS.inc()
            yield orjson.dumps({"error": str(e)}).decode('utf-8') + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    # GPU present -> workers=1
    # CPU only -> workers=cores
    w = 1 if torch.cuda.is_available() else 4
