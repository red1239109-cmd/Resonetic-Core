#!/usr/bin/env python3
# ==============================================================================
# File: resonetics_engine_v6_3_colab_ready.py
# Project: Resonetics (Reasoning Engine)
# Version: 6.3 (Colab Ready / Cached Cortex / Safe Defaults)
# License: AGPL-3.0
# ==============================================================================
#
# Notebook-friendly single-file engine:
# - FastAPI/Prometheus 제거 (노트북에서 중복 레지스트리/워커 이슈 방지)
# - TDA(ripser/gudhi) 없어도 동작 (Fallback cosine distance)
# - 텍스트→임베딩 CPU 캐시 포함 (순서 보존)
#
# 사용 예시(코랩/주피터):
#   !pip -q install sentence-transformers
#   from resonetics_engine_v6_3_colab_ready import Engine, ResoneticConfig
#   import asyncio
#   async def demo():
#       eng = Engine(ResoneticConfig(coherence_min=0.05, shock_threshold=0.85))
#       q = "Is AI dangerous?"
#       premises = ["AI learns from data.", "Data can be biased.", "Therefore AI can be biased."]
#       async for step in eng.infer_stream(q, premises):
#           print(step)
#   await demo()
#
# ==============================================================================

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, AsyncGenerator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# [1] Config
# ------------------------------------------------------------------------------
@dataclass
class ResoneticConfig:
    # SBERT settings
    system_dim: int = 384
    sbert_name: str = "all-MiniLM-L6-v2"

    # Flow / Thresholds (Colab-safe defaults)
    flow_rate: float = 0.20
    coherence_min: float = 0.05
    shock_threshold: float = 0.85

    # Coherence mapping strength (lower = less aggressive drop)
    phase_scale: float = 2.0

    # Cache
    cache_max_items: int = 4096


# ------------------------------------------------------------------------------
# [2] Simple CPU Cache (Text → Embedding)
# ------------------------------------------------------------------------------
class EmbeddingCache:
    """
    Minimal LRU-like cache (in-memory) storing normalized float32 vectors on CPU.
    Key is md5(text).
    """
    def __init__(self, max_items: int = 2048):
        self.max_items = int(max_items)
        self.store: Dict[str, np.ndarray] = {}
        self.order: List[str] = []  # oldest -> newest

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._hash(text)
        v = self.store.get(key)
        if v is None:
            return None
        # refresh LRU order
        try:
            self.order.remove(key)
        except ValueError:
            pass
        self.order.append(key)
        return v

    def put(self, text: str, vec: np.ndarray) -> None:
        key = self._hash(text)

        if key in self.store:
            # refresh
            try:
                self.order.remove(key)
            except ValueError:
                pass
        elif len(self.order) >= self.max_items:
            # evict oldest
            old = self.order.pop(0)
            self.store.pop(old, None)

        # store as float32 contiguous array
        self.store[key] = np.asarray(vec, dtype=np.float32)
        self.order.append(key)


# ------------------------------------------------------------------------------
# [3] Semantic Cortex (SBERT + Cache)
# ------------------------------------------------------------------------------
class SemanticCortex(nn.Module):
    def __init__(self, cfg: ResoneticConfig):
        super().__init__()
        print(f"INFO | Loading SBERT '{cfg.sbert_name}' on {DEVICE}")
        self.encoder = SentenceTransformer(cfg.sbert_name, device=str(DEVICE))
        self.cfg = cfg
        self.cache = EmbeddingCache(max_items=cfg.cache_max_items)
        self.eval()

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Returns (N, system_dim) normalized embeddings on DEVICE.
        Cache stores CPU numpy vectors; we preserve the input order.
        """
        if not texts:
            return torch.zeros((0, self.cfg.system_dim), device=DEVICE, dtype=torch.float32)

        # 1) check cache + collect misses (preserve index mapping)
        out_cpu: List[Optional[np.ndarray]] = [None] * len(texts)
        miss_texts: List[str] = []
        miss_indices: List[int] = []

        for i, t in enumerate(texts):
            cached = self.cache.get(t)
            if cached is not None:
                out_cpu[i] = cached
            else:
                miss_texts.append(t)
                miss_indices.append(i)

        # 2) encode misses in batch
        if miss_texts:
            with torch.no_grad():
                miss_vecs = self.encoder.encode(
                    miss_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            for t, idx, v in zip(miss_texts, miss_indices, miss_vecs):
                self.cache.put(t, v)
                out_cpu[idx] = self.cache.get(t)  # guaranteed present

        # 3) stack in original order
        stacked = np.stack([v for v in out_cpu if v is not None], axis=0).astype(np.float32)
        return torch.from_numpy(stacked).to(DEVICE)


# ------------------------------------------------------------------------------
# [4] Flow Memory
# ------------------------------------------------------------------------------
class TruthFlow(nn.Module):
    def __init__(self, cfg: ResoneticConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("belief", torch.zeros(1, cfg.system_dim, dtype=torch.float32))

    def reset(self) -> None:
        self.belief.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D)
        if self.belief.abs().sum().item() == 0.0:
            self.belief = x.mean(0, keepdim=True)
        else:
            self.belief = (
                self.belief * (1.0 - self.cfg.flow_rate)
                + x.mean(0, keepdim=True) * self.cfg.flow_rate
            )
        return self.belief.clone()


# ------------------------------------------------------------------------------
# [5] Fallback Distance (Cosine)
# ------------------------------------------------------------------------------
def fallback_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Distance in [0, 2] from cosine similarity in [-1, 1]:
      dist = (1 - cos) * 2
    """
    sim = float(F.cosine_similarity(a, b).item())
    return (1.0 - sim) * 2.0


# ------------------------------------------------------------------------------
# [6] Data Structures
# ------------------------------------------------------------------------------
@dataclass
class Step:
    premise: str
    conclusion: str
    coherence: float
    shock: float
    method: str
    confidence: float


# ------------------------------------------------------------------------------
# [7] Engine
# ------------------------------------------------------------------------------
class Engine:
    def __init__(self, cfg: Optional[ResoneticConfig] = None):
        self.cfg = cfg or ResoneticConfig()
        self.cortex = SemanticCortex(self.cfg).to(DEVICE)
        self.flow = TruthFlow(self.cfg).to(DEVICE)
        self.phase_scale = float(self.cfg.phase_scale)

        print(
            f"INFO | Engine v6.3 Ready on {DEVICE} "
            f"(dim={self.cfg.system_dim}, cache=ON)"
        )

    async def infer_stream(self, query: str, premises: List[str]) -> AsyncGenerator[Step, None]:
        """
        Streams pairwise reasoning steps over consecutive premises.
        """
        if len(premises) < 2:
            return

        self.flow.reset()

        q_vec = self.cortex.encode([query])
        belief = self.flow(q_vec)
        all_vecs = self.cortex.encode(premises)

        for i in range(len(premises) - 1):
            pv = all_vecs[i : i + 1]
            cv = all_vecs[i + 1 : i + 2]

            raw = fallback_distance(pv, cv)
            method = "FALLBACK"

            # coherence ∈ (0,1): higher when distance is small
            coherence = float(torch.sigmoid(-self.phase_scale * (torch.tensor(raw, device=DEVICE) - 0.5)).item())

            # "shock": how strongly the new conclusion aligns with current belief
            # (you can flip this to 1-|cos| if you want "surprise" instead of "alignment")
            shock_val = float(torch.abs(F.cosine_similarity(cv, belief)).item())

            belief = self.flow(cv)

            # confidence heuristic (simple): coherence scaled
            confidence = float(coherence * (1.0 if method != "FALLBACK" else 0.5))

            yield Step(
                premise=premises[i],
                conclusion=premises[i + 1],
                coherence=coherence,
                shock=shock_val,
                method=method,
                confidence=confidence,
            )

            if coherence < self.cfg.coherence_min:
                yield Step("", "STOPPED: Low Coherence", 0.0, 0.0, method, 0.0)
                break

            if shock_val > self.cfg.shock_threshold:
                yield Step("", "STOPPED: High Shock", 0.0, 0.0, method, 0.0)
                break


# ------------------------------------------------------------------------------
# [8] Demo (Jupyter/Colab safe)
# ------------------------------------------------------------------------------
async def demo() -> None:
    eng = Engine()
    query = "Is AI dangerous?"
    premises = [
        "AI learns from data.",
        "Data can be biased.",
        "Therefore AI can be biased.",
    ]
    async for step in eng.infer_stream(query, premises):
        print(step)


if __name__ == "__main__":
    # Running as a script: python resonetics_engine_v6_3_colab_ready.py
    asyncio.run(demo())
