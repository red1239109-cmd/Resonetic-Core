#!/usr/bin/env python3
# ==============================================================================
# File: resonetics_engine_v6_3_singlefile.py
# Project: Resonetics (Reasoning Engine)
# Version: 6.3 (Colab/Jupyter Single-File + Cached Cortex + Safe Defaults)
# License: AGPL-3.0
#
# ✅ "통파일"로 복붙해서 바로 실행 가능
# - FastAPI/Prometheus 제거
# - TDA(ripser/gudhi) 선택사항 (없어도 fallback)
# - SBERT 임베딩 캐시 포함
# ==============================================================================

import asyncio
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Dict, AsyncGenerator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError(
        "sentence-transformers 가 필요합니다. Colab에서 먼저 실행하세요:\n"
        "!pip -q install sentence-transformers\n"
        f"(원인: {e})"
    )

# -----------------------------
# Optional TDA (선택)
# -----------------------------
HAS_TDA = False
try:
    from ripser import ripser
    from gudhi.bottleneck_distance import bottleneck_distance
    HAS_TDA = True
except Exception:
    HAS_TDA = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# [1] Config
# ==============================================================================
@dataclass
class ResoneticConfig:
    system_dim: int = 384
    sbert_name: str = "all-MiniLM-L6-v2"

    # Flow / Thresholds (노트북용 안전 기본값)
    flow_rate: float = 0.2
    coherence_min: float = 0.05
    shock_threshold: float = 0.95
    phase_scale: float = 2.0

    # Cache
    cache_max_items: int = 4096


# ==============================================================================
# [2] Simple CPU LRU Cache (Text → Embedding ndarray)
# ==============================================================================
class EmbeddingCache:
    def __init__(self, max_items: int = 4096):
        self.max_items = int(max_items)
        self.store: Dict[str, np.ndarray] = {}
        self.order: List[str] = []  # LRU (oldest first)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        k = self._key(text)
        v = self.store.get(k)
        if v is None:
            return None
        # LRU update
        try:
            self.order.remove(k)
        except ValueError:
            pass
        self.order.append(k)
        return v

    def put(self, text: str, vec: np.ndarray) -> None:
        k = self._key(text)

        if k in self.store:
            try:
                self.order.remove(k)
            except ValueError:
                pass
        elif len(self.order) >= self.max_items:
            old = self.order.pop(0)
            self.store.pop(old, None)

        self.store[k] = np.asarray(vec, dtype=np.float32)
        self.order.append(k)


# ==============================================================================
# [3] Semantic Cortex (SBERT + Cache)
# ==============================================================================
class SemanticCortex(nn.Module):
    def __init__(self, cfg: ResoneticConfig):
        super().__init__()
        self.cfg = cfg
        self.cache = EmbeddingCache(cfg.cache_max_items)

        print(f"INFO | Loading SBERT '{cfg.sbert_name}' on {DEVICE} ...")
        self.encoder = SentenceTransformer(cfg.sbert_name, device=str(DEVICE))
        self.eval()

    def encode(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.cfg.system_dim, device=DEVICE)

        out: List[Optional[torch.Tensor]] = [None] * len(texts)
        missing: List[str] = []
        missing_pos: List[int] = []

        for i, t in enumerate(texts):
            v = self.cache.get(t)
            if v is None:
                missing.append(t)
                missing_pos.append(i)
            else:
                out[i] = torch.from_numpy(v)

        if missing:
            with torch.no_grad():
                emb = self.encoder.encode(
                    missing,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            for t, v in zip(missing, emb):
                self.cache.put(t, v)

            for pos in missing_pos:
                out[pos] = torch.from_numpy(self.cache.get(texts[pos]))

        stacked = torch.stack([t for t in out if t is not None], dim=0)
        return stacked.to(DEVICE)


# ==============================================================================
# [4] Flow Memory (belief state)
# ==============================================================================
class TruthFlow(nn.Module):
    def __init__(self, cfg: ResoneticConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("belief", torch.zeros(1, cfg.system_dim))

    def reset(self) -> None:
        self.belief.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_mean = x.mean(0, keepdim=True)

        if float(self.belief.abs().sum().item()) == 0.0:
            self.belief = x_mean
        else:
            self.belief = self.belief * (1 - self.cfg.flow_rate) + x_mean * self.cfg.flow_rate
        return self.belief.clone()


# ==============================================================================
# [5] Distance Methods
# ==============================================================================
def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = float(F.cosine_similarity(a, b).item())
    return (1.0 - sim) * 2.0  # [0..2]


def tda_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    if not HAS_TDA:
        return cosine_distance(a, b)

    av = a.squeeze(0).detach().float().cpu().numpy()
    bv = b.squeeze(0).detach().float().cpu().numpy()

    pc_a = np.stack([av, np.arange(av.shape[0], dtype=np.float32)], axis=1)
    pc_b = np.stack([bv, np.arange(bv.shape[0], dtype=np.float32)], axis=1)

    dg_a = ripser(pc_a, maxdim=1)["dgms"][1]
    dg_b = ripser(pc_b, maxdim=1)["dgms"][1]

    try:
        d = float(bottleneck_distance(dg_a, dg_b))
    except Exception:
        d = cosine_distance(a, b)

    return min(2.0, d)


# ==============================================================================
# [6] Output Structure
# ==============================================================================
@dataclass
class Step:
    premise: str
    conclusion: str
    coherence: float
    shock: float
    method: str
    confidence: float


# ==============================================================================
# [7] Engine
# ==============================================================================
class Engine:
    """
    ✅ 사용법:
    eng = Engine()
    async for step in eng.infer_stream(query, premises):
        print(step)
    """
    def __init__(self, cfg: Optional[ResoneticConfig] = None):
        self.cfg = cfg or ResoneticConfig()
        self.cortex = SemanticCortex(self.cfg).to(DEVICE)
        self.flow = TruthFlow(self.cfg).to(DEVICE)

        print(
            f"INFO | Engine v6.3 Ready on {DEVICE} "
            f"(dim={self.cfg.system_dim}, cache=ON, TDA={HAS_TDA})"
        )

    def close(self) -> None:
        pass

    async def infer_stream(self, query: str, premises: List[str]) -> AsyncGenerator[Step, None]:
        if len(premises) < 2:
            yield Step("", "STOPPED: Need at least 2 premises", 0.0, 0.0, "NONE", 0.0)
            return

        self.flow.reset()

        q_vec = self.cortex.encode([query])      # (1,D)
        belief = self.flow(q_vec)               # (1,D)
        all_vecs = self.cortex.encode(premises) # (N,D)

        for i in range(len(premises) - 1):
            pv = all_vecs[i:i+1]
            cv = all_vecs[i+1:i+2]

            if HAS_TDA:
                raw = tda_distance(pv, cv)
                method = "TDA"
                base_conf = 1.0
            else:
                raw = cosine_distance(pv, cv)
                method = "FALLBACK"
                base_conf = 0.5

            coherence = float(torch.sigmoid(-self.cfg.phase_scale * (torch.tensor(raw) - 0.5)).item())

            # shock: 0(안정)~1(충격)로 직관화
            shock_val = float((1.0 - F.cosine_similarity(cv, belief).abs()).item())

            belief = self.flow(cv)

            confidence = float(base_conf * coherence)

            yield Step(premises[i], premises[i+1], coherence, shock_val, method, confidence)

            if coherence < self.cfg.coherence_min:
                yield Step("", "STOPPED: Low Coherence", 0.0, 0.0, method, 0.0)
                break

            if shock_val > self.cfg.shock_threshold:
                yield Step("", "STOPPED: High Shock", 0.0, 0.0, method, 0.0)
                break

            await asyncio.sleep(0)


# ==============================================================================
# [8] Demo
# ==============================================================================
async def demo():
    eng = Engine()
    query = "Is AI dangerous?"
    premises = [
        "AI learns from data.",
        "Data can be biased.",
        "Therefore AI can be biased.",
    ]

    async for step in eng.infer_stream(query, premises):
        print(step)

    eng.close()


def run_demo_sync():
    asyncio.run(demo())


if __name__ == "__main__":
    run_demo_sync()
