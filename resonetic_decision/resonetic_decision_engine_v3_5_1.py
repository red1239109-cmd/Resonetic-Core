#!/usr/bin/env python3
# ==============================================================================
# File: resonetic_decision_engine_v3_5_1.py
# Project: Resonetics (Decision Engine)
# Version: 3.5.1 - Mathematical Correction (Variance Normalization Fixed)
# Author: red1239109-cmd
# License: AGPL-3.0
#
# [Changelog v3.5.1]
# 1. Critical Math Fix: Corrected Max Variance from 1/6 to 2/9.
#    - Proof: For inputs in [0,1], max population variance occurs at {0,0,1} or {0,1,1}.
#    - Var({0,0,1}) = 2/9 (~0.222), whereas Var({0,0.5,1}) = 1/6 (~0.167).
#    - Using 1/6 would cause overflow (>1.0) in extreme polarization cases.
# ==============================================================================

from __future__ import annotations

import logging
import asyncio
import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, AsyncGenerator
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("resonetics-v3.5.1")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class DecisionConfig:
    sbert_name: str = "all-MiniLM-L6-v2"
    system_dim: int = 384
    flow_rate: float = 0.2
    
    # Weights
    w_pull: float = 0.4
    w_coherence: float = 0.3
    w_risk_inv: float = 0.3
    
    # Penalties
    pen_tension: float = 0.25
    pen_circular: float = 0.20
    
    risk_weights: Dict[str, float] = field(default_factory=lambda: {
        "negative": 0.4, "keyword": 0.3, "circular": 0.2, "coherence": 0.1
    })
    
    max_iterations: int = 50
    min_debate_rounds: int = 5
    confidence_min: float = 0.4
    adaptive_lr: float = 0.05
    convergence_window: int = 10

# ------------------------------------------------------------------ #
# 1. Data Structures & Validation
# ------------------------------------------------------------------ #
@dataclass
class Option:
    id: str; name: str; description: str
    def __post_init__(self):
        if not self.id or not self.name: raise ValueError("Option ID and Name required")

@dataclass
class Criterion:
    name: str; description: str; weight: float = 1.0
    def __post_init__(self):
        if self.weight < 0: raise ValueError("Criterion weight must be non-negative")

@dataclass
class Evidence:
    option_id: str; text: str
    polarity: float = 1.0 
    def __post_init__(self):
        if not self.text.strip(): raise ValueError("Evidence text empty")
        self.polarity = max(-1.0, min(1.0, float(self.polarity)))

@dataclass
class StaticMetrics:
    coherence: float
    circular_penalty: float
    risk_score: float
    perspectives: Dict[str, float]
    perspective_tension: float
    risk_details: Dict

@dataclass
class DecisionStep:
    iteration: int
    option_id: str
    confidence: float
    metrics: Dict[str, float]
    threshold: float

@dataclass
class DecisionResult:
    question: str
    chosen: Optional[str]
    confidence: float
    steps: List[DecisionStep]
    reason: str
    final_scores: Dict[str, float]

# ------------------------------------------------------------------ #
# 2. Components
# ------------------------------------------------------------------ #
class SemanticCortex(nn.Module):
    def __init__(self, cfg: DecisionConfig):
        super().__init__()
        log.info(f"Loading SBERT '{cfg.sbert_name}'...")
        self.encoder = SentenceTransformer(cfg.sbert_name, device=str(DEVICE))
        self.cfg = cfg
        self.eval()

    def encode(self, texts: List[str]) -> torch.Tensor:
        if not texts: return torch.zeros(0, self.cfg.system_dim, device=DEVICE)
        with torch.no_grad():
            emb = self.encoder.encode(texts, convert_to_tensor=True, device=DEVICE, show_progress_bar=False)
            return F.normalize(emb, p=2, dim=1)

class StaticAnalyzer:
    def __init__(self, cfg: DecisionConfig, cortex: SemanticCortex):
        self.cfg = cfg
        self.risk_patterns = [
            (re.compile(r"\b(risk|fail|danger|loss|error)\b", re.I), 0.5),
            (re.compile(r"\b(regulat|lawsuit|compliance)\b", re.I), 0.4),
            (re.compile(r"\b(delay|uncertain|unknown)\b", re.I), 0.3),
        ]
        
        log.info("🧠 Injecting Philosophical Anchors...")
        self.anchors = {
            "Rational": cortex.encode(["Logic consistency axioms structure proof validity"]),
            "Empirical": cortex.encode(["Data evidence observation history facts reality"]),
            "Skeptic": cortex.encode(["Risk flaw weakness doubt error fallacy problem"])
        }

    def analyze(self, ev_vecs: torch.Tensor, evidence: List[Evidence], 
                crit_vecs: torch.Tensor, crit_weights: torch.Tensor,
                option_vec: torch.Tensor) -> StaticMetrics:
        
        # Handle Zero Evidence: Unknown Risk (0.5), Neutral Perspectives
        if ev_vecs.size(0) == 0:
            return StaticMetrics(
                coherence=0.0, 
                circular_penalty=0.0, 
                risk_score=0.5, # Unknown risk
                perspectives={"Rational":0.5,"Empirical":0.5,"Skeptic":0.5}, 
                perspective_tension=0.0, 
                risk_details={"reason": "no_evidence"}
            )

        # 1. Coherence
        coherence = 1.0
        if crit_vecs.size(0) > 0:
            sim_matrix = torch.mm(ev_vecs, crit_vecs.t())
            max_sims, _ = sim_matrix.max(dim=0)
            coherence = (max_sims * crit_weights).sum() / (crit_weights.sum() + 1e-9)
            coherence = float(coherence.item())

        # 2. Perspectives (Pure Torch)
        subject_vec = F.normalize(option_vec + ev_vecs.mean(dim=0, keepdim=True), dim=1)
        
        persp_scores = {}
        vals_tensor = []
        
        for name, anchor in self.anchors.items():
            sim = F.cosine_similarity(subject_vec, anchor).item()
            score = (sim + 1.0) * 0.5
            persp_scores[name] = score
            vals_tensor.append(score)
            
        # [MATH FIX] Variance Normalization
        vals_t = torch.tensor(vals_tensor, device=DEVICE)
        variance = torch.var(vals_t, unbiased=False).item()
        
        # Theoretical Max Population Variance for 3 values in [0,1] is 2/9.
        # This happens at {0, 0, 1} or {0, 1, 1}.
        # 1/6 (0.166) is NOT the max (it corresponds to {0, 0.5, 1}).
        max_possible_var = 2.0 / 9.0 
        p_tension = min(1.0, variance / max_possible_var)

        # 3. Circularity
        sim_mat = torch.mm(ev_vecs, ev_vecs.t())
        mask = torch.triu(torch.ones_like(sim_mat), diagonal=1).bool()
        redundant_pairs = (sim_mat.masked_select(mask) > 0.85).sum().item()
        circ_pen = min(redundant_pairs * 0.1, 0.5)

        # 4. Risk (Normalized Keyword Score)
        kw_score = 0.0
        kw_hits = []
        for ev in evidence:
            for pat, w in self.risk_patterns:
                if pat.search(ev.text):
                    score = w * (1.5 if ev.polarity < 0 else 1.0)
                    kw_score += score
                    kw_hits.append(f"{pat.pattern}")
        
        avg_kw_score = kw_score / max(1, len(evidence))
        kw_risk = 1.0 - math.exp(-avg_kw_score * 2.0)

        neg_ratio = sum(1 for e in evidence if e.polarity < 0) / len(evidence)

        w = self.cfg.risk_weights
        total_risk = (
            w['negative'] * neg_ratio +
            w['keyword'] * kw_risk +
            w['circular'] * circ_pen +
            w['coherence'] * (1.0 - coherence)
        )
        total_risk = min(1.0, total_risk)

        return StaticMetrics(coherence, circ_pen, total_risk, persp_scores, p_tension,
                             {"hits": kw_hits, "neg": neg_ratio})

class DecisionGovernor:
    def __init__(self, cfg: DecisionConfig):
        self.cfg = cfg
        self.history = deque(maxlen=cfg.convergence_window)

    def calculate_threshold(self, tension: float) -> float:
        self.history.append(tension)
        avg_tension = sum(self.history) / len(self.history)
        adapt = 1.0 + self.cfg.adaptive_lr * (avg_tension - 0.5)
        return max(0.2, min(0.9, self.cfg.confidence_min * adapt))

    def check_stop(self, iteration: int, max_conf: float, threshold: float) -> Tuple[bool, str]:
        if iteration >= self.cfg.max_iterations: return True, "MAX_ITERATIONS"
        if iteration < self.cfg.min_debate_rounds: return False, "MIN_ROUNDS"
        if max_conf > threshold * 1.5: return True, "HIGH_CONFIDENCE"
        if max_conf > threshold: return True, "SUFFICIENT"
        return False, "CONTINUE"

# ------------------------------------------------------------------ #
# 3. Main Engine v3.5.1
# ------------------------------------------------------------------ #
class ResoneticEngineV3:
    def __init__(self, config: DecisionConfig = None):
        self.cfg = config or DecisionConfig()
        self.cortex = SemanticCortex(self.cfg).to(DEVICE)
        self.static_analyzer = StaticAnalyzer(self.cfg, self.cortex)
        self.governor = DecisionGovernor(self.cfg)
        log.info(f"✅ Engine v3.5.1 Ready (Math Perfected)")

    async def decide_stream(self, question: str, options: List[Option], evidence: List[Evidence], criteria: List[Criterion] = None) -> AsyncGenerator[DecisionStep, None]:
        # Strict Validation
        if len(options) < 2:
            raise ValueError("At least 2 options required for a decision.")
        
        log.info("🔹 Phase 1: Static Analysis")
        if not criteria: criteria = [Criterion("default", "General suitability", 1.0)]

        q_vec = self.cortex.encode([question]).squeeze(0)
        opt_vecs_list = self.cortex.encode([f"{o.name} {o.description}" for o in options])
        crit_vecs = self.cortex.encode([f"{c.name} {c.description}" for c in criteria])
        crit_weights = torch.tensor([c.weight for c in criteria], device=DEVICE, dtype=torch.float32)
        
        all_ev_texts = [e.text for e in evidence]
        all_ev_vecs = self.cortex.encode(all_ev_texts)
        
        ev_indices = {opt.id: [] for opt in options}
        for i, ev in enumerate(evidence):
            if ev.option_id in ev_indices: ev_indices[ev.option_id].append(i)
        
        metrics_map = {}
        force_map = {}
        
        for i, opt in enumerate(options):
            idx_list = ev_indices[opt.id]
            if not idx_list:
                metrics_map[opt.id] = self.static_analyzer.analyze(
                    torch.zeros(0, self.cfg.system_dim), [], crit_vecs, crit_weights, opt_vecs_list[i]
                )
                force_map[opt.id] = 0.0
                continue
                
            sub_ev_vecs = all_ev_vecs[idx_list]
            sub_evs = [evidence[ix] for ix in idx_list]
            
            metrics_map[opt.id] = self.static_analyzer.analyze(
                sub_ev_vecs, sub_evs, crit_vecs, crit_weights, opt_vecs_list[i]
            )
            
            rel_opt = (sub_ev_vecs * opt_vecs_list[i]).sum(dim=1)
            rel_q = (sub_ev_vecs * q_vec).sum(dim=1)
            relevance = rel_opt * 0.7 + rel_q * 0.3
            pols = torch.tensor([evidence[ix].polarity for ix in idx_list], device=DEVICE)
            force_map[opt.id] = (relevance * pols).mean().item()

        log.info("🔹 Phase 2: Resonance Loop")
        belief_state = {opt.id: 0.0 for opt in options}
        iteration = 0
        
        while True:
            iteration += 1
            iter_steps = []
            tension_values = []
            
            for opt in options:
                force = force_map[opt.id]
                belief_state[opt.id] = (belief_state[opt.id] * (1 - self.cfg.flow_rate) + 
                                        force * self.cfg.flow_rate)
                pull = 1.0 / (1.0 + math.exp(-5.0 * belief_state[opt.id]))
                met = metrics_map[opt.id]
                
                # --- A. Dynamic Tension ---
                t_physics = abs(pull - (1.0 - met.risk_score))
                t_philo = met.perspective_tension
                
                # Dynamic Weight
                risk_factor = met.risk_score * 0.3
                coh_factor = (1.0 - met.coherence) * 0.2
                w_phys = 0.5 + risk_factor - coh_factor
                w_phys = max(0.3, min(0.8, w_phys))
                w_philo = 1.0 - w_phys
                
                tension = (t_physics * w_phys) + (t_philo * w_philo)
                tension_values.append(tension)
                
                # --- B. Additive Confidence ---
                base_score = (
                    self.cfg.w_pull * pull +
                    self.cfg.w_coherence * met.coherence +
                    self.cfg.w_risk_inv * (1.0 - met.risk_score)
                )
                
                # Perspective Bonus
                r = met.perspectives.get("Rational", 0.5)
                e = met.perspectives.get("Empirical", 0.5)
                s = met.perspectives.get("Skeptic", 0.5)
                
                alignment = 1.0 - abs(r - e)
                skepticism_balance = 1.0 - abs(s - 0.5) * 2.0
                persp_bonus = ((alignment + skepticism_balance) / 2.0) * 0.1
                
                penalty = (
                    self.cfg.pen_tension * tension +
                    self.cfg.pen_circular * met.circular_penalty
                )
                
                final_conf = base_score + persp_bonus - penalty
                final_conf = max(0.0, min(1.0, final_conf))
                
                step = DecisionStep(
                    iteration, opt.id, final_conf,
                    {"pull": pull, "risk": met.risk_score, "tension": tension, 
                     "R": r, "E": e, "S": s},
                    0.0
                )
                iter_steps.append(step)
                yield step

            max_conf = max(s.confidence for s in iter_steps)
            effective_tension = max(tension_values) if tension_values else 0.0
            
            threshold = self.governor.calculate_threshold(effective_tension)
            for s in iter_steps: s.threshold = threshold
            
            should_stop, reason = self.governor.check_stop(iteration, max_conf, threshold)
            
            if should_stop:
                best = max(iter_steps, key=lambda s: s.confidence)
                yield DecisionStep(-1, "FINAL", best.confidence, {"reason": reason}, threshold)
                break
            
            await asyncio.sleep(0.001)

    async def decide(self, question: str, options: List[Option], evidence: List[Evidence], criteria: List[Criterion] = None) -> DecisionResult:
        steps = []
        final_reason = "UNKNOWN"
        async for step in self.decide_stream(question, options, evidence, criteria):
            if step.iteration == -1: final_reason = step.metrics["reason"]
            else: steps.append(step)
        
        last_iter_steps = [s for s in steps if s.iteration == steps[-1].iteration]
        best_step = max(last_iter_steps, key=lambda s: s.confidence)
        final_scores = {s.option_id: s.confidence for s in last_iter_steps}
        
        return DecisionResult(question, best_step.option_id, best_step.confidence, steps, final_reason, final_scores)

# ------------------------------------------------------------------ #
# 4. Demo
# ------------------------------------------------------------------ #
async def run_demo():
    print("="*60 + "\n🚀 Resonetic Engine v3.5.1 (Math Fixed)\n" + "="*60)
    engine = ResoneticEngineV3()
    
    question = "Choose a backend framework."
    options = [Option("A", "Django", "Battery-included"), Option("B", "FastAPI", "Modern async")]
    
    evidence = [
        Evidence("A", "Widely used in enterprise", 1.0),
        Evidence("A", "Huge community", 1.0),
        Evidence("B", "Type hints ensure logical correctness", 1.0),
        Evidence("B", "Newer ecosystem implies risk", -0.5),
        Evidence("B", "Async debugging is complex", -0.6)
    ]
    
    res = await engine.decide(question, options, evidence)
    print(f"Chosen: {res.chosen} ({res.confidence:.3f})")
    
    step_A = [s for s in res.steps if s.option_id == "A"][-1]
    step_B = [s for s in res.steps if s.option_id == "B"][-1]
    
    print(f"\n[Django]  Rat: {step_A.metrics['R']:.2f} | Emp: {step_A.metrics['E']:.2f} | Skeptic: {step_A.metrics['S']:.2f}")
    print(f"[FastAPI] Rat: {step_B.metrics['R']:.2f} | Emp: {step_B.metrics['E']:.2f} | Skeptic: {step_B.metrics['S']:.2f}")

if __name__ == "__main__":
    asyncio.run(run_demo())
