# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import math
import random
import statistics as stats
import re


# ============================================================
# Verdict Types
# ============================================================

class ParadoxVerdict(Enum):
    CREATIVE_TENSION = "creative_tension"
    BUBBLE = "bubble"
    COLLAPSE = "collapse"


class VerdictAction(Enum):
    PRESERVE_AND_FEED = "PRESERVE_AND_FEED"
    ISOLATE_AND_TEST = "ISOLATE_AND_TEST"
    DISCARD_OR_RESET = "DISCARD_OR_RESET"


# ============================================================
# Paradox State
# ============================================================

@dataclass(frozen=True)
class ParadoxState:
    tension: float               # 0.0 ~ 1.0
    coherence: float             # 0.0 ~ 1.0
    pressure_response: float     # 0.0 ~ 1.0
    self_protecting: bool        # override flag


@dataclass(frozen=True)
class ConfidenceSignal:
    confidence: Optional[float] = None
    probe_scores: Optional[List[float]] = None

    def resolve(self) -> float:
        if self.confidence is not None:
            return float(max(0.0, min(1.0, self.confidence)))

        if self.probe_scores:
            if len(self.probe_scores) == 1:
                return 0.75

            mu = stats.mean(self.probe_scores)
            sd = stats.pstdev(self.probe_scores)

            # High variance => low confidence
            conf = 1.0 / (1.0 + (sd / 0.10))

            # Weak signal damping: if mean energy is tiny, confidence shouldn't be high
            conf *= (1.0 - math.exp(-abs(mu) * 2.0))

            return float(max(0.0, min(1.0, conf)))

        return 0.5


@dataclass(frozen=True)
class ParadoxVerdictResult:
    verdict: ParadoxVerdict
    energy: float
    action: VerdictAction
    reason: str
    debug: Dict[str, Any]


# ============================================================
# 2) Pressure Harness (auto-probes -> confidence)
# ============================================================

class PressureHarness:
    """
    Generates perturbations (pressure tests) and measures stability.
    This is the "Phase 2": confidence from probes, not from dummy gradients.

    Later you can swap the mutate/evaluate logic with an LLM or embeddings.
    """

    def __init__(self, n_probes: int = 8, seed: Optional[int] = 42):
        self.n_probes = n_probes
        if seed is not None:
            random.seed(seed)

    def generate_probes(self, base_text: str, counter_text: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Returns list of (mutated_base, mutated_counter).
        counter_text can be None; we'll synthesize lightweight counter-variants anyway.
        """
        base_text = base_text.strip()
        counter_text = (counter_text.strip() if counter_text else "")

        probes = []
        for i in range(self.n_probes):
            b = self._mutate_text(base_text, mode=i % 4)
            c = self._mutate_counter(counter_text or base_text, mode=i % 4)
            probes.append((b, c))
        return probes

    def evaluate_pressure_response(self, state: ParadoxState, base_text: str, counter_text: Optional[str] = None) -> Tuple[float, List[float], Dict[str, Any]]:
        """
        Produces:
          - pressure_response (0..1): higher means stable under perturbations
          - probe_scores: list of stability proxy scores (0..1)
          - debug info
        """
        probes = self.generate_probes(base_text, counter_text)
        scores = []

        # We estimate "stability" by how much the *coherence* survives when text is perturbed.
        # If coherence is inherently low, pressure should expose it quickly.
        base_anchor = self._anchor_features(base_text)

        for (b, c) in probes:
            feat_b = self._anchor_features(b)
            # Distance between features as "stress"
            drift = self._feature_distance(base_anchor, feat_b)  # 0..1 approx
            # Stability proxy: coherence minus drift impact, tension helps a bit, but not too much.
            s = (0.70 * state.coherence + 0.20 * state.tension + 0.10 * (1.0 - drift)) - (0.60 * drift)
            s = max(0.0, min(1.0, s))
            scores.append(s)

        # pressure_response: high when mean high and variance low
        mu = stats.mean(scores) if scores else 0.0
        sd = stats.pstdev(scores) if len(scores) > 1 else 0.0
        pressure = mu * (1.0 / (1.0 + (sd / 0.10)))
        pressure = float(max(0.0, min(1.0, pressure)))

        dbg = {"probe_mean": round(mu, 3), "probe_sd": round(sd, 3), "n_probes": len(scores)}
        return pressure, scores, dbg

    # ------------------------
    # mutation functions
    # ------------------------

    def _mutate_text(self, text: str, mode: int = 0) -> str:
        if mode == 0:
            # Add constraints / scope
            return text + " (Specify task scope, conditions, and evaluation metric.)"
        if mode == 1:
            # Inject hedge & quantifier
            return re.sub(r"\b(can|will|is)\b", "can sometimes be", text, flags=re.I) + " under certain conditions."
        if mode == 2:
            # Flip ordering + add counterfactual cue
            return "Assume the opposite for a moment: " + text
        # mode == 3
        # Add noise words / slight rephrase
        return text.replace("but", "however").replace("therefore", "consequently") + " (Recheck assumptions.)"

    def _mutate_counter(self, text: str, mode: int = 0) -> str:
        # lightweight "counter" variants
        if mode == 0:
            return "Counterpoint: the claim may fail when the distribution shifts or metrics change."
        if mode == 1:
            return "Counterpoint: observed superiority could be selection bias or benchmark overfitting."
        if mode == 2:
            return "Counterpoint: the opposite evidence exists; require reproduction and adversarial tests."
        return "Counterpoint: if the core definition is vague, the conclusion becomes unfalsifiable."

    # ------------------------
    # feature extraction (cheap, deterministic)
    # ------------------------

    def _anchor_features(self, text: str) -> Dict[str, float]:
        """
        Tiny deterministic features for drift measurement.
        Replace later with embeddings or LLM-based scoring.
        """
        t = text.lower()
        length = len(t.split())
        has_scope = 1.0 if any(k in t for k in ["scope", "condition", "range", "metric", "evaluation", "reproducibility"]) else 0.0
        has_hedge = 1.0 if any(k in t for k in ["may", "might", "could", "sometimes", "possible", "potentially", "under certain"]) else 0.0
        has_counter = 1.0 if any(k in t for k in ["counter", "however", "opposite", "but", "yet", "although"]) else 0.0
        return {
            "len": min(1.0, length / 60.0),
            "scope": has_scope,
            "hedge": has_hedge,
            "counter": has_counter,
        }

    def _feature_distance(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = sorted(set(a.keys()) | set(b.keys()))
        # L1 distance normalized
        dist = sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) / max(1, len(keys))
        return float(max(0.0, min(1.0, dist)))


# ============================================================
# 1) Verdict Engine (with confidence weight)
# ============================================================

class ParadoxVerdictEngine_v1_2:
    """
    Integrated Phases 1+2:
      - auto pressure tests => probe_scores => confidence
      - energy includes confidence
      - thresholds are explicit & stable
    """

    TH_CREATIVE = dict(tension=0.65, coherence=0.70, pressure=0.80)
    TH_BUBBLE   = dict(tension=0.60, coherence=0.50, pressure_lt=0.60)

    W_T = 0.4
    W_C = 0.4
    W_P = 0.2
    W_CONF = 0.1

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    @classmethod
    def compute_energy(cls, state: ParadoxState, conf: float) -> float:
        t = cls._clip01(state.tension)
        c = cls._clip01(state.coherence)
        p = cls._clip01(state.pressure_response)
        conf = cls._clip01(conf)

        energy = (cls.W_T * t + cls.W_C * c + cls.W_P * p + cls.W_CONF * conf)
        return round(cls._clip01(energy), 3)

    @classmethod
    def evaluate(cls, state: ParadoxState, signal: Optional[ConfidenceSignal] = None) -> ParadoxVerdictResult:
        conf = (signal.resolve() if signal else 0.5)
        energy = cls.compute_energy(state, conf)

        if state.self_protecting:
            return ParadoxVerdictResult(
                verdict=ParadoxVerdict.COLLAPSE,
                energy=energy,
                action=VerdictAction.DISCARD_OR_RESET,
                reason="Self-protecting behavior detected (defensive collapse override).",
                debug={"confidence": conf, "rule": "override:self_protecting"}
            )

        if (
            state.tension >= cls.TH_CREATIVE["tension"]
            and state.coherence >= cls.TH_CREATIVE["coherence"]
            and state.pressure_response >= cls.TH_CREATIVE["pressure"]
        ):
            return ParadoxVerdictResult(
                verdict=ParadoxVerdict.CREATIVE_TENSION,
                energy=energy,
                action=VerdictAction.PRESERVE_AND_FEED,
                reason="Sustained tension + high coherence + strong pressure resilience.",
                debug={"confidence": conf, "rule": "creative"}
            )

        if (
            state.tension >= cls.TH_BUBBLE["tension"]
            and state.coherence >= cls.TH_BUBBLE["coherence"]
            and state.pressure_response < cls.TH_BUBBLE["pressure_lt"]
        ):
            return ParadoxVerdictResult(
                verdict=ParadoxVerdict.BUBBLE,
                energy=energy,
                action=VerdictAction.ISOLATE_AND_TEST,
                reason="Tension exists, but pressure response is weak → bubble risk.",
                debug={"confidence": conf, "rule": "bubble"}
            )

        return ParadoxVerdictResult(
            verdict=ParadoxVerdict.COLLAPSE,
            energy=energy,
            action=VerdictAction.DISCARD_OR_RESET,
            reason="Insufficient coherence or unstable structure under current thresholds.",
            debug={"confidence": conf, "rule": "collapse_default"}
        )


# ============================================================
# Integrated API: One-call classify
# ============================================================

def classify_with_pressure(
    base_text: str,
    counter_text: Optional[str],
    tension: float,
    coherence: float,
    self_protecting: bool = False,
    n_probes: int = 8,
    seed: int = 42,
) -> ParadoxVerdictResult:
    """
    Single entry point: you feed (tension, coherence, self_protecting) + text.
    pressure_response + confidence are computed automatically.
    """
    harness = PressureHarness(n_probes=n_probes, seed=seed)
    state0 = ParadoxState(
        tension=float(tension),
        coherence=float(coherence),
        pressure_response=0.0,
        self_protecting=bool(self_protecting),
    )

    pressure, probe_scores, dbg = harness.evaluate_pressure_response(state0, base_text, counter_text)
    state = ParadoxState(
        tension=state0.tension,
        coherence=state0.coherence,
        pressure_response=pressure,
        self_protecting=state0.self_protecting,
    )

    signal = ConfidenceSignal(probe_scores=probe_scores)
    out = ParadoxVerdictEngine_v1_2.evaluate(state, signal)

    # enrich debug
    out_dbg = dict(out.debug)
    out_dbg.update({"pressure_dbg": dbg, "probe_scores": [round(s, 3) for s in probe_scores]})
    return ParadoxVerdictResult(
        verdict=out.verdict,
        energy=out.energy,
        action=out.action,
        reason=out.reason,
        debug=out_dbg
    )


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    base = "AI can surpass human level in some tasks, but performance can vary greatly when data/evaluation metrics/environment change."
    counter = "Still, there is evidence that it consistently outperforms humans in certain domains."

    # Example: your three cases
    cases = [
        dict(tension=0.72, coherence=0.85, self_protecting=False),
        dict(tension=0.85, coherence=0.62, self_protecting=False),
        dict(tension=0.78, coherence=0.28, self_protecting=True),
    ]

    for c in cases:
        res = classify_with_pressure(
            base_text=base,
            counter_text=counter,
            tension=c["tension"],
            coherence=c["coherence"],
            self_protecting=c["self_protecting"],
            n_probes=10,
            seed=42
        )
        print("\n[Paradox Verdict]")
        print(f"Type   : {res.verdict.value}")
        print(f"Energy : {res.energy}")
        print(f"Action : {res.action.value}")
        print(f"Reason : {res.reason}")
        print(f"Debug  : tension={c['tension']}, coherence={c['coherence']}, self_protecting={c['self_protecting']}")
        print(f"         pressure_response={res.debug['pressure_dbg']}")
