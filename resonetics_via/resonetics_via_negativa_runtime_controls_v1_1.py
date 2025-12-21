#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: resonetics_via_negativa_runtime_controls_v1_1.py
# Project: Resonetics (Via Negativa)
# Version: 1.1 (Runtime Controls - Safety Patch)
# Author: red1239109-cmd
# Copyright (c) 2025 Resonetics Project
# ==============================================================================
"""
Resonetics Via Negativa - Runtime Controls v1.1
Features:
1) 🔥 Verdict -> Reward shaping
2) 🧠 Risk EMA -> Action suppression (policy damping)
3) 🪦 Near-collapse -> Forced survival policy switch

v1.1 Fixes:
- signals now report CLAMPED values (no mismatch between verdict math and logs)
- RiskEMAController validates beta/min_alpha/max_alpha
- compute_energy normalizes by weight sum (avoids easy saturation)
- suppress_action_vector is safer across types (list/tuple/tensors/scalars)
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Any
import math

# -----------------------------
# Configuration (tweakable)
# -----------------------------
DEFAULT_THRESHOLDS = {
    # Collapse is "dangerously inconsistent or defensive"
    "collapse_tension_min": 0.70,
    "collapse_coherence_max": 0.35,
    "collapse_pressure_max": 0.45,
    "collapse_self_protecting": True,

    # Bubble is "inflated claim without support"
    "bubble_coherence_max": 0.70,
    "bubble_pressure_max": 0.60,

    # Near-collapse switch trigger
    "near_collapse_energy": 0.35,     # low energy implies system is unstable
    "near_collapse_pressure": 0.45,   # weak pressure response
    "near_collapse_coherence": 0.40,  # low coherence
}

DEFAULT_WEIGHTS = {
    "w_tension": 0.40,
    "w_coherence": 0.40,
    "w_pressure": 0.20,
    # Optional confidence term (if you compute it); default 0 if not provided
    "w_confidence": 0.10,
}

DEFAULT_REWARD_MAP = {
    # Reward shaping coefficients
    "creative_tension": +1.0,
    "bubble": -0.4,
    "collapse": -1.2,
}

DEFAULT_ACTION_RULES = {
    # How strongly to suppress actions when risk rises
    # alpha close to 1 => mild damping, close to 0 => heavy damping
    "risk_damping_min_alpha": 0.15,
    "risk_damping_max_alpha": 1.00,
    # EMA smoothing
    "risk_ema_beta": 0.90,
}

# -----------------------------
# Data container
# -----------------------------
@dataclass
class ParadoxState:
    tension: float            # 0..1
    coherence: float          # 0..1
    pressure_response: float  # 0..1
    self_protecting: bool
    confidence: float = 0.0   # 0..1 (optional)

    def clamp(self) -> "ParadoxState":
        self.tension = float(max(0.0, min(1.0, self.tension)))
        self.coherence = float(max(0.0, min(1.0, self.coherence)))
        self.pressure_response = float(max(0.0, min(1.0, self.pressure_response)))
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        self.self_protecting = bool(self.self_protecting)
        return self


# -----------------------------
# Verdict + Energy
# -----------------------------
def compute_energy(state: ParadoxState,
                   weights: Dict[str, float] = DEFAULT_WEIGHTS) -> float:
    """
    Energy is a weighted score of "productive tension structure".

    v1.1: Normalize by sum(weights) to avoid easy saturation when weights sum > 1.
    """
    st = state.clamp()

    w_t = float(weights.get("w_tension", 0.0))
    w_c = float(weights.get("w_coherence", 0.0))
    w_p = float(weights.get("w_pressure", 0.0))
    w_cf = float(weights.get("w_confidence", 0.0))

    denom = max(1e-9, (w_t + w_c + w_p + w_cf))

    e = (
        w_t * st.tension +
        w_c * st.coherence +
        w_p * st.pressure_response +
        w_cf * st.confidence
    ) / denom

    return float(max(0.0, min(1.0, e)))


def classify_paradox(state: ParadoxState,
                     thresholds: Dict[str, float] = DEFAULT_THRESHOLDS) -> Tuple[str, str]:
    """
    Returns: (verdict_type, reason)
      - creative_tension: sustained, coherent tension with decent pressure response
      - bubble: looks coherent-ish but collapses under pressure (inflated)
      - collapse: incoherent/defensive and unstable
    """
    st = state.clamp()

    # Collapse: incoherent + weak pressure + self-protecting defense pattern
    if (
        st.tension >= thresholds["collapse_tension_min"] and
        st.coherence <= thresholds["collapse_coherence_max"] and
        st.pressure_response <= thresholds["collapse_pressure_max"] and
        (st.self_protecting == thresholds["collapse_self_protecting"])
    ):
        return "collapse", "Internally inconsistent and defensive under pressure"

    # Bubble: weak coherence + weak pressure response, but NOT defensive/self-protecting
    if (
        st.coherence <= thresholds["bubble_coherence_max"] and
        st.pressure_response <= thresholds["bubble_pressure_max"] and
        not st.self_protecting
    ):
        return "bubble", "Inflated structure without sufficient pressure support"

    # Otherwise: creative tension (productive contradiction)
    return "creative_tension", "Sustained tension with enough coherence/pressure support"


def verdict_action(verdict_type: str) -> str:
    """Maps verdict -> recommended action."""
    if verdict_type == "creative_tension":
        return "PRESERVE_AND_FEED"
    if verdict_type == "bubble":
        return "IGNORE"
    return "FORCE_COLLAPSE"


def evaluate_paradox(state: ParadoxState) -> Dict[str, Any]:
    """
    One-shot evaluation output.

    v1.1: signals now come from CLAMPED state to avoid log/math mismatch.
    """
    st = state.clamp()
    vtype, reason = classify_paradox(st)
    energy = compute_energy(st)
    action = verdict_action(vtype)

    return {
        "type": vtype,
        "energy": round(float(energy), 3),
        "action": action,
        "reason": reason,
        "signals": {
            "tension": round(float(st.tension), 3),
            "coherence": round(float(st.coherence), 3),
            "pressure_response": round(float(st.pressure_response), 3),
            "self_protecting": bool(st.self_protecting),
            "confidence": round(float(st.confidence), 3),
        }
    }


# -----------------------------
# 🧠 Risk EMA -> Action suppression
# -----------------------------
class RiskEMAController:
    """
    Tracks EMA risk and outputs an action-damping alpha in [min_alpha, max_alpha].
    Higher risk => lower alpha => more conservative action amplitude.
    """
    def __init__(self,
                 beta: float = DEFAULT_ACTION_RULES["risk_ema_beta"],
                 min_alpha: float = DEFAULT_ACTION_RULES["risk_damping_min_alpha"],
                 max_alpha: float = DEFAULT_ACTION_RULES["risk_damping_max_alpha"]):

        # v1.1: validate/normalize params
        self.beta = float(max(0.0, min(1.0, beta)))

        min_a = float(min_alpha)
        max_a = float(max_alpha)
        if min_a > max_a:
            min_a, max_a = max_a, min_a  # swap if reversed

        self.min_alpha = float(max(0.0, min(1.0, min_a)))
        self.max_alpha = float(max(0.0, min(1.0, max_a)))

        # ensure range is not degenerate in a bad way
        if self.max_alpha < self.min_alpha:
            self.max_alpha = self.min_alpha

        self.risk_ema = 0.0

    def update(self, instant_risk: float) -> float:
        r = float(max(0.0, min(1.0, instant_risk)))
        self.risk_ema = self.beta * self.risk_ema + (1.0 - self.beta) * r
        return float(self.risk_ema)

    def damping_alpha(self) -> float:
        # risk_ema 0 -> alpha max, risk_ema 1 -> alpha min
        a = self.max_alpha - (self.max_alpha - self.min_alpha) * float(self.risk_ema)
        return float(max(self.min_alpha, min(self.max_alpha, a)))


def suppress_action_vector(action_vec, alpha: float):
    """
    Generic suppression for continuous action vectors:
    - action_vec can be float, list, tuple, numpy array, torch tensor
    - returns scaled action_vec

    v1.1: safer fallback handling
    """
    a = float(alpha)

    # Fast path: tensors / numpy arrays / scalars that support *
    try:
        return action_vec * a
    except Exception:
        pass

    # List/tuple path
    if isinstance(action_vec, (list, tuple)):
        scaled = [x * a for x in action_vec]
        return type(action_vec)(scaled)

    # Unknown type: safest is "do nothing" rather than crash
    return action_vec


# -----------------------------
# 🪦 Near-collapse -> Forced survival policy switch
# -----------------------------
def should_force_survival(verdict: Dict[str, Any],
                          thresholds: Dict[str, float] = DEFAULT_THRESHOLDS) -> Tuple[bool, str]:
    """
    Triggers survival mode when near-collapse is detected.
    We define "near-collapse" as:
      - low energy OR
      - low coherence + weak pressure response
    """
    e = float(verdict.get("energy", 0.0))
    sig = verdict.get("signals", {})
    coh = float(sig.get("coherence", 0.0))
    pr = float(sig.get("pressure_response", 0.0))

    if e <= thresholds["near_collapse_energy"]:
        return True, "Energy below near-collapse threshold"
    if coh <= thresholds["near_collapse_coherence"] and pr <= thresholds["near_collapse_pressure"]:
        return True, "Low coherence + weak pressure response (near-collapse)"
    return False, ""


def survival_policy_switch(action_space_hint: str = "discrete") -> Dict[str, str]:
    """
    Returns a safe-policy directive.
    - For discrete envs: bias toward "safe" actions (e.g., clean, stabilize)
    - For continuous envs: clamp magnitude, avoid exploration spikes
    """
    if action_space_hint == "discrete":
        return {
            "mode": "SURVIVAL",
            "policy": "SAFE_DISCRETE",
            "recommendation": "Prefer stabilizing actions; avoid risky exploration"
        }
    return {
        "mode": "SURVIVAL",
        "policy": "SAFE_CONTINUOUS",
        "recommendation": "Clamp action magnitude; avoid aggressive updates"
    }


# -----------------------------
# 🔥 Verdict -> Reward shaping
# -----------------------------
def shaped_reward(base_reward: float, verdict_type: str,
                  reward_map: Dict[str, float] = DEFAULT_REWARD_MAP,
                  energy: float = 0.0) -> float:
    """
    Reward shaping: verdict affects reward.
    Optionally scale by energy for creative_tension.
    """
    k = float(reward_map.get(verdict_type, 0.0))
    br = float(base_reward)

    if verdict_type == "creative_tension":
        en = float(max(0.0, min(1.0, energy)))
        bonus = k * (0.5 + 0.5 * en)  # bounded
        return float(br + bonus)

    return float(br + k)


# -----------------------------
# One integrated call
# -----------------------------
def apply_controls(state: ParadoxState,
                   base_reward: float,
                   controller: RiskEMAController,
                   action_vec,
                   action_space_hint: str = "discrete") -> Dict[str, Any]:
    """
    Integrates:
      - verdict evaluation
      - reward shaping
      - risk EMA -> action suppression
      - near-collapse -> survival switch
    """
    verdict = evaluate_paradox(state)

    # 1) 🔥 reward shaping
    new_reward = shaped_reward(float(base_reward), verdict["type"], energy=float(verdict["energy"]))

    # 2) 🧠 risk EMA update (instant risk heuristic)
    if verdict["type"] == "collapse":
        instant_risk = 0.95
    elif verdict["type"] == "bubble":
        instant_risk = 0.65
    else:
        # creative tension: risk depends on tension magnitude (productive but still risky)
        st = state.clamp()
        instant_risk = 0.25 + 0.50 * float(st.tension)

    risk_ema = controller.update(instant_risk)
    alpha = controller.damping_alpha()
    damped_action = suppress_action_vector(action_vec, alpha)

    # 3) 🪦 survival switch near collapse
    force_survival, why = should_force_survival(verdict)
    survival_directive = survival_policy_switch(action_space_hint) if force_survival else None

    return {
        "verdict": verdict,
        "reward": {
            "base": float(base_reward),
            "shaped": float(new_reward),
        },
        "risk": {
            "instant": round(float(instant_risk), 3),
            "ema": round(float(risk_ema), 3),
            "damping_alpha": round(float(alpha), 3),
        },
        "action": {
            "original": action_vec,
            "damped": damped_action,
        },
        "survival": {
            "forced": bool(force_survival),
            "reason": why,
            "directive": survival_directive,
        }
    }

