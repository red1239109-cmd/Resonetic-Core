# ==============================================================================
# File: resonetics_via_negativa_runtime_controls_v1.py
# Project: Resonetics (Via Negativa)
# Version: 1.0 (Runtime Controls)
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

"""
Resonetics Via Negativa - Runtime Controls v1
Features:
1) 🔥 Verdict -> Reward shaping
2) 🧠 Risk EMA -> Action suppression (policy damping)
3) 🪦 Near-collapse -> Forced survival policy switch
"""
from dataclasses import dataclass
from typing import Dict, Tuple
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
    "near_collapse_energy": 0.35, # low energy implies system is unstable
    "near_collapse_pressure": 0.45, # weak pressure response
    "near_collapse_coherence": 0.40, # low coherence
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
    tension: float # 0..1
    coherence: float # 0..1
    pressure_response: float # 0..1
    self_protecting: bool
    confidence: float = 0.0 # 0..1 (optional)
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
    """
    st = state.clamp()
    e = (
        weights["w_tension"] * st.tension +
        weights["w_coherence"] * st.coherence +
        weights["w_pressure"] * st.pressure_response +
        weights.get("w_confidence", 0.0) * st.confidence
    )
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
    # Bubble: coherence isn't great OR pressure response is weak => inflated narrative
    # (but not fully collapse/defensive)
    if (
        st.coherence <= thresholds["bubble_coherence_max"] and
        st.pressure_response <= thresholds["bubble_pressure_max"] and
        not st.self_protecting
    ):
        return "bubble", "Inflated structure without sufficient pressure support"
    # Otherwise: creative tension (productive contradiction)
    return "creative_tension", "Sustained tension with enough coherence/pressure support"
def verdict_action(verdict_type: str) -> str:
    """
    Maps verdict -> recommended action.
    """
    if verdict_type == "creative_tension":
        return "PRESERVE_AND_FEED"
    if verdict_type == "bubble":
        return "IGNORE"
    return "FORCE_COLLAPSE"
def evaluate_paradox(state: ParadoxState) -> Dict:
    """
    One-shot evaluation output.
    """
    vtype, reason = classify_paradox(state)
    energy = compute_energy(state)
    action = verdict_action(vtype)
    return {
        "type": vtype,
        "energy": round(energy, 3),
        "action": action,
        "reason": reason,
        "signals": {
            "tension": round(state.tension, 3),
            "coherence": round(state.coherence, 3),
            "pressure_response": round(state.pressure_response, 3),
            "self_protecting": bool(state.self_protecting),
            "confidence": round(state.confidence, 3),
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
        self.beta = float(beta)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.risk_ema = 0.0
    def update(self, instant_risk: float) -> float:
        r = float(max(0.0, min(1.0, instant_risk)))
        self.risk_ema = self.beta * self.risk_ema + (1.0 - self.beta) * r
        return self.risk_ema
    def damping_alpha(self) -> float:
        # risk_ema 0 -> alpha max, risk_ema 1 -> alpha min
        a = self.max_alpha - (self.max_alpha - self.min_alpha) * self.risk_ema
        return float(max(self.min_alpha, min(self.max_alpha, a)))
def suppress_action_vector(action_vec, alpha: float):
    """
    Generic suppression for continuous action vectors:
    - action_vec can be float, list, numpy array, torch tensor
    - returns scaled action_vec
    """
    try:
        return action_vec * alpha
    except Exception:
        # fallback for lists
        return [x * alpha for x in action_vec]
# -----------------------------
# 🪦 Near-collapse -> Forced survival policy switch
# -----------------------------
def should_force_survival(verdict: Dict,
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
def survival_policy_switch(action_space_hint: str = "discrete") -> Dict:
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
    k = reward_map.get(verdict_type, 0.0)
    if verdict_type == "creative_tension":
        # feed high-energy contradictions more (bounded)
        bonus = k * (0.5 + 0.5 * float(max(0.0, min(1.0, energy))))
        return float(base_reward + bonus)
    return float(base_reward + k)
# -----------------------------
# One integrated call
# -----------------------------
def apply_controls(state: ParadoxState,
                   base_reward: float,
                   controller: RiskEMAController,
                   action_vec,
                   action_space_hint: str = "discrete") -> Dict:
    """
    Integrates:
      - verdict evaluation
      - reward shaping
      - risk EMA -> action suppression
      - near-collapse -> survival switch
    """
    verdict = evaluate_paradox(state)
    # 1) 🔥 reward shaping
    new_reward = shaped_reward(base_reward, verdict["type"], energy=verdict["energy"])
    # 2) 🧠 risk EMA update:
    # Use instantaneous risk heuristic:
    # - bubble/collapse are higher risk; creative tension moderate
    if verdict["type"] == "collapse":
        instant_risk = 0.95
    elif verdict["type"] == "bubble":
        instant_risk = 0.65
    else:
        # creative tension: risk depends on tension magnitude (productive but still risky)
        instant_risk = 0.25 + 0.50 * float(state.tension)
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
