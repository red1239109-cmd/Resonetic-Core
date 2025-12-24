#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: belief_update_governor.py
# Product: Resonetic Decision - Belief Update Governor (Single Module)
#
# Purpose:
# - Govern belief updates with "decision not assignment"
# - Inputs: coherence, shock, observation_quality(optional), dt(optional)
# - Outputs: action (ABSORB/DAMPEN/HOLD/ROLLBACK), alpha, reason, telemetry
#
# No external dependencies.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple


# ------------------------------------------------------------------------------
# Public Types
# ------------------------------------------------------------------------------
Action = str  # "ABSORB" | "DAMPEN" | "HOLD" | "ROLLBACK"


@dataclass(frozen=True)
class GovernorConfig:
    # --- Thresholds ---
    coherence_min: float = 0.25     # below -> HOLD/ROLLBACK territory
    coherence_good: float = 0.55    # above -> absorb favored
    shock_threshold: float = 0.75   # above -> HOLD/ROLLBACK
    shock_soft: float = 0.45        # above -> dampen favored

    # --- Update strength (alpha) ---
    alpha_max: float = 0.35         # hard cap for single-step update
    alpha_min: float = 0.00         # allow full block
    alpha_default: float = 0.15     # baseline alpha if "ok"

    # --- Hysteresis (prevents flip-flop) ---
    hold_streak_to_absorb: int = 2  # require consecutive good steps to exit HOLD bias
    rollback_cooldown_steps: int = 3  # after rollback, be conservative for N steps

    # --- Shock memory / decay ---
    shock_memory_decay: float = 0.90  # 0~1 (higher means longer memory)
    shock_accum_gain: float = 0.40    # how much current shock adds into memory

    # --- Optional external signal weights ---
    use_obs_quality: bool = True
    obs_quality_weight: float = 0.40   # penalize alpha when low quality

    # --- Decision policy toggles ---
    enable_rollback: bool = True
    rollback_requires_low_coherence: bool = True  # rollback only if coherence low too


@dataclass
class GovernorState:
    step: int = 0
    hold_streak: int = 0
    rollback_cooldown: int = 0
    shock_memory: float = 0.0
    last_action: Action = "ABSORB"


@dataclass(frozen=True)
class Decision:
    action: Action
    alpha: float
    reason: str
    meta: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x  # excludes NaN


# ------------------------------------------------------------------------------
# Belief Update Governor
# ------------------------------------------------------------------------------
class BeliefUpdateGovernor:
    """
    Governs belief updates: ABSORB / DAMPEN / HOLD / ROLLBACK.

    Inputs:
      - coherence: 0..1 (agreement / consistency)
      - shock:     0..1 (disruption / anomaly)
      - obs_quality (optional): 0..1 (trust in observation/source)
      - dt (optional): time delta to decay shock memory more realistically

    Output:
      - Decision(action, alpha, reason, meta)
    """

    def __init__(self, cfg: Optional[GovernorConfig] = None):
        self.cfg = cfg or GovernorConfig()
        self.state = GovernorState()

    def reset(self) -> None:
        self.state = GovernorState()

    def decide(
        self,
        *,
        coherence: float,
        shock: float,
        obs_quality: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> Decision:
        s = self.state
        c = self.cfg

        s.step += 1

        # Validate / sanitize inputs
        coherence = float(coherence) if _is_number(coherence) else 0.0
        shock = float(shock) if _is_number(shock) else 1.0
        coherence = _clip(coherence, 0.0, 1.0)
        shock = _clip(shock, 0.0, 1.0)

        if obs_quality is None or (not _is_number(obs_quality)):
            obs_q = 1.0
            obs_q_present = False
        else:
            obs_q = _clip(float(obs_quality), 0.0, 1.0)
            obs_q_present = True

        # --- Shock memory update (decay + accumulation) ---
        # dt가 있으면 decay를 좀 더 현실적으로(간단 모델) 적용
        decay = c.shock_memory_decay
        if dt is not None and _is_number(dt):
            # dt가 크면 더 decay되도록: decay^(dt)
            dtv = max(0.0, float(dt))
            decay = decay ** max(1.0, dtv)

        s.shock_memory = _clip(
            s.shock_memory * decay + shock * c.shock_accum_gain,
            0.0,
            1.0
        )

        # --- Rollback cooldown tick ---
        if s.rollback_cooldown > 0:
            s.rollback_cooldown -= 1

        # --- Base alpha computation ---
        alpha = c.alpha_default

        # Coherence 기반 조정: 낮으면 줄이고, 높으면 늘림
        if coherence <= c.coherence_min:
            alpha *= 0.0
        elif coherence < c.coherence_good:
            # 0~1 사이에서 선형 스케일
            span = max(1e-6, (c.coherence_good - c.coherence_min))
            t = (coherence - c.coherence_min) / span
            alpha *= (0.25 + 0.75 * t)  # 25%~100%
        else:
            # 매우 좋으면 약간 상향(하지만 cap)
            alpha *= 1.15

        # Shock 기반 조정: 높으면 강하게 감쇠
        effective_shock = max(shock, s.shock_memory)
        if effective_shock >= c.shock_threshold:
            alpha = 0.0
        elif effective_shock > c.shock_soft:
            span = max(1e-6, (c.shock_threshold - c.shock_soft))
            t = (effective_shock - c.shock_soft) / span
            alpha *= (1.0 - 0.85 * t)  # soft~threshold에서 최대 85% 감쇠

        # Observation quality 페널티 (옵션)
        if c.use_obs_quality and obs_q_present:
            # quality 낮으면 alpha를 감소
            alpha *= (1.0 - c.obs_quality_weight * (1.0 - obs_q))

        # Cooldown 중이면 보수적으로
        if s.rollback_cooldown > 0:
            alpha *= 0.25

        alpha = _clip(alpha, c.alpha_min, c.alpha_max)

        # --- Decide action ---
        reason = "score_based"
        action: Action = "ABSORB"

        # 1) Hard stop conditions
        if coherence <= c.coherence_min:
            # coherence가 너무 낮으면 HOLD 또는 ROLLBACK 후보
            if c.enable_rollback and (not c.rollback_requires_low_coherence or coherence <= c.coherence_min):
                # shock까지 높으면 rollback이 더 말이 됨
                if effective_shock >= c.shock_soft:
                    action = "ROLLBACK"
                    reason = "low_coherence_high_shock"
                else:
                    action = "HOLD"
                    reason = "low_coherence"
            else:
                action = "HOLD"
                reason = "low_coherence"
            alpha = 0.0

        elif effective_shock >= c.shock_threshold:
            # 충격이 임계 초과 -> 업데이트 중지
            if c.enable_rollback and (not c.rollback_requires_low_coherence):
                action = "ROLLBACK"
                reason = "shock_threshold_exceeded"
            else:
                action = "HOLD"
                reason = "shock_threshold_exceeded"
            alpha = 0.0

        # 2) Soft stop / dampening
        else:
            if effective_shock > c.shock_soft or coherence < c.coherence_good:
                action = "DAMPEN"
                reason = "soft_dampen"

            # 3) Hysteresis: HOLD에서 빠져나오려면 연속 good 필요
            if s.last_action == "HOLD":
                if coherence >= c.coherence_good and effective_shock <= c.shock_soft:
                    s.hold_streak += 1
                    if s.hold_streak >= c.hold_streak_to_absorb:
                        # 탈출 허용
                        pass
                    else:
                        action = "HOLD"
                        reason = "hysteresis_hold"
                        alpha = 0.0
                else:
                    s.hold_streak = 0
                    action = "HOLD"
                    reason = "hysteresis_hold"
                    alpha = 0.0
            else:
                # HOLD가 아니면 streak 초기화
                s.hold_streak = 0

        # --- Update state for next decision ---
        if action == "ROLLBACK":
            s.rollback_cooldown = c.rollback_cooldown_steps
            # rollback 이후 shock memory를 약간 올려서 보수적 모드 유지
            s.shock_memory = _clip(max(s.shock_memory, effective_shock) * 1.05, 0.0, 1.0)

        s.last_action = action

        meta = {
            "step": s.step,
            "coherence": coherence,
            "shock": shock,
            "shock_memory": s.shock_memory,
            "effective_shock": effective_shock,
            "obs_quality": obs_q if obs_q_present else None,
            "rollback_cooldown": s.rollback_cooldown,
            "hold_streak": s.hold_streak,
            "alpha_raw_cap": c.alpha_max,
        }

        return Decision(action=action, alpha=alpha, reason=reason, meta=meta)

    # Convenience: apply belief update with alpha (numerical or vector-like)
    def apply_update(self, belief: Any, candidate: Any, decision: Decision) -> Any:
        """
        Applies update:
        - ABSORB: belief <- candidate (alpha forced to 1.0 if candidate supports assignment)
        - DAMPEN: belief <- (1-alpha)*belief + alpha*candidate  (numeric / supports * and +)
        - HOLD: belief unchanged
        - ROLLBACK: belief unchanged (caller may revert to checkpoint if available)
        """
        a = decision.action
        alpha = float(decision.alpha)

        if a == "HOLD" or a == "ROLLBACK":
            return belief

        if a == "ABSORB":
            # absorb means "accept strongly", but keep alpha cap semantics for stability
            # If you want full override, caller can set alpha_max=1.0 or bypass.
            if alpha >= 0.99:
                return candidate
            # fall through to blend as a safe absorb
            a = "DAMPEN"

        # Blend
        try:
            return (1.0 - alpha) * belief + alpha * candidate
        except Exception:
            # If cannot blend, fallback to candidate for absorb-like behavior
            return candidate


# ------------------------------------------------------------------------------
# Minimal self-test / example
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    gov = BeliefUpdateGovernor()

    belief = 0.50
    stream = [
        # coherence, shock, obs_quality
        (0.80, 0.10, 0.90),
        (0.60, 0.40, 0.80),
        (0.30, 0.20, 0.60),
        (0.20, 0.60, 0.70),  # low coherence + shock -> HOLD/ROLLBACK
        (0.70, 0.20, 0.90),
        (0.75, 0.10, 0.90),
    ]

    for i, (coh, sh, q) in enumerate(stream, start=1):
        cand = min(1.0, belief + 0.10)  # dummy candidate
        d = gov.decide(coherence=coh, shock=sh, obs_quality=q)
        new_belief = gov.apply_update(belief, cand, d)
        print(f"[{i}] action={d.action:<8} alpha={d.alpha:.3f} belief={belief:.3f}->{new_belief:.3f} reason={d.reason}")
        belief = new_belief
