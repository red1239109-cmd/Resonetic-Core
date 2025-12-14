Resonetics (Via Negativa) — Runtime Controls v1
This repository contains the Runtime Controls layer for Resonetics:

Verdict-based reward shaping
Risk EMA-based action suppression (policy damping)
Near-collapse forced survival policy switch

Core file:
resonetics_via_negativa_runtime_controls_v1.py
What this is (in one breath)
A module that quantifies philosophical judgments (creative / bubble / collapse) as numerical signals (tension, coherence, pressure),
and uses those judgments to control reward, action, and survival mode at runtime.

Quick Start

from resonetics_via_negativa_runtime_controls_v1 import (
    ParadoxState, RiskEMAController, apply_controls
)

controller = RiskEMAController()

state = ParadoxState(
    tension=0.62,
    coherence=0.55,
    pressure_response=0.60,
    self_protecting=False,
    confidence=0.0,
)

result = apply_controls(
    state=state,
    base_reward=1.0,
    controller=controller,
    action_vec=[0.8, -0.2, 0.1],  # example continuous action vector
    action_space_hint="continuous"
)

print(result["verdict"])
print(result["risk"])
print(result["reward"])
print(result["survival"])

The Signals
The controller expects a ParadoxState:

tension (0–1): intensity of contradiction / conflict energy
coherence (0–1): logical and narrative consistency
pressure_response (0–1): resilience under verification, counterarguments, or noise
self_protecting (bool): defensive or avoidance pattern
confidence (0–1, optional): external certainty (defaults to 0 if omitted)

Outputs
apply_controls() returns:

verdict: type, energy, action, reason, signals
reward: base vs shaped (creative rewarded, bubble/collapse penalised)
risk: instant risk, EMA, damping alpha
action: original vs damped (suppressed when risk is high)
survival: whether forced, reason, and directive (for near-collapse)

Philosophy (tiny)
Via Negativa = “First define what should not happen.”

Suppress actions that lead to bubble or collapse at runtime
Switch to forced survival policy when near-collapse is detected

More details: see PHILOSOPHY.md and THEORY.md.
