# Resonetics (Via Negativa) — Runtime Controls v1.0

**Engineering Tension, Not Intelligence.**

`resonetics_via_negativa_runtime_controls_v1.py` is a small, auditable control layer that turns “contradiction signals” into **runtime behavior**:

1) 🔥 **Verdict → Reward shaping**  
2) 🧠 **Risk EMA → Action suppression (policy damping)**  
3) 🪦 **Near-collapse → Forced survival policy switch**

This is **not AGI**, not a consciousness simulator, and not a chatbot.  
It’s a **runtime governor**: it decides which tensions are worth feeding, which are noise, and which are dangerous.

---

## What problem does this solve?

Most optimization loops (RL, online learning, adaptive systems) treat instability as a bug.

Resonetics treats instability as **a structured signal**:

- Some contradictions are **productive** (`creative_tension`)
- Some are **inflated narratives** (`bubble`)
- Some are **self-protecting breakdowns** (`collapse`)

Instead of “always converge”, this layer asks:

> Should this contradiction be preserved, ignored, or collapsed?

---

## Core Concepts

### Inputs: `ParadoxState`

A minimal state vector describing a contradiction in runtime terms:

- `tension` (0..1): how strong / “red-zone” the contradiction is  
- `coherence` (0..1): structural consistency across layers  
- `pressure_response` (0..1): how well it holds under stress  
- `self_protecting` (bool): defensive / evasive behavior signal  
- `confidence` (0..1, optional): extra reliability term (defaults to 0)

```py
state = ParadoxState(
    tension=0.72,
    coherence=0.85,
    pressure_response=0.88,
    self_protecting=False,
    confidence=0.0
)
Outputs: Verdict + Energy + Action
The evaluator returns a rule-based, reproducible decision:

type: creative_tension | bubble | collapse

energy: 0..1 (weighted structure score)

action: PRESERVE_AND_FEED | IGNORE | FORCE_COLLAPSE

reason: human-readable rationale

How it works
1) 🔥 Verdict → Reward shaping
Reward shaping uses the verdict type:

creative_tension → reward bonus (scaled by energy)

bubble → mild penalty

collapse → strong penalty

Default map:

py
코드 복사
DEFAULT_REWARD_MAP = {
  "creative_tension": +1.0,
  "bubble": -0.4,
  "collapse": -1.2,
}
2) 🧠 Risk EMA → Action suppression
A risk EMA controller tracks ongoing risk and returns a damping factor alpha:

risk low → alpha ≈ 1.0 (normal policy)

risk high → alpha → min_alpha (conservative policy)

This is designed to reduce action spikes during unstable periods.

3) 🪦 Near-collapse → Forced survival switch
When the system is near collapse, it returns a survival directive:

Discrete systems: prefer stabilizing actions (clean/stabilize)

Continuous systems: clamp magnitude, avoid exploration spikes

Triggers (defaults):

low energy or

low coherence + weak pressure_response

Quickstart
Minimal example
py
코드 복사
from resonetics_via_negativa_runtime_controls_v1 import (
    ParadoxState, RiskEMAController, apply_controls
)

controller = RiskEMAController()

state = ParadoxState(
    tension=0.72,
    coherence=0.85,
    pressure_response=0.88,
    self_protecting=False
)

base_reward = 0.10
action = 5  # could be discrete (int) or continuous vector

out = apply_controls(
    state=state,
    base_reward=base_reward,
    controller=controller,
    action_vec=action,
    action_space_hint="discrete"
)

print(out["verdict"])
print(out["reward"])
print(out["risk"])
print(out["survival"])
Tuning
All important behavior is explicitly configurable:

DEFAULT_THRESHOLDS

collapse / bubble classification boundaries

near-collapse survival switch thresholds

DEFAULT_WEIGHTS

energy composition (tension/coherence/pressure/confidence)

DEFAULT_ACTION_RULES

EMA smoothing beta and damping range

DEFAULT_REWARD_MAP

reward shaping coefficients

This is intended to be reviewer-friendly: no “magic learning”, only clear knobs.

Design goals
Rule-based, auditable, reproducible

Runtime-safe: bounded outputs, clamped inputs

Composable: drops into RL, adaptive control, online learning loops

Via Negativa: remove failure modes before “adding intelligence”

What this is NOT
❌ Not AGI

❌ Not consciousness

❌ Not claiming intelligence emergence

❌ Not a general-purpose reasoning engine

Resonetics does not “think”.
It filters which tensions are worth evolving.

License
AGPL-3.0 (see file header).
If you use this over a network, you must provide source to users per AGPL.

Philosophy
“Perfection is achieved, not when there is nothing more to add,
but when there is nothing left to take away.”
— Antoine de Saint-Exupéry

In this project, “take away” means:

unsafe exploration under risk

inflated contradictions (bubbles)

defensive incoherence (collapse)

What remains is a system that can hold tension long enough to do useful work.
