# Resonetics (Via Negativa) — Runtime Controls v1

This repository provides the **Runtime Control Layer** for the Resonetics project.

It translates high-level philosophical judgments  
(**creative tension / bubble / collapse**)  
into **concrete runtime interventions** over reward, action amplitude, and survival behavior.

---

## What This Module Does (Short Version)

This module answers one question:

> **“Given the current state of contradiction, coherence, and pressure,  
> how should the system behave right now?”**

It does this by:
1. Classifying the state into a **verdict**
2. Shaping rewards accordingly
3. Suppressing actions when accumulated risk rises
4. Forcing a survival policy when near-collapse is detected

---

## Core Features

### 1) Verdict → Reward Shaping 🔥
- **creative_tension** → positive reward bonus (scaled by energy)
- **bubble** → mild penalty
- **collapse** → strong penalty

This biases learning toward **productive contradictions**, not delusions or defensive collapse.

---

### 2) Risk EMA → Action Suppression 🧠
- Instant risk is derived from the verdict
- Risk is smoothed with an **Exponential Moving Average (EMA)**
- Higher accumulated risk → lower action amplitude (`alpha`)

This prevents oscillation, runaway behavior, and instability.

---

### 3) Near-Collapse → Forced Survival Policy 🪦
If the system is near collapse:
- Low energy, or
- Low coherence *and* weak pressure response

Then a **survival directive** is triggered:
- Discrete environments → prefer stabilizing actions
- Continuous environments → clamp action magnitude

---

## Core File

resonetics_via_negativa_runtime_controls_v1.py


This file is self-contained and framework-agnostic.

---

## Minimal Usage Example

```python
from resonetics_via_negativa_runtime_controls_v1 import (
    ParadoxState,
    RiskEMAController,
    apply_controls
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
    action_vec=[0.8, -0.2, 0.1],
    action_space_hint="continuous"
)

print(result["verdict"])
print(result["risk"])
print(result["reward"])
print(result["survival"])

The Input: ParadoxState

Each evaluation step requires a ParadoxState:

Field	Range	Meaning
tension	0..1	Degree of contradiction or conflict
coherence	0..1	Internal consistency / alignment
pressure_response	0..1	Robustness under stress, noise, or challenge
self_protecting	bool	Defensive / self-sealing behavior
confidence	0..1	Optional external confidence signal

All values are automatically clamped to valid ranges.

The Output

apply_controls() returns a dictionary containing:

verdict

type (creative_tension, bubble, collapse)

energy score

recommended action

explanatory reason

reward

base reward

shaped reward

risk

instant risk

EMA risk

damping alpha

action

original action

damped action

survival

whether survival mode is forced

reason

policy directive

Design Philosophy (Very Short)

This module follows Via Negativa:

Do not define intelligence directly.
Define what must not be allowed to persist.

Bubble → reduce influence

Collapse → suppress and stabilize

Creative tension → preserve and feed

The goal is runtime sanity, not metaphysical perfection.

See:

PHILOSOPHY.md

THEORY.md

Known Limitations

Verdict classification is heuristic, not learned

Thresholds are domain-specific and require tuning

Action suppression assumes multiplicative scaling is valid

Survival directives are advisory unless enforced upstream

Details: KNOWN_ISSUES.md

License

AGPL-3.0

Network use triggers source disclosure

Derivative works must remain AGPL-3.0

No commercial exceptions

See the LICENSE file for full terms.

Intended Use

This module is designed to be:

Plugged into simulations (e.g., Gardener / Prophet)

Used as a safety or stability layer

Readable and auditable

It is intentionally simple, explicit, and non-magical.

Final Note

This is not a “clever trick.”

It is a discipline:

Observe tension

Test coherence

Apply pressure

Intervene only when necessary
