
---

```md
# PHILOSOPHY.md

# Resonetics Philosophy (Via Negativa)
Resonetics is a practical meditation on instability.

The world is not a static equilibrium. It is a drift.
Entropy shows up as memory leaks, load creep, cascading failure, and “almost-working” systems that collapse at the edge.

Instead of asking “how do we create a perfect system?”, we ask:

> **What prevents collapse when collapse is the default attractor?**

This is “Via Negativa”: define what must *not* happen, detect it early, and apply controls that reduce catastrophe.

---

## The 4 signals

### 1) Reality
“What is the system doing right now?”
- In Gardener: the `grid` (entropy/dirt field)
- In K8s env: CPU/MEM load and OOM thresholds

### 2) Structure
“What are the regularities we enforce?”
- Local policies: move toward dirt, clean effectively, prioritize critical nodes
- Structural regularizers: periodic attraction, cost constraints, bounded actions

### 3) Tension
“What indicates a meaningful conflict between stability and drift?”
- In Gardener: entropy gradient + population volatility + emergency move ratio
- In K8s: budget pressure + hotspot dynamics + OOM risk

### 4) Flow (A-Version)
“Is the system becoming *sensitive* to small perturbations?”
Flow is not “vibes”. It is sensitivity.

We measure Flow as an input-noise Lipschitz-ish estimate:
- pick state metrics f(state) (e.g., total entropy, hotspot rate)
- add small noise with amplitude eps
- measure how much those metrics move
- normalize by eps²
- smooth with EMA

**Interpretation**  
High Flow means “the system is brittle” — small nudges change the system a lot.
That’s near-collapse physics, not mysticism.

---

## Verdict language (for controls, not poetry)
We use a 3-way verdict as a control label:

- **creative**: stable enough to explore
- **bubble**: inflated stability (looks fine but pressure response is weak)
- **collapse**: incoherent + defensive (system is breaking)

Verdict is meant to trigger:
- reward shaping
- action damping (Risk EMA)
- survival policy switch near collapse

---

## What Resonetics is NOT
- Not a claim of new physics.
- Not a proof that “3 is the universe’s secret number.”
- Not a shortcut around careful evaluation.

It is a framework to turn qualitative intuition into measurable signals:
risk, flow, tension, and survival constraints.

---

## A simple promise
If the system survives longer under the same drift, the controls helped.
If it collapses faster, the controls were wrong.

That’s the entire religion. (It’s a science-flavored one.)


