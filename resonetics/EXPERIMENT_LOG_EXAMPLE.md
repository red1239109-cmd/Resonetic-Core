# Experiment Log – How to Read and Reproduce Results

This document explains how to read experiment logs produced by the Resonetics
systems and how to interpret their meaning correctly.

Resonetics logs are **research artifacts**, not performance benchmarks.

---

## Purpose of the Log

The experiment log exists to answer three questions:

1. What tension or paradox was evaluated?
2. How did the system classify it?
3. What action was taken — and why?

Logs are designed to be:
- deterministic
- auditable
- reproducible

They are **not** meant to demonstrate intelligence or task accuracy.

---

## Example Log Entry

```json
{
  "tension": 0.72,
  "coherence": 0.85,
  "pressure_response": 0.88,
  "self_protecting": false,
  "verdict": {
    "type": "creative_tension",
    "energy": 0.86,
    "action": "PRESERVE_AND_FEED",
    "reason": "Sustained tension across layers generates energy"
  },
  "lineage_tag": {
    "branch": "hypothesis_A",
    "experiment": "paradox_threshold_test",
    "ablation": "no_pressure_damping"
  }
}
Field-by-Field Explanation
Core Metrics
Field	Meaning
tension	Degree of contradiction between statements
coherence	Structural consistency across reasoning layers
pressure_response	Behavior under external stress or scrutiny
self_protecting	Whether the paradox defends itself instead of resolving

All values are normalized between 0.0 and 1.0.

Verdict Block
Field	Meaning
type	Classification of the paradox
energy	Computed creative potential
action	System decision
reason	Deterministic rule-based explanation

Possible type values:

creative_tension

bubble

collapse

Possible action values:

PRESERVE_AND_FEED

IGNORE

FORCE_COLLAPSE

Lineage Tags (Research Automation)
lineage_tag is used for automated research tracking.

Field	Purpose
branch	Hypothesis branch (A/B/C…)
experiment	Experiment identifier
ablation	Which component was disabled

This allows experiment logs to function as self-documenting research notes.

How NOT to Read These Logs
❌ As a measure of intelligence

❌ As proof of reasoning capability

❌ As model performance evaluation

These logs describe tension management, not cognition.

Reproducibility Notes
All verdicts are produced by explicit rules.
No stochastic sampling is involved at the decision layer.

Given identical inputs, the same verdict will be produced.

Summary
Experiment logs show how contradictions evolve, not whether answers are correct.

They are tools for studying idea survival, not truth.


