Paradox Verdict Classification Rules

This document defines the formal criteria used to classify paradoxical states in Resonetics-based systems.

The goal is not to label outcomes as good or bad,
but to distinguish productive tension from illusion and failure.

1) Core Assumption

A paradox is not inherently valuable.

Value emerges only when contradiction survives pressure without collapsing or self-deceiving.

2) Primary Signals

Each paradox state is evaluated using the following signals:

2.1 Tension

Measures the magnitude of internal contradiction.

High tension means strong opposing claims or forces coexist.

Range: 0.0 – 1.0

2.2 Coherence

Measures structural consistency across representations or reasoning layers.

High coherence means the contradiction does not fragment logic.

Range: 0.0 – 1.0

2.3 Pressure Response

Measures stability when external constraints or stress are applied.

High values indicate resilience under challenge.

Range: 0.0 – 1.0

2.4 Self-Protecting Flag

Indicates defensive behavior such as:

narrative shielding,

gradient suppression,

confidence inflation,

refusal to update.

Type: Boolean

This flag overrides numerical scores.

3) Energy Calculation (Creative Potential)

Energy represents usable creative potential, not correctness.

Default weighted formulation:

energy =
  0.4 * tension +
  0.4 * coherence +
  0.2 * pressure_response


Optional confidence modulation may be added experimentally,
but verdict thresholds must remain invariant.

4) Verdict Classes
4.1 CREATIVE_TENSION

Definition
A paradox that maintains coherence under sustained pressure.

Formal Conditions

tension ≥ 0.65
coherence ≥ 0.70
pressure_response ≥ 0.80
self_protecting == false


Interpretation

Contradiction generates energy instead of noise.

Suitable for preservation and further evolution.

Action

PRESERVE_AND_FEED

4.2 BUBBLE

Definition
A paradox that appears strong but degrades when stressed.

Formal Conditions

tension ≥ 0.60
coherence ≥ 0.50
pressure_response < 0.60
self_protecting == false


Interpretation

Initial plausibility without structural resilience.

Requires further testing or isolation.

Action

ISOLATE_AND_TEST

4.3 COLLAPSE

Definition
A paradox that fails structurally or becomes defensive.

Formal Conditions

coherence < 0.40
OR self_protecting == true


Interpretation

Either incoherent or actively shielding itself from correction.

Not suitable for creative continuation.

Action

DISCARD_OR_RESET

5) Precedence Rules (Non-Negotiable)

self_protecting == true
→ Always COLLAPSE, regardless of other metrics.

Thresholds are exclusive, not averaged.

High tension cannot compensate for low coherence.

High coherence cannot excuse pressure failure.

Energy score does not override verdict class.

6) Example Classifications
Example A — Creative Tension
{
  "tension": 0.72,
  "coherence": 0.85,
  "pressure_response": 0.88,
  "self_protecting": false
}


→ CREATIVE_TENSION

Example B — Bubble
{
  "tension": 0.85,
  "coherence": 0.62,
  "pressure_response": 0.42,
  "self_protecting": false
}


→ BUBBLE

Example C — Collapse
{
  "tension": 0.78,
  "coherence": 0.28,
  "pressure_response": 0.35,
  "self_protecting": true
}


→ COLLAPSE

7) What This System Explicitly Avoids

No subjective judgment

No post-hoc rationalization

No intelligence claims

No optimization for “impressive” outcomes

8) Design Principle

A paradox is only valuable
if it can be pressed harder than it wants to be.
