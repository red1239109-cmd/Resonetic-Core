How to Read Resonetics Experiment Logs

This document explains how to interpret experiment logs produced by Resonetics-based systems.
These logs are designed for traceability, falsifiability, and review under pressure, not for storytelling.

1) Purpose of the Log

Each experiment log answers four questions:

What was tested

Under what conditions

How the system responded under pressure

Why a specific verdict was assigned

The log is not meant to prove success.
It is meant to preserve evidence, including partial failure and instability.

2) Minimal Log Structure

A typical log entry contains the following fields:

experiment_id: EXP-042
timestamp: 2025-03-18T14:22:11Z

lineage_tag:
  branch: hypothesis_A
  experiment_tag: paradox_stress_test
  ablation: coherence_disabled

metrics:
  tension: 0.72
  coherence: 0.85
  pressure_response: 0.88
  self_protecting: false

derived:
  energy: 0.857
  verdict: creative_tension

action:
  decision: PRESERVE_AND_FEED

reason:
  summary: Sustained tension with high coherence under pressure


Each section is described below.

3) Lineage Tracking (Reproducibility)

The lineage_tag block records why this run exists.

branch
Indicates the hypothesis line (e.g., A/B branching).

experiment_tag
Describes the intent of the run (stress test, ablation, comparison).

ablation
Explicitly states what was removed or disabled.

This ensures that results are never detached from their experimental context.

4) Core Metrics (Raw Signals)

The following metrics are measured independently:

tension
Degree of contradiction or internal conflict detected.

coherence
Structural consistency across layers or representations.

pressure_response
Stability of behavior when constraints or stressors are applied.

self_protecting
Boolean flag indicating defensive behavior (e.g., collapse avoidance,
narrative shielding, gradient suppression).

These are not scores of intelligence.
They are signals used for classification.

5) Derived Values (Interpretation Layer)
Energy

Energy is a weighted aggregation of raw signals:

energy = weighted(tension, coherence, pressure_response [, confidence])


It represents usable creative potential, not quality or correctness.

Energy can be high in both productive and unstable states.

Verdict Classification

Each log is assigned exactly one verdict:

Verdict	Meaning
creative_tension	Contradiction remains coherent under pressure
bubble	Apparent strength collapses when stressed
collapse	Defensive or incoherent failure

The verdict is rule-based, not subjective.

6) Action Field (System Response)

The action field records what the system does next:

PRESERVE_AND_FEED
Retain and allow further evolution.

ISOLATE_AND_TEST
Quarantine for further stress testing.

DISCARD_OR_RESET
Remove from active exploration.

Actions are logged to ensure non-retroactive justification.

7) Reason Field (Human-Readable Justification)

The reason section explains why the verdict was assigned.

This is intentionally concise and grounded in metrics.

It exists for:

reviewers,

future audits,

and self-critique.

8) What This Log Does Not Claim

These logs do not claim:

intelligence,

understanding,

intention,

or autonomy.

They document behavior under constraints.

9) How Reviewers Should Use This Log

Reviewers are encouraged to:

compare logs across ablations,

check verdict consistency under similar metrics,

and challenge classification thresholds.

If a verdict appears questionable,
that is a signal for further experiment, not an error to be hidden.

10) Design Principle

If a result cannot survive logging,
it cannot survive reality.
