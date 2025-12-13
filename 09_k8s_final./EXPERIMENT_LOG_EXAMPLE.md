EXPERIMENT_LOG_EXAMPLE.md
How to Read These Logs

This document shows example experiment logs produced by the Resonetics environments.
The goal is not to impress with numbers, but to demonstrate how tension-aware systems behave over time.

Each experiment records:

configuration

baseline comparison

aggregate statistics

qualitative interpretation

Experiment 01 — Baseline Comparison
Objective

Evaluate whether structured decision-making outperforms randomness under entropy pressure.

Environment

Environment: KubernetesSmartTensorEnv

Grid: 4×4

Observation: 3×3×3 tensor + position + budget

Max steps per episode: 500

Red Queen Effect: enabled (memory drift + CPU noise)

Budget recovery: 0.5 / step

Agents
Random Policy

Uniform random action selection

No memory, no strategy

Heuristic Policy

Rule-based controller:

Clean when local memory > 0.7

Scale only when hotspots exceed threshold

Avoid actions when budget is critically low

Results (50 episodes each)
======================================================================
✅ BENCHMARK RESULTS (Random vs Heuristic)
======================================================================
random    | reward μ=-12.456 σ=4.821 | steps μ=312.4 | oom_rate=1.82 | crit_rate=0.68 | avg_load μ=0.512
heuristic | reward μ=+18.912 σ=3.215 | steps μ=456.8 | oom_rate=0.34 | crit_rate=0.12 | avg_load μ=0.378
----------------------------------------------------------------------
Heuristic win-rate: 50/50 (ties: 0)
======================================================================

Interpretation

Random agents survive only by chance
→ frequent OOM events
→ early termination
→ negative expected reward

Heuristic agents:

delay failure

reduce average load

preserve budget

avoid critical nodes

The gap confirms that entropy-aware behavior matters even in a simplified environment.

Experiment 02 — Budget Pressure Stress Test
Objective

Observe agent behavior under aggressive budget constraints.

Configuration Changes

Initial budget reduced by 50%

Scaling cost increased by 20%

OOM penalties unchanged

Observed Behavior

Random agents collapse faster than baseline

Heuristic agents:

reduce scaling frequency

prefer targeted cleaning

accept mild overload instead of catastrophic OOM

Key Insight

Under scarcity, inaction can be optimal.

The environment rewards restraint, not constant intervention.

Experiment 03 — Reward Sensitivity Probe
Objective

Detect reward-shaping artifacts.

Method

Increase clean-action reward by +25%

Decrease time penalty by −50%

Result

Agents begin over-cleaning

Budget depletion accelerates

Long-term survival worsens despite higher short-term rewards

Interpretation

This confirms that:

Local reward maximization does not guarantee global stability.

Reward shaping introduces ethical bias into agent behavior.

Experiment 04 — Failure Mode Trace
Typical Collapse Pattern

Memory drift accumulates unnoticed

Budget spent on low-priority cleaning

Critical node enters OOM

Episode terminates abruptly

Why This Matters

The agent does not fail because it lacks power —
it fails because it misallocates attention.

Reproducibility Notes

All experiments run with fixed seeds

Gymnasium API compliance ensures deterministic resets

Logs can be reproduced by re-running benchmark scripts

What These Logs Do NOT Claim

No generalization beyond this environment

No claim of intelligence

No emergence of planning or foresight

These logs show behavior under tension, not cognition.

Final Remark

Resonetics experiments are not about winning.

They are about identifying:

which pressures destroy systems

which tensions generate resilience

and which interventions make things worse
