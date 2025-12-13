KNOWN_ISSUES

This document lists known limitations and intentional constraints of the Resonetics environments and systems.

These are not bugs hidden in the implementation, but explicit design boundaries chosen to preserve interpretability, reproducibility, and research clarity.

1. Environment Is a Simplification of Real Kubernetes

The Kubernetes environments (resonetics_k8s_*) simulate cluster behavior using a small grid-based tensor abstraction.

Limitations:

No real pod scheduling

No networking, disk I/O, or latency modeling

CPU and memory dynamics are stochastic approximations

Scaling is modeled as instantaneous resource reduction

Why this is intentional

The goal is not operational fidelity, but decision-structure fidelity:

Does an agent learn when and where to intervene under pressure?

Realistic simulators would obscure this question behind engineering noise.

2. Red Queen Effect Is Exogenous and Inevitable

Memory growth (“leak”) and CPU drift are hard-coded dynamics, not emergent behavior.

Consequences:

The system will degrade if the agent does nothing

There is no stable equilibrium without intervention

Long-term survival requires continuous action

Why this is intentional

This models systems where:

entropy grows regardless of correctness

“doing nothing” is not neutral

maintenance is mandatory, not optional

The environment is designed to test maintenance intelligence, not optimization in static worlds.

3. Reward Shaping Is Hand-Crafted

Rewards are explicitly defined for:

cleaning effectiveness

scaling efficiency

budget waste

OOM and critical-node failures

This introduces potential biases:

Some strategies may be over- or under-rewarded

Certain edge behaviors may be unintentionally discouraged

Mitigation

Random and heuristic baselines are provided

Ablation experiments are supported

Reward terms are isolated and inspectable

The reward function is auditable by design.

4. Partial Observability Is Local and Fixed

The agent observes only a 3×3 neighborhood plus minimal global scalars.

Known effects:

Global state cannot be inferred perfectly

Long-range dependencies require movement

Credit assignment is delayed and noisy

Why this is intentional

This enforces:

exploration under uncertainty

local decision-making

failure without global omniscience

The environment is not meant to reward omnipotent policies.

5. Budget Mechanics Are Abstracted

The budget system models resource scarcity, not real billing.

Limitations:

No time-based billing

No differentiated cost per node type

No long-term debt or borrowing

Why this is intentional

The budget exists to enforce trade-offs, not accounting realism.

The agent must choose between:

acting now vs. saving resources

local fixes vs. global interventions

6. Termination Conditions Are Conservative

Episodes terminate when:

average system load exceeds a threshold

any OOM occurs

step limit is reached

This may shorten episodes for poorly performing agents.

Rationale

Early termination makes:

catastrophic policies visible quickly

training signals sharper

failure modes easier to analyze

7. Not Designed for AGI Claims

These systems:

do not reason symbolically

do not self-improve architectures

do not exhibit open-ended learning

do not generalize beyond defined dynamics

Any appearance of “intelligence” is a property of environment structure, not cognition.

Resonetics systems filter tensions — they do not generate understanding.

8. Benchmark Results Are Context-Specific

Reported benchmark numbers depend on:

random seeds

reward coefficients

grid size

termination thresholds

They should be interpreted as:

comparative (random vs heuristic vs learned)

structural (directional, not absolute)

reproducible under the same config

They are not universal performance claims.

Summary

Resonetics prioritizes:

interpretability over realism

structure over scale

auditable tension over opaque performance

These known issues are design decisions, not oversights.

They define the scope within which conclusions are valid.
