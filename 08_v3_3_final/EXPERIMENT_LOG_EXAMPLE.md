# EXPERIMENT_LOG_EXAMPLE.md

This document explains **how to read, write, and interpret experiment logs**
generated while working with the Resonetics project.

It is intended for:
- reviewers
- future maintainers
- yourself, six months later

This is **not a benchmark leaderboard**.  
It is a **research diary with structure**.

---

## Purpose of Experiment Logs

Resonetics experiments are exploratory by design.

The log exists to answer **why something was tried**, not just **what worked**.

Each experiment should make it possible to reconstruct:
- the hypothesis
- the tension being tested
- the verdict produced by the system
- the consequence of that verdict

If an idea fails, that is still valid data.

---

## Log Entry Structure

Each experiment entry follows this canonical structure:

```text
[DATE] [VERSION] [LINEAGE_TAG]

Hypothesis:
What contradiction or tension is being tested?

Setup:
- Code version
- Parameters / thresholds
- What is intentionally enabled or disabled

Paradox Signals:
- tension
- coherence
- pressure_response
- self_protecting
- confidence (if used)

Verdict:
- type
- energy
- action
- reason

Runtime Effects:
- reward shaping outcome
- risk EMA behavior
- action suppression
- survival switch (if triggered)

Outcome:
What actually happened in the system?

Interpretation:
Why this result makes sense (or doesn’t).

Next Action:
- feed
- ignore
- collapse
- branch
Example 1 — Creative Tension (Feed)

YYYY-MM-DD | Via Negativa v1.0 | branch:A / exp:paradox-scope / ablation:none

Hypothesis:
A scoped claim can outperform humans in narrow tasks
without generalizing beyond its conditions.

Setup:
- collapse thresholds unchanged
- confidence disabled
- reward shaping enabled

Paradox Signals:
tension=0.72
coherence=0.85
pressure_response=0.88
self_protecting=False

Verdict:
type=creative_tension
energy=0.857
action=PRESERVE_AND_FEED
reason=Sustained tension with coherence and pressure support

Runtime Effects:
- reward bonus applied (+0.93)
- risk EMA stabilized at 0.41
- no survival switch

Outcome:
Agent maintained performance across perturbations.

Interpretation:
The contradiction remained productive because
it stayed bounded by scope and evidence.

Next Action:
Feed with controlled variation.
Example 2 — Bubble (Ignore)
text

YYYY-MM-DD | Via Negativa v1.0 | branch:B / exp:marketing-claim / ablation:pressure

Hypothesis:
A strong narrative can survive even with weak pressure response.

Setup:
- pressure_response weight reduced
- confidence enabled

Paradox Signals:
tension=0.85
coherence=0.62
pressure_response=0.42
self_protecting=False

Verdict:
type=bubble
energy=0.53
action=IGNORE
reason=Inflated structure without sufficient pressure support

Runtime Effects:
- reward penalty (-0.4)
- risk EMA increased to 0.58
- action damping applied

Outcome:
Performance decayed rapidly under stress.

Interpretation:
The claim looked coherent but collapsed under pressure.
Classic bubble pattern.

Next Action:
Ignore. Do not branch.
Example 3 — Collapse (Force Collapse)

YYYY-MM-DD | Via Negativa v1.0 | branch:C / exp:defensive-loop / ablation:coherence

Hypothesis:
Self-protecting logic can stabilize a failing system.

Setup:
- coherence threshold lowered
- survival switch enabled

Paradox Signals:
tension=0.78
coherence=0.28
pressure_response=0.35
self_protecting=True

Verdict:
type=collapse
energy=0.31
action=FORCE_COLLAPSE
reason=Internally inconsistent and defensive under pressure

Runtime Effects:
- strong reward penalty (-1.2)
- risk EMA spiked to 0.81
- survival policy forced

Outcome:
System entered safe mode; exploration halted.

Interpretation:
Defense mechanisms masked failure instead of resolving it.
Collapse was correct and necessary.

Next Action:
Collapse. Remove assumption.
Reading the Logs (Reviewer Guide)
When reviewing logs, ask:

Is the hypothesis explicit?

Do the signals justify the verdict?

Does the runtime behavior match the verdict?

Was the next action consistent?

If all four align, the experiment is valid
—even if the result is negative.

What Not to Do
❌ Do not log only successes
❌ Do not hide failed ideas
❌ Do not retroactively rewrite hypotheses
❌ Do not treat logs as marketing material

This is a research artifact, not a pitch deck.

Philosophy
Resonetics experiments are not about proving intelligence.

They are about learning which contradictions:

deserve to evolve

deserve to be ignored

deserve to die

The log is where that judgment becomes explicit.
