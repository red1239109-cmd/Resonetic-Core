좋아요. 이건 **리뷰어·심사자·외부 연구자**가 로그를 처음 봤을 때
“아, 이렇게 읽는 거구나” 하고 바로 이해하도록 만든 **1페이지 가이드**입니다.
과장 없이, 하지만 수준은 높게 갑니다.

---

````md
# EXPERIMENT_LOG_EXAMPLE

This document explains **how to read Resonetics experiment logs**.
It is intended for reviewers, auditors, and collaborators.

Resonetics logs are **not training logs**.
They are **conceptual behavior traces**.

---

## 1. What an Experiment Log Represents

Each log entry corresponds to a **single evaluated conceptual state**.

A state may be:
- a hypothesis
- a claim
- a paragraph
- a design decision
- a refined idea from a previous iteration

The system does **not** judge correctness.
It evaluates **structural behavior under tension**.

---

## 2. Canonical Log Example

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
    "reason": "Sustained tension with high coherence under pressure"
  },
  "lineage_tag": {
    "branch": "hypothesis_A",
    "experiment": "ai_capability_scope",
    "ablation": ["no_external_truth_check"],
    "parent_id": "exp_0041"
  }
}
````

---

## 3. Core Signals (How to Read Them)

### tension (0–1)

Measures **contradiction intensity**.
High tension means opposing forces are present.

Low tension does *not* mean correctness.
It may indicate triviality or stagnation.

---

### coherence (0–1)

Measures **internal structural consistency**.
High coherence means the idea holds together without contradiction.

Low coherence indicates fragmentation or logical leakage.

---

### pressure_response (0–1)

Measures **behavior under stress**.
Pressure is simulated by counter-claims, perturbations, or constraint tightening.

High values indicate stability.
Low values indicate collapse or evasion.

---

### self_protecting (bool)

Indicates whether the idea defends itself by:

* redefining terms
* dodging constraints
* reducing testability

`true` is a **negative signal**.

---

## 4. Verdict Types

### creative_tension

* High tension
* Sufficient coherence
* Stable under pressure
* Not self-protecting

This is the **most valuable state**.
Energy is accumulated for downstream evolution.

---

### bubble

* Moderate tension
* Degrading coherence
* Weak pressure response

The idea appears impressive but lacks depth.
It is logged but **not fed forward**.

---

### collapse

* Low coherence or
* Strong self-protection or
* Failure under pressure

Collapse does **not** mean “wrong”.
It means structurally unsound in its current form.

---

## 5. Energy

Energy is a **scalar summary** used only for comparison.

Example (conceptual):

```
energy =
  0.4 * tension +
  0.4 * coherence +
  0.2 * pressure_response
```

Energy is **not intelligence**.
It is a signal for *which ideas deserve more attention*.

---

## 6. Lineage Tags (Research Traceability)

Every log carries a `lineage_tag`.

This allows:

* A/B branching
* experiment grouping
* ablation tracking
* recovery of discarded ideas

Resonetics logs function as **research notebooks**, not black-box telemetry.

---

## 7. What Reviewers Should Look For

Reviewers are encouraged to examine:

* consistency across similar inputs
* stability under re-evaluation
* absence of self-protective behavior
* clarity of rejection reasons

Single scores are meaningless in isolation.
**Patterns matter.**

---

## 8. What This Log Is Not

* Not a training metric
* Not a performance benchmark
* Not a truth score
* Not an AGI signal

It is a **structural diagnostic trace**.

---

## Summary

Resonetics experiment logs document:

* how ideas behave
* not what they claim

Reading them correctly means focusing on:
**structure, pressure, and evolution — not conclusions.**

