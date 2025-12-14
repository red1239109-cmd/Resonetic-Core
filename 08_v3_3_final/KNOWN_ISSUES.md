# KNOWN_ISSUES.md
Known limitations, design trade-offs, and unresolved questions in the Resonetics project.

This document is intentional.  
Every item listed here is either:
- a conscious design choice, or
- an open problem kept visible on purpose.

---

## 1) This Is Not Intelligence (By Design)

**Issue:**  
Resonetics does not learn tasks, optimize rewards, or converge toward a goal in the traditional ML sense.

**Why it exists:**  
The project explicitly avoids task optimization. Its purpose is to **measure and regulate contradiction**, not to solve problems.

**Impact:**  
- No benchmark-style “performance improvement”
- No claim of general intelligence
- Cannot be compared directly to RL agents, LLMs, or planners

**Status:**  
Intentional. Not a bug.

---

## 2) Verdict Thresholds Are Hand-Tuned

**Issue:**  
Thresholds for `collapse`, `bubble`, and `creative_tension` are manually specified.

**Why it exists:**  
Automatic threshold learning would collapse the system into self-justifying behavior.  
Fixed thresholds keep the logic **auditable and falsifiable**.

**Impact:**  
- Requires human judgment to adjust
- Sensitivity may vary across domains

**Status:**  
Known limitation. Possible future work: domain-specific presets (not self-learning).

---

## 3) Energy Is a Structural Heuristic

**Issue:**  
“Energy” is not a physical quantity and not guaranteed to correlate with usefulness.

**Why it exists:**  
Energy is a **structural signal**, not a utility function.  
It measures balance between tension, coherence, and pressure response.

**Impact:**  
- High energy ≠ correctness
- Low energy ≠ uselessness

**Status:**  
Working as intended. Misuse as a reward proxy is discouraged.

---

## 4) Risk EMA Can Over-Suppress Exploration

**Issue:**  
When risk EMA rises quickly, action damping may become overly conservative.

**Why it exists:**  
Safety is prioritized over exploration near collapse states.

**Impact:**  
- Reduced behavioral diversity
- Possible stagnation in edge cases

**Mitigation:**  
Tune `risk_ema_beta`, `min_alpha`, or temporarily disable suppression in controlled experiments.

**Status:**  
Trade-off accepted.

---

## 5) Survival Switch Is Heuristic, Not Optimal

**Issue:**  
Near-collapse detection uses simple rules (energy + coherence + pressure).

**Why it exists:**  
Collapse is treated as a **failure mode**, not an optimization target.

**Impact:**  
- May trigger survival too early
- Or fail to trigger in rare edge cases

**Status:**  
Known limitation. Chosen over opaque learned safety policies.

---

## 6) No Internal Memory Across Runs

**Issue:**  
The system does not persist learning across executions.

**Why it exists:**  
Persistent memory risks narrative lock-in and self-confirmation.

**Impact:**  
- No long-term accumulation of behavior
- Requires explicit experiment logging

**Status:**  
Intentional. External logs are the memory.

---

## 7) Not Immune to Adversarial Inputs

**Issue:**  
Carefully crafted paradox states could exploit threshold boundaries.

**Why it exists:**  
Any rule-based system has boundary conditions.

**Impact:**  
- Possible oscillation near verdict edges
- Requires careful interpretation of results

**Status:**  
Documented and accepted. This is a measurement tool, not a security system.

---

## 8) Human Interpretation Is Required

**Issue:**  
Verdicts and actions do not explain “meaning” — only structure.

**Why it exists:**  
Resonetics refuses to hallucinate intent or semantics.

**Impact:**  
- Results must be read, not trusted blindly
- Reviewer judgment is mandatory

**Status:**  
Core philosophical stance.

---

## 9) Experimental Scope Is Narrow

**Issue:**  
Resonetics is tested on controlled, abstract environments.

**Why it exists:**  
The project explores *structure before scale*.

**Impact:**  
- Not validated in real-world decision systems
- Transferability is unknown

**Status:**  
Open research direction.

---

## 10) No Claim of Emergence

**Issue:**  
The system may appear “creative” in behavior or output.

**Why it exists:**  
Structured tension can look like creativity.

**Clarification:**  
Resonetics does **not** claim emergence, agency, or understanding.

**Status:**  
Firmly rejected as a goal.

---

## Final Note

These issues are not hidden because Resonetics is not selling certainty.

It is a tool for **seeing where systems break, inflate, or deserve to continue**.

If a limitation makes you uncomfortable, that discomfort is part of the measurement.
