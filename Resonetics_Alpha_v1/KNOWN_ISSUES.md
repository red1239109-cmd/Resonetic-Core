# KNOWN_ISSUES.md
## Resonetics Alpha v4.2 – Known Issues & Limitations

This document lists **known technical and conceptual limitations** of the current
Resonetics Alpha v4.2 implementation.

These are **not bugs**, but constraints of a research prototype.

---

## 1. Flow Term (L2) Uses Finite Differences

**Issue**  
L2 (Flow / Heraclitus) is implemented via a finite-difference approximation:
\[
(\mu(x+\varepsilon)-\mu(x))^2 / \varepsilon^2
\]

This requires an additional forward pass per batch.

**Implications**
- O(N) overhead per step
- Not scalable to large models or long sequences

**Status**
- Acceptable for toy experiments
- Future versions may replace this with:
  - analytical Jacobian norms
  - spectral regularization
  - implicit smoothness constraints

---

## 2. Structure Potential Is Manually Defined

**Issue**  
L5 (Structure / Plato) uses a hand-designed periodic potential
with minima at multiples of a constant (k=3).

**Implications**
- Structure is imposed, not discovered
- Choice of k is arbitrary and task-dependent

**Status**
- Intentional for demonstration
- Future work may:
  - learn structural manifolds
  - infer discrete attractors from data

---

## 3. Tension Term Is Heuristic by Design

**Issue**  
L6 (Tension) is defined as a bounded product of reality-gap and structure-gap.

This formulation is heuristic and not derived from a physical or probabilistic model.

**Implications**
- Other tension formulations may behave differently
- No guarantee that this captures all meaningful conflicts

**Status**
- Chosen for interpretability and stability
- Alternative formulations are possible and encouraged

---

## 4. Uncertainty Weighting Assumes Task Independence

**Issue**  
Uncertainty-based loss weighting (Kendall et al.) implicitly assumes
relative independence between loss terms.

In this system, losses are intentionally coupled (e.g., L5 influences L6).

**Implications**
- Weight adaptation may converge slowly
- Certain loss terms may dominate temporarily

**Status**
- Empirically stable in toy settings
- Not theoretically analyzed for strongly coupled objectives

---

## 5. Toy Problem Domain

**Issue**
- One-dimensional input
- Single scalar output
- Synthetic conflict scenario (Reality = 10 vs Structure = {9,12})

**Implications**
- Results do not generalize
- No performance claims are meaningful

**Status**
- Deliberate simplification
- Used to visualize optimization dynamics, not to benchmark models

---

## 6. No Guarantees of Convergence or Optimality

**Issue**
The system explores trade-offs between conflicting constraints.

There is:
- no proof of convergence
- no optimality guarantees
- no claims of stability beyond observed behavior

**Status**
- Expected for exploratory research
- Monitoring and visualization are required

---

## 7. Not a Cognitive or Linguistic Model

**Issue**
Despite philosophical terminology, this system does **not** model:
- human reasoning
- language understanding
- cognition or consciousness

**Status**
- Philosophy is used as an interpretive layer only
- No cognitive claims are made

---

## Summary

Resonetics Alpha v4.2 is an **experimental sandbox**, not a finished system.

Its value lies in:
- explicit separation of conflicting objectives
- inspectable optimization dynamics
- clarity about limitations

Any use beyond exploratory research is **out of scope**.
