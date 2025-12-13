# KNOWN_ISSUES.md  
Resonetics Transformer — Known Issues & Limitations

## Scope
This document lists **known limitations and trade-offs** of the Resonetics Transformer.

These are not implementation bugs.
They are consequences of explicit design choices prioritizing
**stability, interpretability, and constrained dynamics** over raw throughput.

---

## 1. Attention Complexity (O(L²))
The resonance-biased attention mechanism inherits the standard
quadratic memory and compute complexity of scaled dot-product attention.

- Phase-distance penalties are applied pairwise across sequence positions.
- Long sequences may exceed practical memory limits.

**Status:** Known limitation  
**Mitigation:** Block-wise attention, sparse attention, or FlashAttention variants (see Roadmap)

---

## 2. Boundary Scores Are Estimators, Not Guarantees
Boundary layers estimate internal instability.
They do **not**:
- detect correctness
- verify truth
- enforce hard safety rules

A low boundary score indicates potential instability,
not semantic failure.

**Status:** By design  
**Implication:** Boundary outputs must be interpreted probabilistically

---

## 3. Auxiliary Loss Interactions Can Be Non-Linear
Plato, Heraclitus, and Socrates losses regulate different failure modes.
When combined, their gradients may interact non-linearly.

- In some regimes, losses may temporarily conflict.
- This is the primary reason uncertainty-based weighting is required.

**Status:** Known trade-off  
**Mitigation:** Learned uncertainty parameters adapt relative influence over time

---

## 4. Sensitivity to Uncertainty Initialization
Uncertainty parameters influence convergence speed and rigidity.

- Poor initialization may slow early training.
- Overconfident initialization may cause premature constraint tightening.

**Status:** Known sensitivity  
**Mitigation:** Conservative initialization and warm-up schedules

---

## 5. Performance Is Not the Primary Objective
This architecture is **not optimized for benchmark dominance**.

- Small datasets or short training runs may show no improvement
  over standard transformers.
- Benefits are expected primarily in stability-sensitive or long-horizon settings.

**Status:** Intentional design choice

---

## 6. Boundary Damping Is Heuristic
Boundary-based damping is a continuous heuristic mechanism.

- It reduces amplification of unstable activations.
- It does not provide formal safety guarantees.

**Status:** Heuristic, empirically motivated  
**Implication:** Should not be interpreted as a safety system

---

## 7. Limited Empirical Validation
Current experiments are limited to:
- small-scale runs
- synthetic or simplified tasks

Large-scale or real-world benchmarks have not yet been evaluated.

**Status:** Open  
**Mitigation:** Planned ablation and scaling studies

---

## Summary
The Resonetics Transformer trades:
- maximal performance
for
- controlled representational dynamics

All limitations listed here are **known, monitored, and intentional**.
They define the scope in which this system should be evaluated.
