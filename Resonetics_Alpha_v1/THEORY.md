# THEORY.md  
## Resonetics Alpha – Flow / Structure / Tension Formulation

This document describes the **mathematical formulation** underlying the
Resonetics Alpha v4.2 demo.

It intentionally avoids philosophical language where possible and focuses on:
- objective functions
- optimization dynamics
- interaction between competing constraints

This is a **theoretical note**, not a proof of optimality or general intelligence.

---

## 1. Problem Setting

We consider a model that predicts two quantities:

- **μ (mu)**: the primary decision / prediction
- **σ (sigma)**: the model’s self-reported uncertainty

Given:
- input \( x \in \mathbb{R} \)
- target \( y \in \mathbb{R} \)

The goal is **not** merely to minimize prediction error, but to study how
multiple, partially conflicting objectives interact during training.

---

## 2. Total Objective

The total loss is defined as a weighted sum of component losses:

\[
\mathcal{L}_{total} =
\sum_{i=1}^{8} \exp(-s_i)\,\mathcal{L}_i + s_i
\]

where:
- \( s_i \) are **learnable log-variance parameters**
- weighting follows uncertainty-based multi-task learning  
  (Kendall et al., 2018)

To prevent numerical instability:
\[
s_i = L \cdot \tanh(\tilde{s}_i / L)
\]

---

## 3. Loss Components

### L1 — Reality Gap

Standard mean squared error:

\[
\mathcal{L}_1 = (\mu - y)^2
\]

Encodes fidelity to observed data.

---

### L2 — Flow (Smoothness of Change)

Rather than enforcing a preferred value, L2 enforces **continuity**.

Using a finite-difference approximation:

\[
\mathcal{L}_2 =
\frac{(\mu(x+\varepsilon) - \mu(x))^2}{\varepsilon^2}
\]

This discourages abrupt local changes while remaining value-agnostic.

---

### L3, L4 — Reserved

Currently set to zero.
These slots are intentionally left for future constraints
(e.g., symmetry, invariance, monotonicity).

---

### L5 — Structural Potential

Attraction toward discrete structural manifolds defined by:

\[
\mu \approx k \cdot n,\quad n \in \mathbb{Z}
\]

Implemented as a smooth periodic potential:

\[
\mathcal{L}_5 = 1 - \cos\left(\frac{2\pi \mu}{k}\right)
\]

This avoids non-differentiable operators (e.g., rounding).

---

### L6 — Tension (Reality vs Structure Conflict)

Tension is defined **only when both**:
- prediction deviates from reality
- prediction deviates from structure

Let:
\[
g_R = (\mu - y)^2,\quad
g_S = \mathcal{L}_5
\]

Then:
\[
\mathcal{L}_6 =
\tanh(\alpha g_R)\cdot\tanh(\beta g_S)
\]

This formulation ensures:
- low tension if either constraint is satisfied
- high tension only under simultaneous failure

---

### L7 — Self-Consistency (EMA Teacher)

Temporal consistency with an exponential moving average (EMA) teacher:

\[
\mathcal{L}_7 = (\mu - \mu_{teacher})^2
\]

This stabilizes training and discourages oscillatory strategies.

---

### L8 — Humility (Uncertainty-Aware Likelihood)

Gaussian negative log-likelihood:

\[
\mathcal{L}_8 =
\frac{1}{2}\log(\sigma^2)
+ \frac{(\mu - y)^2}{2\sigma^2}
\]

Uncertainty is **smoothly bounded**:

\[
\sigma = \sigma_{min}
+ (\sigma_{max} - \sigma_{min})\cdot\sigma(\cdot)
\]

This prevents:
- overconfidence (\(\sigma \to 0\))
- evasion via infinite uncertainty

---

## 4. Optimization Dynamics

- All components are differentiable and bounded
- Loss weights adapt during training
- No fixed coefficients are required

The system often exhibits:
1. Early dominance of L1 (reality alignment)
2. Mid-stage competition between L1 and L5
3. Late-stage stabilization via L7 and L8

---

## 5. Interpretation Notes

- No loss term is assumed to be “correct”
- The system explores trade-offs rather than enforcing a single optimum
- Philosophical labels are **interpretive metaphors**, not claims

---

## 6. Limitations

- Finite-difference L2 is O(N)
- One-dimensional toy domain
- No generalization guarantees
- No claim of cognitive validity

This formulation is intended for **conceptual and experimental exploration only**.

---

## References

- Kendall, A., Gal, Y., & Cipolla, R. (2018).  
  *Multi-task learning using uncertainty to weigh losses.*  
  CVPR.
