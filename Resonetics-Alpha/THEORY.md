# THEORY.md  
Resonetics Transformer — Theoretical Foundations

## Scope
This document formalizes the mathematical structure behind the Resonetics Transformer.

It focuses on:
- auxiliary loss formulation
- uncertainty-based weighting
- resonance-biased attention
- boundary-driven stabilization

No philosophical background is required to read this document.
All terms correspond to explicit tensor operations.

---

## 1. Notation

Let:
- \( x \in \mathbb{R}^{B \times L \times D} \) be hidden states
- \( Q, K, V \in \mathbb{R}^{B \times H \times L \times d_h} \) be attention projections
- \( \phi \in \mathbb{R}^{B \times H \times L} \) be learned phase values
- \( \sigma \) denote sigmoid
- \( \mathcal{L} \) denote loss terms

---

## 2. Resonance-Biased Attention

Standard scaled dot-product attention:

\[
\text{Attn}(Q,K,V)
= \text{softmax}\left( \frac{QK^\top}{\sqrt{d_h}} \right)V
\]

Resonetics introduces a **phase-distance penalty**:

\[
\Delta\phi_{ij} = (\phi_i - \phi_j)^2
\]

\[
\text{scores}_{ij}
= \frac{Q_i K_j^\top}{\sqrt{d_h}}
- \lambda \cdot \Delta\phi_{ij}
\]

where:
- \( \lambda \) is a learnable resonance scale
- phase values are bounded via \( \tanh \)

This biases attention toward **phase-consistent tokens**,
reducing abrupt representational jumps.

> Note: Memory complexity remains \( O(L^2) \).  
> For long sequences, blockwise or sparse variants are recommended.

---

## 3. R-Grammar Projection

Hidden states are projected into a 4D semantic coordinate:

\[
g = \sigma(W_g x) \in [0,1]^4
\]

This vector does **not** supervise meaning.
It provides a **low-dimensional semantic tension signal**.

The projection is used to modulate activations:

\[
x' = x \cdot (1 + \alpha \cdot \text{mean}(g))
\]

where \( \alpha \ll 1 \).

---

## 4. Boundary Layers (Shock Estimation)

Each transformer block includes an independent boundary estimator:

\[
b = \sigma(f_{\text{boundary}}(x)) \in (0,1)
\]

Boundary scores estimate **internal instability**, not correctness.

Damping is applied multiplicatively:

\[
x \leftarrow x \cdot (0.3 + 0.7b)
\]

Properties:
- continuous
- differentiable
- always active (train & eval)

This prevents runaway amplification without hard clipping.

---

## 5. Auxiliary Losses

Resonetics uses **multiple auxiliary losses**, each targeting a distinct failure mode.

### 5.1 Plato Loss — Structural Anchoring

Encourages discrete, stable representations.

Example form:

\[
\mathcal{L}_{\text{Plato}}
= \left\| x - \text{round}\left(\frac{x}{k}\right) \cdot k \right\|^2
\]

This penalizes excessive fragmentation
without enforcing strict quantization.

---

### 5.2 Heraclitus Loss — Phase Smoothness

Encourages continuity across sequence positions:

\[
\mathcal{L}_{\text{Heraclitus}}
= \mathbb{E}_{i,j}\left[\sin^2\left(\frac{2\pi}{k}(x_i - x_j)\right)\right]
\]

This suppresses sharp phase discontinuities
while allowing controlled change.

---

### 5.3 Socrates Loss — Uncertainty Calibration

Let:
- \( \delta \) be a divergence or shock measure
- \( \tau \) be a soft threshold
- \( s \) a smoothness parameter

Target uncertainty:

\[
u^* = \sigma\left(\frac{\delta - \tau}{s}\right)
\]

The model learns to estimate its own epistemic uncertainty,
used to regulate constraint strength.

---

## 6. Uncertainty-Weighted Loss Aggregation

Instead of fixed coefficients:

\[
\mathcal{L}_{\text{aux}} = \sum_i w_i \mathcal{L}_i
\]

Resonetics uses **learned uncertainty** \( \sigma_i \):

\[
\mathcal{L}_{\text{total}}
= \sum_i \left(
\frac{1}{2\sigma_i^2}\mathcal{L}_i
+ \frac{1}{2}\log \sigma_i^2
\right)
\]

This formulation:
- removes hand-tuned weights
- penalizes overconfidence
- balances exploration vs convergence

This is a direct application of heteroscedastic uncertainty modeling.

---

## 7. Optimization Behavior

Combined effects:
- early training: higher uncertainty → looser constraints
- late training: lower uncertainty → tighter structure
- boundaries dampen instabilities locally
- resonance bias reduces attention noise

No single component enforces correctness.
Stability emerges from **interacting soft constraints**.

---

## 8. Failure Modes & Limitations

Known limitations:
- \( O(L^2) \) memory in attention
- auxiliary losses may conflict if poorly scaled
- boundary scores are estimators, not guarantees

This system trades raw performance for **controlled dynamics**.

---

## 9. Summary

Resonetics Transformer is defined by three principles:

1. **Bias attention by internal consistency**
2. **Stabilize representations via soft boundaries**
3. **Replace fixed heuristics with learned uncertainty**

All components are differentiable.
All constraints are measurable.
No rule is absolute.

This document describes the system as it is,
not as an aspiration.
