# Experiment Log – How to Read This

This document explains how to interpret experiment logs produced by the Resonetics benchmarks.

---

## Why Logs Matter Here

Resonetics does not rely on single scalar metrics alone.
Many effects (smoothness, entropy response, stability) emerge only across *patterns*.

Logs are therefore structured to support **pattern inspection**, not cherry-picked numbers.

---

## Log Structure

Each experiment log contains:

### 1. Configuration
- Model variant
- Enabled / disabled components
- Random seed
- Dataset and training length

This ensures reproducibility.

---

### 2. Core Metrics

Typical metrics include:
- Reconstruction loss
- KL divergence
- Local latent smoothness (mean ± std)
- Entropy-response curves

Interpretation tip:
> Lower is not always “better”. Consistency and trend shape matter more than absolute minima.

---

### 3. Comparative Sections

Logs are grouped by **mode comparison**, e.g.:

- standard vs ether
- ether vs resonetics

These comparisons are the *primary evidence*, not standalone scores.

---

### 4. Statistical Tests

Where applicable:
- Paired t-tests
- Effect size (Cohen’s d)

A statistically significant but unstable effect is **not considered a win**.

---

## What to Look For

✔ Smooth transitions across entropy levels  
✔ Reduced variance without collapse  
✔ Effects that survive ablation  
✘ Single-run spikes  
✘ Improvements that vanish under control conditions  

---

## Intended Use

These logs are meant to support:
- Peer review
- Ablation reasoning
- Design critique

They are **not marketing artifacts**.
