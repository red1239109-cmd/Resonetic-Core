# Known Issues & Limitations

This document lists known weaknesses and open problems.

---

## Computational Cost

- Full attention remains O(L²)
- Benchmarks are intentionally small-scale

Mitigation:
- Focus on dynamics, not throughput
- Future sparse variants planned

---

## Hyperparameter Sensitivity

- Resonance strength and entropy thresholds interact
- Poor settings can suppress learning

Mitigation:
- Ablation defaults provided
- Logs expose failure modes explicitly

---

## Generalization Claims

- Experiments focus on MNIST-scale data
- No claims about large-scale superiority

Mitigation:
- Results are framed as *behavioral evidence*, not performance benchmarks

---

## Interpretability Risk

Some internal signals are abstract and require care to interpret.

Mitigation:
- Auxiliary metrics are always paired with controls
- No single signal is trusted alone

---

## Intentional Non-Goals

- Production readiness
- AGI inference
- Biological realism

---

## Why These Issues Are Documented

Because hiding them would be worse.
