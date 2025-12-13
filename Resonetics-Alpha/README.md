# Resonetics Transformer
A research prototype Transformer with resonance-biased attention and philosophy-grounded auxiliary losses, balanced via uncertainty weighting.

**License:** AGPL-3.0

## Overview
This repository contains a single-file reference implementation:

- **Resonance-biased attention:** phase-distance penalty added to attention scores
- **R-Grammar encoder:** projects hidden states into a 4D semantic space (S/R/T/G)
- **Boundary layers:** per-layer safety/consistency estimator
- **Stabilization:** residual damping driven by boundary scores
- **Aux losses:** (Plato / Heraclitus / Socrates) combined with **learnable uncertainty weighting** to avoid hand-tuned coefficients

> Note: Standard attention is **O(L²)** in memory/time. For long sequences, consider FlashAttention / sparse / sliding attention variants.

## Quickstart
### Requirements
- Python 3.10+
- PyTorch

### Run verification + demo step
```bash
python resonetics_transformer_v3_3_uncertainty.py

