# Experiment Log

This file records *reproducible* benchmark runs for EtherVAE / Resonetics variants.

## Conventions
- **VAE-standard**: beta=1.0, no inference-time latent modulation
- **β-VAE**: beta>1.0, trained baseline
- **ether / resonetics**: inference-time latent modulation on the same trained VAE

## Summary Table (latest runs)

| Date | Commit | Seed | Epochs | Beta | Mode | Val BCE ↓ | Val KL | Smooth ↓ | Notes |
|------|--------|------|--------|------|------|----------:|-------:|---------:|------|
| 2025-12-13 | abc123 | 42 | 12 | 4.0 | β-VAE | 0.0981 | 18.2 | 0.0039 | baseline disentanglement pressure |
| 2025-12-13 | abc123 | 42 | 12 | 1.0 | resonetics | 0.0932 | 12.8 | 0.0035 | entropy-aware modulation improves smoothness |

## Notes
- Smoothness metric is local reconstruction sensitivity under small latent perturbations.
- Full raw logs: `runs/*.jsonl`
