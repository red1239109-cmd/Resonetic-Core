# Resonant Navier–Stokes (Resonetics Demo)

A **lightweight visualization prototype** that maps a 2D velocity field into a latent “phase field” and produces:
1) a **proxy saliency map** (untrained) and  
2) a **resonance feature field** (untrained)

> This repository is currently a **model-assembly + visualization demo**.
> It is **NOT** a validated physics solver and does **NOT** claim scientific discovery without training + evaluation.

## What this is
- A clean PyTorch pipeline to test:
  - tensor shapes & data flow
  - dynamic internal-field resizing
  - stable rendering & output export

## What this is not (yet)
- Not a trained detector
- Not an unsupervised “discovery” system (yet)
- Not a Navier–Stokes simulator

## Outputs
Running the script generates:
- `resonant_fluid_result.png` with:
  - input vortex field (quiver)
  - proxy singularity map (untrained)
  - resonance feature visualization (untrained)

## Quickstart
```bash
pip install torch numpy matplotlib
python resonant_fluid_demo.py

Roadmap (to make it real)
Option A — Weak supervision (fastest)

Train the proxy map to correlate with a physics-derived target such as vorticity magnitude.

Option B — Self-supervised consistency (closest to “unsupervised”)

Enforce invariance/equivariance under transformations (rotation/translation) and consistency across perturbed inputs.
