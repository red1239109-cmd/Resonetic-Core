# EXPERIMENT_LOG_EXAMPLE.md

# Experiment Log Example
A lightweight format to keep experiments reproducible and reviewable.
Dates are real calendar dates.

---

## 2025-12-14 | Gardener v3.3.2 + Risk/Verdict + Flow | exp:system-flow-ema

### Goal
Add “A-Version Flow” (system sensitivity to perturbation) and feed it into collapse risk.

### Change Summary
- Added `CONFIG["flow"]`:
  - eps (noise scale)
  - ema_alpha
  - w_flow (risk mix weight)
  - hot_th (hotspot threshold)
- Implemented `_compute_system_flow()`:
  - metrics: `total_entropy`, `hotspot_rate`
  - flow = (Δentropy_norm² + Δhotspot²)/eps²
  - EMA smoothing
- Mixed flow into risk:
  - `raw = raw + w_flow * flow`
- CSV schema updated:
  - added `Flow` column
- HUD updated:
  - shows `FLOW` and `VERDICT`

### Command
```bash
python resonetics_v3_3_2_final_with_patches.py

Outputs

resonetics_data.csv

Metrics to watch

Collapse_Risk should rise before obvious collapse behavior

Flow should spike during brittle phases (hotspot emergence / instability)

Verdict distribution should not be 100% collapse (if it is, re-tune thresholds)

Notes / Next

If Flow is too noisy: increase eps (0.03~0.05) or increase EMA smoothing.

Next step: connect verdict to reward shaping and action suppression at the agent level.

2025-12-14 | K8s Env v4.5 | benchmark:random-vs-heuristic
Result
random    | reward μ=-12.456 σ=4.821 | steps μ=312.4 | oom_rate=1.82 | crit_rate=0.68 | avg_load μ=0.512
heuristic | reward μ=+18.912 σ=3.215 | steps μ=456.8 | oom_rate=0.34 | crit_rate=0.12 | avg_load μ=0.378

Interpretation

Heuristic policy reliably stabilizes load and avoids OOM more often than random, confirming the environment is learnable.

2025-12-14 | Prophet v8.4.2 Kernel | exp:flow-eval-mode
Goal

Ensure Flow measurement is not dominated by dropout jitter.

Fix

Compute Flow in model.eval() under torch.no_grad(), then restore training mode.

Expected Effect

Flow becomes a meaningful smoothness signal instead of random training-mode noise.
