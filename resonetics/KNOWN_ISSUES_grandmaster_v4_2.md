# Known Issues (resonetics_alpha_grandmaster_v4_2_flow_structure_tension.py)

This document lists *known limitations* of the current **Grandmaster v4.2** script.
It is written for reviewers/testers so they can judge results appropriately and know what to try next.

> Scope: This is a **didactic research prototype** (single-file demo). It is not a claim of AGI and it is not a production training pipeline.

---

## 1) Objective design / loss semantics

### 1.1 L2 (Flow) is implemented via finite-difference gradient
**What it does now**
- `L2` estimates a “flow penalty” using `dL1/dmu` via finite differences: `L1(mu+eps) - L1(mu)`.
- This couples L2 tightly to L1 and the chosen `eps`.

**Why it matters**
- Sensitive to `eps`, scale of targets, and dtype (especially fp16/bfloat16).
- When `mu` is close to target, gradients shrink and L2 may go near-zero even if “flow” should still be meaningful.

**Next steps**
- Replace finite-difference with an analytic proxy (e.g., `|mu - teacher_mu|` as flow mismatch, or `|∂mu/∂x|` if you introduce a meaningful input).
- Or compute gradients with autograd in a controlled way (but beware higher-order cost).

### 1.2 L5 (Structure) and L6 (Tension) still share a dependency on the snap target
**What it does now**
- `L5`: distance-to-structure using a **soft snap** to multiples of 3.
- `L6`: tension as a bounded (tanh) penalty between **reality target** and the same snap target.

**Why it matters**
- Better than the older “hard snap + direct sin” design, but L6 is still downstream of the *same* structural anchor.
- In some regimes, the optimizer can reduce both by moving `mu` toward a compromise that’s neither “truth” nor “structure” in an interpretable way.

**Next steps**
- Make L6 depend on *two competing proposals* (e.g., “reality proposal” vs “structure proposal”) rather than reusing the snap target.
- Alternatively, treat structure as a *prior* (regularizer) and keep tension as an *energy* computed from their mismatch, with its own schedule.

### 1.3 Hyperparameter arbitrariness (scale/temperature/weights)
Current knobs include:
- `eps` for finite differences (L2)
- `temp` (soft snap softness)
- `tension_scale`, `tension_weight`
- EMA decay schedule
- log-var clamp range

**Why it matters**
- These are not identified by theory in this file; they must be tuned empirically and can change conclusions.

**Next steps**
- Add a config block and log all hyperparameters into the run output.
- Add a small grid search or sensitivity sweep.

---

## 2) Training loop limitations

### 2.1 Toy data (no meaningful input-output structure)
**What it does now**
- `x` is random noise; `target` is constant (10.0).
- The model can “solve” this by learning a constant function, which is not evidence of general reasoning.

**Next steps**
- Introduce a family of targets (multiple regimes) or a learnable mapping `target = f(x)` where `x` has structure.
- Add held-out validation tasks.

### 2.2 No reproducibility guarantees
- Random seeds are not set (torch / numpy).
- Device-dependent nondeterminism can change outcomes.

**Next steps**
- Add `torch.manual_seed`, `np.random.seed`, and deterministic flags (optional).

### 2.3 Metrics are primarily internal
You print `mu`, `sigma`, and the learned loss weights, but:
- There is no explicit evaluation metric besides proximity to a constant target.
- No calibration checks for `sigma` (uncertainty).

**Next steps**
- Add calibration tests (e.g., NLL vs empirical error).
- Track per-loss component curves over time (CSV logging).

---

## 3) Numerical / engineering concerns

### 3.1 Potential instability from mixed precision
If users run AMP / fp16:
- finite differences (L2) can underflow/overflow
- `exp(-log_var)` can blow up without careful clamping

**Next steps**
- Explicit dtype notes, and optionally keep loss computations in fp32.

### 3.2 Plotting code uses fixed aesthetics and assumes a GUI backend
- Some environments (headless servers) can fail on `plt.show()`.

**Next steps**
- Default to saving figures only; optionally guard `show()`.

---

## 4) Interpretation caveats (important for reviewers)

### 4.1 “Value hierarchy” != moral truth
The learned weights `exp(-log_vars)` are *optimization weights*, not philosophical proof.

### 4.2 This is not a benchmark
Without baselines, controls, and ablations, it’s a demo, not a scientific claim.

---

## 5) Roadmap suggestions (small, high-value)

1. **Ablations**
   - Toggle L2/L5/L6/L7/L8 one at a time and compare curves.
2. **Sensitivity sweep**
   - Vary `temp`, `tension_scale`, `eps`, EMA schedule.
3. **Structured dataset**
   - Make `target = 3 * round(f(x)/3)` vs `target = f(x)` and see where the model settles.
4. **Logging**
   - Write CSV: epoch, mu, sigma, L1..L8, weights, decay.
