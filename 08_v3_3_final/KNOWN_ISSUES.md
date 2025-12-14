# KNOWN_ISSUES.md

# Known Issues (as of 2025-12-14)

## Gardener (resonetics_v3_3_2_final_with_patches.py)

### 1) Risk/Flow scale calibration
- **Symptom**: `Collapse_Risk` can saturate near 1.0 if weights/thresholds are too aggressive.
- **Impact**: verdict becomes mostly `collapse`, reducing usefulness.
- **Mitigation**:
  - lower `risk.w_entropy_grad` / `flow.w_flow`
  - raise `risk.collapse_th`
  - adjust `flow.eps` upward (e.g., 0.03~0.05) if flow becomes too noisy

### 2) Flow measurement is “system-level”, not “agent-level”
- **Symptom**: flow reflects brittleness of the grid dynamics, not per-agent behavior.
- **Impact**: good for collapse sensing, but not directly for per-agent action suppression unless you wire it.
- **Mitigation**: later add `type: "agent"` flow (policy sensitivity) or feed system flow into agent decision rules.

### 3) Torch device placement (CPU only in Gardener)
- **Symptom**: Agent brains run on CPU tensors; no device management.
- **Impact**: fine for small N, but GPU acceleration is not enabled.
- **Mitigation**: optional refactor to move brains + hidden states onto CUDA when available.

### 4) LSTM hidden-state lifetime
- **Symptom**: hidden state persists indefinitely, can accumulate bias.
- **Impact**: may cause “stale memory” behavior over long runs.
- **Mitigation**: reset hidden every N steps or when verdict is collapse.

### 5) Logging robustness
- **Symptom**: CSV writes buffered; abrupt kill may lose last buffer.
- **Impact**: last chunk of run missing.
- **Mitigation**: reduce `log_buffer_size`, or flush every N frames.

---

## Kubernetes Env (resonetics_k8s_v4_5_fixed.py)

### 1) Termination policy is harsh
- **Symptom**: any OOM ends episode.
- **Impact**: can bias learning toward overly conservative policies.
- **Mitigation**: change termination to allow limited OOM with penalty.

### 2) Reward shaping may require tuning per task
- **Symptom**: scale/clean reward balance can favor one action.
- **Mitigation**: tune `clean_cost`, `unit_cost`, and OOM penalties.

---

## Prophet Enterprise Kernel

### 1) Dropout affects Flow if computed in training mode
- **Fix recommended**: compute flow with `model.eval()` + `no_grad()`, then restore `model.train()`.
- If you applied the diff patch you shared, you’re good.

### 2) RiskPredictor input shape edge cases
- **Symptom**: `recent_error` shape mismatch.
- **Fix**: shape coercion to `(B,1)` (also in your diff).

### 3) “Structure period=3” is a design choice, not a theorem
- **Symptom**: over-regularization to multiples of 3 can harm fit depending on target distribution.
- **Mitigation**: expose `structure_period` in config and sweep it.


