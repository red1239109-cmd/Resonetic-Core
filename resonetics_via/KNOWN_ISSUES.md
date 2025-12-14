
---

## KNOWN_ISSUES.md

```md
# Known Issues — Runtime Controls v1

## 1) “Energy” is a score, not physics
`compute_energy()` is a bounded weighted sum:
- This is a **control signal**, not “real energy”.
- Interpretation should remain pragmatic.

## 2) Threshold tuning is domain-specific
DEFAULT_THRESHOLDS are generic.
Different environments may need re-tuning:
- discrete gridworld vs continuous control
- noisy reward vs stable reward
- adversarial pressure vs benign pressure

## 3) Bubble/Collapse classification is heuristic
`classify_paradox()` uses hand-made rules:
- It can mislabel edge cases (e.g. high coherence but defensive).
- Consider logging confusion cases and adjusting rules.

## 4) Risk mapping is coarse
`instant_risk` uses fixed constants:
- collapse -> 0.95
- bubble -> 0.65
- creative_tension -> 0.25 + 0.50*tension
If you already compute a proper risk metric, replace this section.

## 5) Suppression assumes “multiplication makes sense”
`suppress_action_vector(action_vec, alpha)` scales the action by multiplication:
- Works for floats / numpy / torch / list
- For complex action structures (dict, nested tuples), you must implement a custom suppressor.

## 6) Survival mode directive is advisory
`survival_policy_switch()` returns a text directive.
It does NOT enforce policy by itself unless your environment applies it.

## 7) self_protecting signal quality matters
If `self_protecting` is guessed poorly, collapse detection becomes noisy.
Recommended:
- define a consistent detector in your upstream pipeline
- or set it always False until you can measure it reliably

