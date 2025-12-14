# Experiment Log Example — Runtime Controls v1

This is an example of how to log and interpret outputs from `apply_controls()`.

## Example loop (pseudo)

```python
controller = RiskEMAController(beta=0.90, min_alpha=0.15, max_alpha=1.0)

for t in range(T):
    # upstream signals (you compute these)
    state = ParadoxState(
        tension=..., coherence=..., pressure_response=...,
        self_protecting=..., confidence=...
    )

    base_reward = ...
    action_vec = ...

    out = apply_controls(
        state=state,
        base_reward=base_reward,
        controller=controller,
        action_vec=action_vec,
        action_space_hint="continuous"
    )

    # log
    log_row = {
        "t": t,
        "verdict": out["verdict"]["type"],
        "energy": out["verdict"]["energy"],
        "risk_instant": out["risk"]["instant"],
        "risk_ema": out["risk"]["ema"],
        "alpha": out["risk"]["damping_alpha"],
        "reward_base": out["reward"]["base"],
        "reward_shaped": out["reward"]["shaped"],
        "survival_forced": out["survival"]["forced"],
    }
    print(log_row)

