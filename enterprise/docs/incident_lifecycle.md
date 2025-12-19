# 🔄 Incident Lifecycle Management

An Incident is not a failure of the system, but a **process of system evolution (Dialectics).**
We manage incidents through a strict 3-stage State Machine.

## 1. State Machine

### 🔴 OPEN (Thesis / Onset)
- **Trigger:** Detection of `risk > 0.8` or a `loss spike`.
- **Meaning:** "The system is currently unstable."
- **Action:** Call Operator (or Autopilot), Log `Anomaly` event.

### 🟠 MITIGATING (Antithesis / Crisis)
- **Trigger:** The first **Constitutional Action** is approved and applied.
- **Meaning:** "Intervention in progress. Monitoring required."
- **Action:**
    - `ActionApplicator` modifies parameters (Records `Before` -> `After` Diff).
    - `ActionEffectAnalyzer` tracks the trajectory for 10 steps.

### 🟢 RESOLVED (Synthesis / Conclusion)
- **Trigger:** The `stability` score remains above the target (e.g., 0.9) for N-steps.
- **Meaning:** "Situation normalized. Learning complete."
- **Action:**
    - `ActionEffectAnalyzer` issues a final verdict (`EFFECTIVE` vs `INEFFECTIVE`).
    - `PostmortemGenerator` compiles the entire timeline into a report.

## 2. Postmortem
When an incident is `RESOLVED`, the system automatically generates an **Autopsy Report** containing:
1. **Summary:** Start time, End time, Total duration.
2. **Timeline:** Chronological list of key events.
3. **Verdict:** Data-driven judgment on whether the actions taken were actually effective.

## 3. Playbook (Response Manual)
- **If VETOED (High LR):** The `SupremeCourt` blocks radical changes. Try smaller, incremental adjustments.
- **If Verdict is INEFFECTIVE:** Rollback the previous action or try a different strategy (e.g., adjusting Reality Weight).
