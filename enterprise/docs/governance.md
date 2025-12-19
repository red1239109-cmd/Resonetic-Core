# ⚖️ Governance & Constitution (Supreme Court)

This system is designed based on **"Kant's Categorical Imperative."**
No matter how important efficiency is, the system cannot violate the established **"Constitution."**

## 1. The Supreme Court
The `ActionApplicator` must pass through a `SupremeCourt.review()` before executing any action.
The Court has **VETO power**, and any rejection is immediately recorded in the timeline as an `action_vetoed` event.

## 2. Articles of Constitution

### 📜 Article 1: Schema Integrity
- **Rule:** Modification of parameters not defined in `KNOB_SCHEMA` is prohibited.
- **Purpose:** Prevent hackers or bugs from directly manipulating system outputs (e.g., `risk`, `loss`).

### 📜 Article 2: Stability Safety Lock
- **Rule:** In critical situations where stability (`stability`) is below `0.2`, **"High-Risk Operations (e.g., changing Learning Rate)" are prohibited.**
- **Exception:** Safe parameter adjustments (e.g., `dropout`) are allowed (Paradox Fix applied).
- **Purpose:** Prevent fatal system collapse caused by radical interventions during a critical state.

### 📜 Article 3: Radical Change Limit
- **Rule:** The `learning_rate` cannot exceed the absolute ceiling of `0.01`.
- **Purpose:** Prevent system divergence and runaway instability.

## 3. Policy Changes
Modifying this Constitution requires code-level changes (modifying `src/action.py`) and a mandatory **Git Commit Approval (Pull Request Review)** process.
It serves as an immutable law during runtime.
