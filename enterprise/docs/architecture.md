# 🏗️ System Architecture

## 1. Design Philosophy
This system adheres to the **"Zero-Dependency"** and **"Event Sourcing"** principles.
It achieves enterprise-grade stability using only the file system and in-memory structures, without relying on complex databases or external infrastructure.

## 2. Core Modules (`src/`)
The brain of the system. Written in pure Python with no external dependencies.

| Module | Role | Analogy |
| :--- | :--- | :--- |
| **Kernel** | `action.py`, `stability.py` | CPU (Central Processing Unit) |
| **Memory** | `timeline.py`, `incident.py` | Hippocampus (Memory Store) |
| **Judiciary** | `SupremeCourt` (in `action.py`) | Constitutional Court (Overseer) |
| **Analyst** | `effect.py`, `postmortem.py` | Analyst (Reporting) |

## 3. Data Flow
All state changes occur in a **Uni-directional** flow:

1. **Observe:** Collect metrics (via `collect_metrics`).
2. **Detect:** Detect anomalies (`Anomaly`) → Register in `IncidentRegistry`.
3. **Judge:** Formulate an intervention plan (`ActionPlan`) → `SupremeCourt` Review.
4. **Act:** If constitutional, `ActionApplicator` modifies kernel values (Records Diff).
5. **Verify:** `ActionEffectAnalyzer` verifies the effect after N-steps.
6. **Report:** Upon resolution, `PostmortemGenerator` creates a retrospective report.

## 4. Persistence (Storage)
- **Log:** `runs/timeline.jsonl` (Append-only, Source of Truth).
- **Snapshot:** `runs/incidents.json` (Optional state snapshot).
- **Dashboard:** Flask reads the `jsonl` file in real-time to render views (No DB Query required).
