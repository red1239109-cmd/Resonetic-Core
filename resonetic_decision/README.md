# 🏛️ Resonetics: The Philosophically Grounded AI Decision Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-AGPL--3.0-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

> *"Logic flows like water, but tension holds the structure."*

**Resonetics** is a deterministic, explainable AI decision-making engine. Unlike "Black Box" LLM reasoning, Resonetics combines **High-Dimensional Vector Semantics (SBERT)**, **Physics-based Resonance Models**, and **Statistical Tension Analysis** to provide transparent and mathematically rigorous decisions.

It simulates a debate between three philosophical stances—**Rationalist, Empiricist, and Skeptic**—to calculate not just the *best option*, but the *cognitive tension* involved in choosing it.

---

## 🌟 Key Features

### 1. 🧠 Multi-Perspective Architecture
Instead of a simple similarity search, the engine evaluates evidence through three distinct philosophical lenses using **Zero-Shot Semantic Anchors**:
* **🔵 Rationalist:** Focuses on logic, consistency, and axioms.
* **🔴 Empiricist:** Focuses on data, history, and observable facts.
* **⚫ Skeptic:** Focuses on risk, flaws, and uncertainty.

### 2. ⚡ Dynamic Tension Modeling
We quantify the "cognitive dissonance" of a decision using a statistically normalized formula.
* **Physics Tension:** The conflict between "Desire (Pull)" and "Danger (Risk)".
* **Philosophical Tension:** The variance between the three philosophical perspectives.
* **Corrected Math:** Utilizes a theoretically proven normalization factor (`2/9`) for variance in bounded intervals `[0,1]`, ensuring precise tension measurement.

### 3. 🛡️ Enterprise-Grade Reliability
* **Defensive Programming:** Strict validation via `__post_init__` and type safety.
* **Pure PyTorch Implementation:** Fully vectorized operations with no CPU/GPU context switching overhead.
* **Streaming Support:** Buffered generator pattern for real-time dashboards without data inconsistency glitches.

---

## 🏗️ Architecture

```mermaid
graph TD
    User[User Input] --> Cortex[Semantic Cortex (SBERT)]
    Cortex --> Static[Static Analyzer]
    
    subgraph "Phase 1: Static Analysis"
        Static --> Coherence[Weighted Coherence]
        Static --> Risk[Rule-based Risk & Keywords]
        Static --> Perspectives[Perspective Anchors (R/E/S)]
    end
    
    subgraph "Phase 2: Resonance Loop (Dynamic)"
        Physics[Physics Engine] --> Belief[Belief State (EMA)]
        Belief --> Pull[Sigmoid Pull Strength]
        Perspectives --> Tension[Dynamic Tension Calculation]
        Governor[Adaptive Governor] --> Threshold{Convergence Check}
    end
    
    Static --> Physics
    Pull --> FinalScore
    Tension --> Governor
    Threshold -- Continue --> Physics
    Threshold -- Stop --> Result[Final Decision]
