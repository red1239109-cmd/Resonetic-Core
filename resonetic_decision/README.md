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

Usage Example

import asyncio
from src.engine import ResoneticEngineV3, Option, Evidence, Criterion

async def main():
    engine = ResoneticEngineV3()
    
    question = "Choose a backend framework for a fintech startup."
    
    options = [
        Option("A", "Django", "Battery-included, mature ecosystem"),
        Option("B", "FastAPI", "Modern, high-performance async"),
    ]
    
    evidence = [
        Evidence("A", "Proven stability in banking sectors for 10+ years", 1.0),
        Evidence("A", "Slightly slower performance compared to Go/Rust", -0.3),
        Evidence("B", "Type hints prevent many logical errors", 1.0),
        Evidence("B", "Newer ecosystem implies potential hidden risks", -0.5),
    ]
    
    # Run Decision
    result = await engine.decide(question, options, evidence)
    
    print(f"🏆 Winner: {result.chosen} (Conf: {result.confidence:.3f})")
    print(f"📉 Tension: {result.steps[-1].metrics['tension']:.3f}")
    print(f"💡 Reason: {result.reason}")

if __name__ == "__main__":
    asyncio.run(main())

    🧮 Mathematical Integrity

    Tension Variance CorrectionIn v3.5.1, we corrected the variance normalization factor.For three values $\{a, b, c\} \in [0, 1]$, the theoretical maximum population variance occurs at the extremes (e.g., $\{0, 0, 1\}$).$$ \text{Max Variance} = \frac{(0 - 1/3)^2 + (0 - 1/3)^2 + (1 - 1/3)^2}{3} = \frac{2}{9} \approx 0.222 $$Previous heuristics used $1/6$, which led to overflow in extreme polarization. Resonetics now strictly adheres to this bound.

    🗺️ Roadmap
v4.0: Integration with LLM agents for automated evidence gathering.

v4.1: Multi-agent debate simulation (Arena Mode).

Dashboard: Web-based UI using Streamlit for real-time visualization of the "Resonance Loop".
