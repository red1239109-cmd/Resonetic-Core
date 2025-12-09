# Paradox Refinement Engine

> **"Reasoning through Recursion."**
> A text refinement engine that translates **Sovereign Logic** (Physical/Structural Constraints) into linguistic meta-rules.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version](https://img.shields.io/badge/Version-1.0.0_Alpha-orange.svg)]()

## 🌌 Overview

The **Paradox Refinement Engine** is a module of Project Resonetics. It replaces standard "Chain-of-Thought" with a **"Recursive Critique Loop."**

Instead of blindly generating text, this engine evaluates its own output using a **Meta-Rule Layer**, which mirrors the 8-layer `SovereignLoss` used in numeric AI training. It stops refining only when the text achieves "Meta-Convergence" (Stability + Structure + Logical Consistency).

## 🧠 The Philosophy: Math to Language
This engine translates the numeric constraints of Resonetics into textual signals:

| Sovereign Layer (Numeric) | Meta-Signal (Text) | Description |
| :--- | :--- | :--- |
| **L1 (Gravity)** | **Similarity** | Ensures the refined text doesn't drift too far from the original intent. |
| **L3 (Phase Boundary)** | **Logic Flip Check** | Prevents critical logic violations (e.g., "All" $\to$ "Some"). |
| **L5 (Quantization)** | **Structure Score** | Enforces the "Rule of 3," Numbering, and clear formatting. |
| **L7 (Self-Consistency)** | **Stability** | Measures if the magnitude of changes is damping (converging). |
| **L8 (Humility)** | **Severity Weight** | Adjusts convergence threshold based on the AI's self-reported confidence. |

## ⚙️ Key Features

* **Recursive Refinement:** The engine critiques and rewrites the text iteratively (Loop: Critique $\to$ Refine $\to$ Check).
* **Dynamic Thresholding:** Implements "Simulated Annealing" for thought generation. It allows exploration in early iterations and enforces strict convergence in later stages.
* **Safety Mechanisms:** Automatically detects and rejects "Logic Flips" (e.g., changing "Never" to "Always").

## 🚀 Usage

### 1. Basic Setup
You need a Python function (`llm_fn`) that connects to your LLM (GPT, Claude, Local LLM, etc.).

```python
from paradox_engine import ParadoxRefinementEngine

# Define your LLM callback
def my_llm_callback(prompt: str) -> str:
    # Call your API here (e.g., OpenAI, Anthropic)
    return call_gpt4(prompt)

# Initialize the Engine
engine = ParadoxRefinementEngine(
    llm_fn=my_llm_callback,
    max_iterations=5,
    verbose=True
)
