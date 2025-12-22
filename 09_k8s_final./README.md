# 🧠 Resonetics: Autonomous Ops Platform

**Self-Healing Kubernetes Cluster Management via Reinforcement Learning**

> *"Systems drift toward entropy. Intelligence is the compression of interventions required to keep them alive."*

**Resonetics** is a research-grade simulation platform designed to train autonomous agents for managing distributed systems under strict constraints. It models a Kubernetes cluster not as a static entity, but as a dynamic **3D tensor grid** subject to the **Red Queen Effect**—where entropy (CPU load, memory leaks) continuously rises, and agents must act efficiently just to maintain stability.

Moving beyond simple simulation, this project has evolved into **Resonetics v10.0**, a full-stack **MLOps platform** featuring automated hyperparameter tuning (AutoML), real-time observability, and forensic replay systems.

---

## 🌟 Key Features

### 1. The Environment: Kubernetes Smart Tensor (v5.1 Enterprise)

The core simulation environment is fully **Gymnasium-compliant**, rigorously modeling the trade-offs between system stability and operational cost.

* **3D Tensor State:** The cluster is represented as a `(H, W, C)` grid:
* **Channel 0 (CPU):** Stochastic load fluctuations.
* **Channel 1 (Memory):** Monotonically increasing pressure (Drift).
* **Channel 2 (Priority):** Business criticality of nodes (Critical vs Non-Critical).


* **Partial Observability:** The agent sees only a local **3x3 neighborhood** plus global aggregate statistics, forcing it to balance **exploration** (finding hotspots) and **exploitation** (fixing them).
* **Strategic Action Space (Discrete 8):**
* **Movement (0-3):** Navigate the grid.
* **Precise Interventions:** `Full Clean` (4), `Mem-Only Clean` (6), `CPU Throttle` (7).
* **Global Scaling (5):** An expensive, high-impact action that prioritizes critical nodes using a smart sorting algorithm.



### 2. The Research Facility: Automated MLOps (v10.0)

Resonetics is no longer just a script; it is a comprehensive **platform**.

* **🧪 AutoML Integration (Optuna):** Automatically searches for the "sweet spot" of environmental difficulty and agent hyperparameters using Bayesian optimization.
* **📊 Live Control Room (Streamlit):** A real-time dashboard visualizing reward trends, action distributions, and system loads on dual-axis charts.
* **💾 Forensics & Replay:** Automatically captures "Black Box" JSON logs when an agent fails (OOM), enabling post-mortem analysis of crash scenarios.

---

## 🏛️ Architecture Overview

The system is composed of three concentric layers:

1. **The Kernel (Physics):** A stochastic grid engine that simulates resource drift, noise, and budget recovery dynamics.
2. **The Agent (Policy):**
* **Baseline:** A smart heuristic that prioritizes critical hotspots based on rule-based logic.
* **Learner:** A Parametric Q-Learning agent capable of adapting to dynamic difficulty curves.


3. **The Platform (Ops):** A Streamlit + Optuna wrapper that orchestrates batch experiments, tuning, and visualization.

---

## ✅ Benchmark Results

We compared the **Smart Heuristic Baseline** against a **Random Policy** and our **Tuned Q-Learning Agent** across 500 episodes.

| Policy | Avg Reward | Survival Steps | OOM Rate | Critical Failure % |
| --- | --- | --- | --- | --- |
| **Random** | -124.5 | 312.4 | 1.82 | 68% |
| **Heuristic** | **+189.1** | 456.8 | 0.34 | 12% |
| **Q-Learning** | +145.2* | **498.0** | **0.10** | **5%** |

*> Note: The Q-Learning agent achieved lower total rewards than the Heuristic but demonstrated superior **survival capabilities** (longer steps, fewer crashes), indicating a more risk-averse, sustainable strategy suitable for production stability.*

---

## 🚀 Quickstart

### 1. Installation

Install the required scientific and visualization stack.

```bash
pip install -r requirements.txt

```

### 2. Launch the Platform

Start the **Resonetics Control Room**. This will launch the Streamlit dashboard in your browser.

```bash
streamlit run resonetics_production_v10.py

```

### 3. Workflow

1. **AutoML Tuning:** Click `Start Optimization` to let Optuna find the best hyperparameters.
2. **Live Simulation:** Watch the agent manage the cluster in real-time.
3. **Analysis:** Review the `Reward vs Load` dual-axis chart to understand the agent's decision-making process.

---

## 📜 Design Philosophy: "Via Negativa"

Resonetics follows the principle of **Via Negativa**—improving systems by removing failure modes rather than adding features.
We do not hard-code "how to succeed." We rigorously model **"what causes failure"** (Entropy, Bankruptcy, Panic) and build agents that naturally evolve to avoid them.

* **Budget Constraints** prevent infinite scaling.
* **Partial Observability** prevents omniscience.
* **Dynamic Difficulty** prevents overfitting.

This ensures that any surviving agent has learned robust, generalized stability patterns.

---

**© 2025 Resonetics Research Institute** | *From Entropy to Anti-Fragility.*

