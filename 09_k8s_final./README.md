# Resonetics K8s: The Smart Tensor Gardener

![Version](https://img.shields.io/badge/version-4.2-blue) ![Gymnasium](https://img.shields.io/badge/gymnasium-0.29-green) ![License](https://img.shields.io/badge/license-AGPL--3.0-red)

> **"A Reinforcement Learning Environment for Autonomous Kubernetes Cluster Maintenance."**

This project simulates a Kubernetes cluster as a **3D Tensor Grid** where an autonomous agent (The Gardener) learns to manage resources under strict budget constraints. It models the "Red Queen Effect" where system entropy (memory leaks, CPU load) constantly rises, and the agent must efficiently allocate resources to prevent collapse.

## 🌟 Key Features

* **3D Tensor Observation:**
    * **Ch 0 (CPU):** Random load fluctuations.
    * **Ch 1 (Memory):** Constant memory leaks (Red Queen Effect).
    * **Ch 2 (Priority):** Critical vs Non-critical nodes.
* **Smart Budget System:**
    * Scale-up actions are expensive.
    * The agent learns **Partial Scaling** when budget is low.
    * Penalizes inefficient cleaning actions.
* **Gymnasium Standard:**
    * Fully compliant with Gymnasium API (`reset`, `step` with seed).
    * Reproducible experiments.

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/red1239109-cmd/Resonetic-Cores.git](https://github.com/red1239109-cmd/Resonetic-Cores.git)
    cd Resonetic-Cores
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

Run the environment test to verify logic:

```bash
python resonetics_k8s_final.py

Action ID,Name,Description,Cost
0-3,Move,Move Up/Down/Left/Right,Time
4,Clean,Restart Pod (Reduce CPU/Mem),0.5 Budget
5,Scale,Global Scale-up (Cool down hotspots),5.0 per Node
