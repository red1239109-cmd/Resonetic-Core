# 🧠 Resonetics v13.2: The Final Artifact

**Prophet-Guided Deep Reinforcement Learning for Autonomous Ops**

> *"Intelligence is not just about solving problems; it's about predicting them before they exist."*

**Resonetics v13.2** represents the culmination of the autonomous operations research. Unlike previous tabular approaches (v10), this artifact deploys a **Deep Q-Network (DQN)** augmented with a specialized **"Prophet" Network**—a secondary neural brain dedicated solely to forecasting system collapse risks (OOM/Crash).

This system transforms Kubernetes cluster management from a reactive "whack-a-mole" game into a predictive, strategic art form, running within a **Streamlit-based production environment** secured against data loss and adversarial attacks.

---

## ⚡ Key Innovations (v13.2)

### 1. The Dual-Brain Architecture (DQN + Prophet)

The agent is no longer a simple look-up table. It consists of two distinct neural networks working in tandem:

* **The Pilot (DQN):** Learns optimal resource management strategies (Scaling, Cleaning, Throttling) to maximize uptime and efficiency.
* **The Prophet (Risk Estimator):** A dedicated neural network that observes global trends to predict the *probability of future system failure*.
* *Mechanism:* If the Prophet senses high risk, it dynamically suppresses the Pilot's exploratory behavior (`epsilon` decay), forcing a "Safety First" mode.



### 2. Production-Grade Stability & Security

We have moved beyond "research code" to "production artifact":

* **🛡️ Secure Model Loading:** Implemented `weights_only=True` serialization to prevent Pickle-based Remote Code Execution (RCE) attacks. Includes legacy fallback for compatibility.
* **💾 Graceful Shutdown & Persistence:**
* **Auto-Save:** Automatically snapshots the neural weights to local disk upon reaching runtime limits.
* **Zero Data Loss:** Metrics are logged to CSV in real-time with overwrite protection.


* **📉 Optimized Visualization:** Rendering logic has been decoupled from the training loop, ensuring smooth 60fps UI updates without slowing down the learning core.

---

## 🏛️ System Architecture

### The Environment: Kubernetes Smart Tensor (v5.1)

* **State:** 3D Tensor Grid `(Height, Width, Channels)` representing CPU, Memory Drift, and Priority.
* **Entropy:** Simulates the "Red Queen Effect"—constant degradation requiring active intervention.

### The Agent: Prophet-Guided Deep Learner

* **Input:** Local 3x3 Convolution + Global System Stats.
* **Action Space:** Discrete (8 Actions) including movement, specific resource cleaning, and global scaling.
* **Optimizer:** Dual Adam Optimizers with decoupled learning rates (Prophet learns cautiously at `0.5x` speed to provide stable guidance).

---

## 🚀 Quickstart

### 1. Prerequisites

Upgrade your stack to support Deep Learning (PyTorch).

```bash
pip install torch torchvision plotly pandas numpy streamlit packaging

```

### 2. Launch the Artifact

Run the finalized application.

```bash
streamlit run app.py

```

### 3. Operational Workflow

1. **Configure:** Set `Max Steps` and `Learning Rate` in the sidebar.
2. **Safety Limits:** Set `Max Runtime (h)` to ensure the agent saves data before cloud environments timeout.
3. **Train:** Click `▶️ Start`. The agent will begin exploring.
4. **Observe:** Watch the **"Prophet Risk"** metric. A rising risk score indicates the AI perceives impending doom even before the load spikes.

---

## 📊 Legacy Support (v10)

* **Resonetics v10 (Q-Table/Optuna):** Kept as a benchmark baseline. It represents the "Classical ML" approach compared to the "Deep RL" approach of v13.
* To run the legacy system: `streamlit run old_resonetics_v10.py` (rename if necessary).

---

## 📜 Design Philosophy

**"Survive First, Optimize Later."**
Most RL agents fail because they greedily chase rewards until they crash. The **Prophet Network** introduces the concept of "Fear" to the AI. By quantifying the risk of death, the agent learns to sacrifice short-term efficiency for long-term survival—mimicking the evolutionary pressure of biological systems.

---

**© 2025 Resonetics Research Institute** | *From Entropy to Anti-Fragility.*
