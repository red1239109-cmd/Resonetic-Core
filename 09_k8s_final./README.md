---

## Autonomous Kubernetes Cluster Maintenance via Reinforcement Learning

This project provides a **reinforcement learning environment** that simulates a Kubernetes cluster as a **3D tensor grid**, where an autonomous agent (“the gardener”) learns to manage resources under **strict budget constraints**.

The environment explicitly models the **Red Queen Effect**: system entropy (CPU load, memory leaks) increases over time regardless of agent behavior. The agent must therefore **continuously allocate limited resources** to prevent cluster collapse (OOM, overload), rather than optimizing toward a static equilibrium.

---

## 🌟 Key Features

### 1) 3D Tensor Observation Space

The cluster is represented as a tensor of shape `(H, W, C)`:

* **Channel 0 (CPU):** stochastic load fluctuations
* **Channel 1 (Memory):** monotonically increasing memory pressure (Red Queen Effect)
* **Channel 2 (Priority):** node importance (critical vs non-critical workloads)

The agent does **not** observe the full grid. Instead, it receives:

* a **local 3×3×3 neighborhood view** centered on its position
* additional normalized state variables: `(y_position, x_position, budget)`

Total observation size: **30-dimensional vector**

This enforces **partial observability** and local decision-making.

---

### 2) Budget-Constrained Resource Management

All actions incur costs.

* **Scaling:** expensive; when budget is insufficient, the agent must learn **partial scaling**
* **Cleaning:** reduces CPU/memory locally, but inefficient cleaning is penalized
* Budget recovers slowly over time, rewarding **long-term planning** over greedy actions

The agent must balance *where*, *when*, and *how much* to intervene.

---

### 3) Gymnasium-Compliant & Reproducible

The environment strictly follows the **Gymnasium API**:

* deterministic seeding via `reset(seed=...)`
* standard `step(action)` semantics
* plug-and-play compatibility with RL libraries (SB3, RLlib, CleanRL, etc.)

This enables **fully reproducible experiments** and fair policy comparisons.

---

## ✅ Benchmark Results (Random vs Heuristic)

```
======================================================================
BENCHMARK RESULTS (Random vs Heuristic)
======================================================================
random    | reward μ=-12.456 σ=4.821 | steps μ=312.4 | oom_rate=1.82 | crit_rate=0.68 | avg_load μ=0.512
heuristic | reward μ=+18.912 σ=3.215 | steps μ=456.8 | oom_rate=0.34 | crit_rate=0.12 | avg_load μ=0.378
----------------------------------------------------------------------
Heuristic win-rate: 50/50 (ties: 0)
======================================================================
```

### Interpretation

* **Random policy** rapidly depletes budget and fails to control memory pressure, leading to frequent OOM events.
* **Heuristic policy** survives significantly longer, maintains lower average system load, and reduces both total and critical OOM rates.
* A **100% win-rate** across 50 runs confirms that the environment rewards *structural decision-making*, not luck.

These results validate that the environment meaningfully distinguishes between **uninformed actions** and **resource-aware policies**.

---

## What This Environment Is For

* Studying **autonomous infrastructure management**
* Training RL agents under **non-stationary, adversarial dynamics**
* Evaluating decision-making under **budgeted intervention constraints**

## What This Is Not

* ❌ Not a Kubernetes simulator
* ❌ Not an autoscaler replacement
* ❌ Not production control software

This is a **research-grade abstraction**, designed to expose trade-offs that real systems often hide.

---

