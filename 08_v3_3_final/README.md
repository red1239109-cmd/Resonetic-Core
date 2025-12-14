# README.md

# Resonetics (Via Negativa / Gardener / Prophet)
Autonomous stability experiments for “systems that drift toward collapse” — and agents that must keep them alive under constraints.

This repo currently contains three pillars:

1) **Gardener (v3.3.2 + patches)**  
   A bio-mimetic simulation where agents survive in a noisy entropy field (“Red Queen effect”).  
   We measure and log *risk*, *verdict*, and now *flow*.

2) **Kubernetes Smart Tensor Env (v4.5)**  
   A Gymnasium-compatible RL environment for cluster maintenance under strict budget constraints.

3) **Prophet (Enterprise + Kernel)**  
   A training loop with monitoring + checkpoints, enhanced by a mathematically meaningful “Resonetics Kernel”.

---

## Core Idea (One paragraph)
Real systems don’t stay stable by default. Entropy creeps in: memory leaks, load drift, emergent hotspots. Resonetics models that drift and asks a practical question: **what policies keep a system alive when instability is inevitable?**  
We use a “Via Negativa” control layer to identify failure modes and suppress actions that accelerate collapse.

---

## Components

### 1) Gardener: Entropy Grid + Agents
**File:** `resonetics_v3_3_2_final_with_patches.py`

- **Reality**: `grid` (dirt/entropy intensity)
- **Structure**: local rule (`find_dirty_neighbor`) + cleaning dynamics
- **Tension**: risk model (entropy gradient + population volatility + emergency ratio)
- **Flow (NEW)**: system sensitivity to small perturbations  
  - “Input-noise Lipschitz-ish” measurement:
    - define metrics `f(state)` such as total entropy and hotspot rate
    - inject small noise `eps`
    - compute `Flow ≈ (Δf²)/eps²`
    - smooth with EMA and mix into risk

#### Logged signals (CSV)
The simulation writes `resonetics_data.csv` with:

- `Gen`
- `Population`
- `Avg_Energy`
- `Total_Entropy`
- `Emergency_Moves`
- `Isolation_Saves`
- `Entropy_Gradient`
- `Population_Volatility`
- `Emergency_Ratio`
- `Collapse_Risk`
- `Flow`
- `Verdict` (`creative | bubble | collapse`)

---

### 2) Kubernetes Smart Tensor Environment (Gymnasium)
**File:** `resonetics_k8s_v4_5_fixed.py`

A minimal, reproducible RL environment for “autonomous cluster maintenance”.

**Observation (30-dim vector)**  
- 3×3 neighborhood × 3 channels = 27  
  - channel 0: CPU
  - channel 1: Memory (drifting upward)
  - channel 2: Priority (critical vs non-critical)
- + agent position (2)
- + budget (1)

**Actions (Discrete 6)**
- 0..3: move
- 4: clean node (cheap)
- 5: scale hotspots (expensive / partial scaling supported)

**Red Queen effect**
- memory drifts upward each step → agent must continuously stabilize

---

### 3) Prophet (Enterprise + Kernel)
**File:** `resonetics_prophet_v8_4_2_enterprise_kernel.py`

Enterprise-friendly trainer with:
- monitoring (Prometheus / Flask health endpoints)
- checkpoints
- “ProphetOptimizer” that tunes LR based on predicted risk

#### Resonetics Kernel v2 (A-Version Flow)
This kernel makes “Flow / Structure / Tension” mathematically meaningful:

- **Reality gap**: MSE(pred, target)
- **Flow**: sensitivity of outputs to small input noise  
  `flow = E[(f(x+eps·noise)-f(x))²] / eps²`
- **Structure**: periodic attraction (default period=3)
- **Tension**: interaction term `tanh(gap_R) * tanh(gap_S)`

> Kernel goal: turn philosophical language into measurable regularizers without pretending it’s magic.

---

## Benchmark (Random vs Heuristic)
======================================================================
✅ BENCHMARK RESULTS (Random vs Heuristic)
random | reward μ=-12.456 σ=4.821 | steps μ=312.4 | oom_rate=1.82 | crit_rate=0.68 | avg_load μ=0.512
heuristic | reward μ=+18.912 σ=3.215 | steps μ=456.8 | oom_rate=0.34 | crit_rate=0.12 | avg_load μ=0.378
Heuristic win-rate: 50/50 (ties: 0)

---

## Quickstart

### Gardener
```bash
python resonetics_v3_3_2_final_with_patches.py
# outputs resonetics_data.csv
Kubernetes Env test
bash
코드 복사
python resonetics_k8s_v4_5_fixed.py
Prophet (Enterprise + Kernel)
bash
코드 복사
python resonetics_prophet_v8_4_2_enterprise_kernel.py --steps 2000
# Optional: --no-monitor, --no-kernel
