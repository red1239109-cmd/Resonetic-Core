# Resonetics  
Constraint-Grounded Learning Systems for Stable Adaptation

Resonetics is a project dedicated to researching and implementing learning systems centered on constraints.  
It begins not with the question **“What should we maximize?”**,  
but with **“What must remain invariant while change occurs?”**.

### Core Ideas
All Resonetics systems share three fundamental principles:

1. **Constraints are not to be eliminated — they are to be measured.**  
2. **Instability is not suppressed — it is predicted.**  
3. **Autonomy becomes safe only through meta-regulation.**

These principles are realized in different forms across numerical models (tensors), language systems (text), and operational systems (enterprise control).

### Project Structure

1. **Resonetics Prophet (Enterprise System)**  
   Example file: `resonetics_prophet_v8_4_1_enterprise_clean.py`

   Role  
   - Trains a worker model in dynamic environments  
   - Uses a meta-model (Risk Predictor) to forecast instability (risk)  
   - Automatically adjusts learning rate and mode (PANIC / ALERT / CRUISE) based on predicted risk  

   Key Features  
   - Risk-aware learning rate tuning  
   - Concept drift simulation  
   - Prometheus & health-check based operational monitoring  
   - Graceful shutdown & checkpointing  

   This system is not a “performance optimizer” —  
   it is a **regulator that sustains learning while avoiding collapse**.

2. **Resonetics Auditor (Code Analysis Tool)**  
   Example file: `resonetics_auditor_v6_5_hardened.py`

   Role  
   - Structural stability analysis of Python code  
   - Evaluates function length, complexity, resilience, and async usage  

   Output  
   - Quantitative overall score  
   - Detailed per-function / per-class reports  
   - Complexity penalty and resilience metrics  

   The Auditor is not a “style checker” —  
   it is a **detector of potential structural collapse**.

3. **EtherVAE Benchmarks (Scientific Evaluation)**  
   Example file: `ethervae_benchmark_v6_logging.py`

   Role  
   - Rigorous evaluation of VAE variants (Ether / Resonetics modes)  
   - Analysis of manifold stability under varying entropy  
   - Measurement of local smoothness and global manifold quality  

   Included Elements  
   - Statistical testing (t-test, effect size)  
   - Visualization-based comparison  
   - Reproducible experiment logging  

### Documentation Structure

| File              | Purpose                                      |
|-------------------|----------------------------------------------|
| README.md         | Overall project overview                     |
| THEORY.md         | Design theory of numerical systems           |
| PHILOSOPHY.md     | Philosophical background on constraints and instability |
| KNOWN_ISSUES.md   | Known limitations and risks                  |
| EXPERIMENT_LOG_EXAMPLE.md | Guide to reading experiment logs             |

### License
**Dual License Model**

**Open Source**  
- AGPL-3.0  
- Source code must be disclosed when used over a network

**Commercial License**  
- Available for proprietary use  
- Terms negotiated based on organization size and usage scope

📧 Contact: red1239109@gmail.com

### What Resonetics Is Not

- A claim of universal AGI ❌  
- A black-box “self-aware” model ❌  
- A benchmark-chasing performance competition ❌

Resonetics is an experimental learning framework  
centered on control, stability, and interpretability.

### One-Line Summary
Resonetics does not maximize freedom.  
It learns the boundaries that must not collapse.

Non-AGI Clarification

This system does not claim Artificial General Intelligence.
All objectives, task boundaries, and evaluation criteria are externally defined.
The system does not autonomously generate goals, expand its task domain,
or perform open-ended self-directed learning.

Observed adaptive behavior emerges from layered constraint mechanisms
designed to preserve stability under non-stationary conditions,
not from general problem-solving capability.

Any appearance of “general intelligence” should be understood as
robust control under uncertainty, not domain-independent cognition.
