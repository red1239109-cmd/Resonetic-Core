# Resonetics — VERSIONS (Lineage Map)

This document is the project’s **lineage map**:
what each version was trying to prove, what it removed, and what you should use today.

Resonetics evolves by **subtraction**:
each version deletes an assumption that previously looked “necessary”.

---

## Quick Pick (If you just want something that works)

- **Need a decision primitive** (small, reusable, testable):
  → `Resonetic-Core/decision/` (BeliefUpdateGovernor)

- **Need a production guardrail / safety envelope**:
  → `godel_guardrail_enterprise_*`

- **Need a data pipeline refiner**:
  → `dre_*_single_trunk.py`

- **Need the full reasoning engine prototype**:
  → `resonetics_engine_v6_2.py` (experimental)

---

## Version Lineage

### 1) Decision Primitive (Core)
**Folder:** `Resonetic-Core/decision/`  
**Goal:** A standalone decision kernel that can be embedded anywhere.  
**What it removed:** “The engine must own the whole system.”  
**Kept philosophy:** tension-aware gating; auditable decisions.  
**Use when:** You need deterministic decisioning without pulling the full stack.

---

### 2) Via Negativa Core (Concept Engine)
**Files:** `resonetics_via_negativa_*`  
**Goal:** Make tension management explicit: preserve / ignore / collapse.  
**What it removed:** “Contradiction is always an error.”  
**Use when:** You want the philosophical core as executable logic.

---

### 3) Auditor (Structural Truth Tester)
**Files:** `resonetics_auditor_*`  
**Goal:** Treat code as a living structure; measure honesty vs complexity.  
**What it removed:** “Style is quality.”  
**Use when:** You need deterministic, explainable analysis outputs.

---

### 4) Prophet / Enterprise Runtime (System Self-Protection)
**Files:** `resonetics_prophet_*`  
**Goal:** Keep a system alive under pressure: drift, instability, budget limits.  
**What it removed:** “Training stability is somebody else’s problem.”  
**Use when:** You want ops-grade monitoring, metrics, safeguards.

---

### 5) K8s RL Environment (Optional Extension)
**Files:** `resonetics_k8s_*`  
**Goal:** A sandbox where entropy fights back (Red Queen dynamics).  
**What it removed:** “Environment is passive.”  
**Use when:** You need a benchmark arena, not a product.

---

## Design Principle (Non-Negotiable)

Resonetics is not “more features per version”.
It is **less illusion per version**.

Each step reduces drift:
- fewer hidden assumptions
- smaller primitives
- clearer boundaries
- more auditability

That’s the whole point.
