# DRE Architecture

The **Data Refinery Engine (DRE)** is not merely a data "refinement" engine—  
it is a **decision-making infrastructure** that explains, records, and controls data-driven judgments.

This document describes the core architecture of DRE and the role of each layer.

---

## 1. Design Principles

DRE treats the following five principles as **non-negotiable**:

1. **Explainability First**  
   - Every decision produces an Explain Card.

2. **Audit by Default**  
   - All important events are recorded in the Timeline/Audit Trail.

3. **Governance over Optimization**  
   - Performance can be tuned, but safety rules (the "constitution") are inviolable.

4. **Systemic Fairness**  
   - We monitor long-term structural bias, not just short-term metrics.

5. **Reproducibility**  
   - Same input → similar judgment path.

---

## 2. Overall Architecture Overview

```
┌───────────────┐
│   Input Data   │
└───────┬───────┘
        ▼
┌──────────────────────────┐
│   Data Refinery Engine   │
│                          │
│  ┌────────────────────┐  │
│  │ Column Evaluator    │  │
│  │ - Quality Score     │  │
│  │ - Relevance Score   │  │
│  └────────┬──────────┘  │
│           │             │
│  ┌────────▼──────────┐  │
│  │ Governance Layer   │  │
│  │ - Threshold PID    │  │
│  │ - Policy Guard     │  │
│  └────────┬──────────┘  │
│           │             │
│  ┌────────▼──────────┐  │
│  │ Decision Engine    │  │
│  │ - KEEP / DROP      │  │
│  └────────┬──────────┘  │
│           │             │
│  ┌────────▼──────────┐  │
│  │ Explain Builder    │  │
│  │ - Explain Cards    │  │
│  └────────┬──────────┘  │
│           │             │
│  ┌────────▼──────────┐  │
│  │ Effect Analyzer    │  │
│  │ - Action Impact    │  │
│  │ - Winner Detection │  │
│  └────────┬──────────┘  │
│           │             │
│  ┌────────▼──────────┐  │
│  │   DAG Store        │  │
│  │ - Lineage Tracking │  │
│  └───────────────────┘  │
└───────────────┬──────────┘
                ▼
        Dashboard / API
```

---

## 3. Core Components

### 3.1 Column Evaluator

**Purpose:**  
Not simply classifying columns as "useful / useless," but **quantifying why** a judgment was made.

- `quality`: Based on missing rate and duplication  
- `relevance`: Based on information entropy  
- `gold_score`: Combined quality + relevance score

> ❗ DRE does not train models.  
> It **explicitly computes** the criteria for judgment.

---

### 3.2 Governance Layer

**Core Idea:**  
> "Performance is adjustable, but safety is absolute."

Components:
- **PID Threshold Controller**  
  - Automatically adjusts threshold toward target retention rate  
- **Immutable Bounds**  
  - Threshold never leaves `[0.10, 0.90]`  
- (Future) **Supreme Court / Policy Engine**  
  - Constitutional rules that prohibit certain actions outright

---

### 3.3 Decision Engine

- `gold_score >= threshold` → KEEP  
- Otherwise → DROP

Key point:
- Decision logic is **intentionally simple**  
- Complexity resides in the explanation, recording, and control layers

---

### 3.4 Explain Card System

Generates an **Explain Card** for every column that can be understood in ~3 seconds:

- Decision (KEEP / DROP)  
- Key metric comparison  
- Reason for judgment (score_based, policy_guard, etc.)

Explain Cards serve as a **common language** for:
- Operators  
- Audit teams  
- Regulatory compliance

---

### 3.5 Action Effect Analyzer

One of DRE’s key differentiators.

**Question:**  
> “Did this judgment actually help?”

Metrics analyzed:
- Risk change  
- Loss change  
- Stability change

Verdict:
- `effective`  
- `partial`  
- `ineffective`

---

### 3.6 Systemic Fairness Layer (Always-Winner Detector)

Looks at the **time axis**, not single batches.

Detects:
- Structures where Risk always wins  
- Bias where Stability is always selected  
- Monopolistic dominance of a single criterion

When conditions are met:
- `systemic_unfairness` event is emitted  
- Explicit warning recorded in Timeline

> Captures the moment short-term optimization turns into long-term unfairness.

---

### 3.7 DAG Store (Lineage)

Every batch is a node in the DAG.

Features:
- Input Dataset → Batch → Output Dataset  
- Automatic chaining when the same output is re-input  
- DAG rehydration on restart

Meaning:
- Answers “Where did this result come from?”  
- Blurs the line between experimentation and production

---

## 4. Dashboard Layer

The dashboard is not decoration—it is an **operational tool**.

Views provided:
- Batch list  
- Queue status  
- Timeline (Audit + Fairness)  
- DAG visualization  
- Explain Card details

Philosophy:
> **Explanation → Understanding → Intervention → Control**

---

## 5. What DRE Intentionally Does Not Do

DRE deliberately avoids:

- End-to-end AutoML  
- Black-box feature selection  
- Accuracy optimization races  
- Silent decision-making

Reason:
> DRE is not an engine that “gets the right answer”—  
> it is an engine you can **trust with judgment**.

---

## 6. Extension Directions

- Polars / Arrow backend  
- Policy DSL (constitutional rule language)  
- External KPI feedback loop  
- Multi-agent governance  
- Human-in-the-loop override

---

## 7. One-Sentence Summary

> **DRE is not a tool that refines data—  
> it is a structure that makes data judgments open, recorded, and controlled.**
