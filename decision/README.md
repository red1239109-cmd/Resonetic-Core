# Resonetic Decision Primitive

This module defines a **belief update decision primitive**.

It is used internally by Resonetics, DRE, and other systems —  
but it is intentionally shipped as a **standalone decision unit**.

---

## What this module does

This module governs **how an existing belief state should be updated**
when a new candidate belief or observation arrives.

Instead of directly overwriting beliefs, it separates the process into two steps:

1. **Decision** — whether an update should occur, and with what strength  
2. **Application** — how the existing belief is adjusted if the update is accepted

The decision is based on signals such as:
- coherence
- shock
- observation quality

---

## What this module does NOT do

- ❌ No model inference  
- ❌ No domain-specific assumptions  
- ❌ No embeddings, networks, or learning  
- ❌ No task-level logic  

This module only answers one question:

> *“Given the current belief and a candidate update, how should the belief change?”*

---

## Design philosophy

- Deterministic and side-effect free
- Scalar or vector-agnostic
- Easy to reason about and test
- Safe to embed inside larger systems

This is not a policy engine.  
It is not an optimizer.  
It is a **decision primitive**.

---

## Typical usage

```python
from belief_update_governor import BeliefUpdateGovernor

gov = BeliefUpdateGovernor()

decision = gov.decide(
    coherence=coherence,
    shock=shock,
    obs_quality=confidence
)

belief = gov.apply_update(
    belief,
    candidate_belief,
    decision
)

Intended audience
System designers
Researchers
Engineers building multi-stage reasoning or refinement pipelines
If you need control over belief updates rather than blind replacement, this module is meant for you.
