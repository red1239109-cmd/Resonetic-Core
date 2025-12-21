# Contributing Guide (DRE / Data Refinery Engine)

Welcome! This project treats **governance (safety boundaries, audit logs, explainability)** and **systemic fairness (structural bias detection)** as first-class citizens while performing data refinement.

## TL;DR (Fastest way to contribute)

1. Fork / clone the repository  
2. Create a virtual environment  
3. `pip install -r requirements-dev.txt`  
4. Run `pytest -q` (must pass)  
5. Open a Pull Request (PRs that break tests, logs, or explain cards will be rejected)

---

## 1. Development Principles (Project Rules)

### 1. Single Trunk Principle (No version fragmentation)
- Even as features grow, we evolve around **one canonical file / one trunk**.  
- Instead of proliferating files like `v1.6.x`, `v1.7.x`, we maintain a single **“canonical”** file (e.g., `dre_single_trunk.py` or `dre_v1_11_0_single_trunk.py`).  
- Experimental code goes into `experiments/`. Merges are allowed only into the canonical trunk.

### 2. Immutable Governance Rules
The following invariants **must never be broken** in any PR:
- **Threshold Bounds:** `0.10 ≤ threshold ≤ 0.90`  
- **Auditability:** Important decisions must be logged in JSONL format (timeline / audit / queue / dag)  
- **Explainability:** Every column decision must produce an ExplainCard  
- **Deterministic-ish behavior:** Tests must be reproducible with fixed seeds / samples

### 3. Systemic Fairness First
- We prioritize detection of **structural bias over time** (Always-Winner dominance) rather than one-off fairness scores.  
- Bias detection events must be recorded in the timeline with `kind="systemic_unfairness"`.

---

## 2. Development Environment Setup

### Windows (PowerShell recommended)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-dev.txt
pytest -q
```

### Windows (CMD)
```cmd
py -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -U pip
pip install -r requirements-dev.txt
pytest -q
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements-dev.txt
pytest -q
```

---

## 3. Branch & PR Conventions

**Branch naming**
- `feat/<topic>` → e.g., `feat/dag-dot-render`
- `fix/<topic>` → e.g., `fix/import-hygiene`
- `perf/<topic>` → e.g., `perf/entropy-sampling`

**PR Checklist (mandatory)**
- [ ] `pytest -q` passes
- [ ] New features include at least one new test
- [ ] Required logs (timeline/audit/queue/dag) are produced
- [ ] Governance invariants are preserved (threshold bounds, explain cards)
- [ ] If performance changes, attach simple benchmark results (text is fine)

---

## 4. Testing Philosophy (Tests We Care About)

Required test categories:
1. **Governance** – threshold bounds, fail-safes, logging
2. **Explainability** – explain card coverage / missing prevention
3. **Fairness** – always-winner detector triggering
4. **DAG Integrity** – dependency ordering, cycle detection
5. **Regression** – import hygiene / missing symbols

> Tip: Most failures come from import errors, undefined symbols, or file-name mismatches.  
> PR reviews will first check whether the file imports cleanly.

---

## 5. Coding Style

Minimum rules:
- Target Python 3.10+ (type hints strongly encouraged)
- Side effects (file writing) limited to the `runs/` directory
- JSONL logs must use `ensure_ascii=False`
- Never swallow exceptions – record them in the timeline with `kind="error"`
- Consider data scale – `value_counts()` / `entropy()` can be bottlenecks → sampling / guards are allowed
- Performance PRs should be separated from feature PRs

---

## 6. License (AGPL-3.0)

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
If you provide this software as a network service, you are required to make the source code available.  
Closed-source SaaS usage without source disclosure is not permitted.

By submitting a PR, you agree that your contribution will be licensed under AGPL-3.0.

---

## 7. Bug Reports / Issue Template

Please include in bug reports:
- OS / Python version
- Exact command used (PowerShell vs CMD, etc.)
- Full error log
- Import statement used (e.g., `from dre_single_trunk import DataRefineryEngine`)

For performance issues:
- Data size (rows / columns)
- Measured wall time
- Memory usage (if possible)

---

## 8. “Awesome Feature” Roadmap (Contributions welcome!)

- [ ] DAGRunner + DOT visualization fusion (DAG → dot → png)
- [ ] DAG cycle detection + error events
- [ ] Benchmark harness (100K / 500K / 1M) + performance gate tests
- [ ] Fairness hook extensions (dominance + drift + collapse)
- [ ] Visualization (graph + dynamics)

---

Thank you for your interest.  
DRE is not just “code that works well” — it is **code that works well while explaining why and detecting bias**.

Welcome aboard!
