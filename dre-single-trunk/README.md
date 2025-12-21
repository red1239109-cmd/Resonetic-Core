# Data Refinery Engine (DRE) - Single Trunk Edition

**Ultra-fast data refinement engine with built-in governance**  
Process **500,000 rows in ~0.5 seconds** while automatically providing explainability, fairness detection, and effect analysis.

- **Single file philosophy** – One file. One truth. No drift.
- **Zero external configuration** – Works out of the box
- **Production-grade features** built-in:
  - Explainable column decisions (ExplainCards)
  - Systemic bias detection (AlwaysWinnerDetector)
  - Action impact analysis (ActionEffectAnalyzer)
  - Adaptive threshold tuning (PID controller)
  - DAG-based pipeline support (DAGRunner)
- **Performance**: >1M rows/sec, <6MB memory overhead (see benchmark below)

![Benchmark Results](benchmark.png)
<img width="680" height="927" alt="image (1)" src="https://github.com/user-attachments/assets/07004308-9cbf-43e3-94f9-a83d080b0147" />

## Quick Start

```python
import pandas as pd
from dre import DataRefineryEngine

# Load your data
df = pd.read_csv("your_dataset.csv")

# Initialize engine (optionally pin a target KPI)
engine = DataRefineryEngine(target_kpi="revenue")  # or None

# Refine – returns cleaned DataFrame + lineage + explanation cards
refined_df, lineage, explain_cards = engine.refine(df)

print(f"Kept {len(lineage['kept'])} columns, dropped {len(lineage['dropped'])}")
print(f"Final threshold: {engine.threshold:.3f}")
print(f"Generated {len(explain_cards)} explanation cards")

# Example: view first explanation
print(explain_cards[0].headline)

Features

Automatic column scoring – quality + relevance → gold score
High-cardinality guard – prevents ID-like columns from exploding entropy
Entropy-based relevance – sampled for large columns (safe & fast)
PID-controlled retention – dynamically adjusts threshold to meet target
Full audit trail – timeline.jsonl + queue.jsonl
DAGRunner included – compose multi-step pipelines with dependency ordering
Test-friendly – all core components instantiate without arguments

Benchmark (500k rows, typical mixed dataset)

Scenario,   Rows,     Time,     RPS,       Memory Δ
Small,     "1,000",   0.016s,  "62,373",    0.75 MB
Medium,    "10,000",  0.018s,  "555,927",   0.29 MB
Large,     "100,000", 0.098s,  "1,017,356", 1.64 MB
Heavy,     "500,000", 0.501s,  "996,714",   5.97 MB

Requirements
txtpandas>=1.5
numpy>=1.21
scipy>=1.7

License
Apache License 2.0 – see LICENSE for details.
Commercial use, modification, and distribution fully permitted.

Author
red1239109-cmd – 2025
Enjoy refining your data!
