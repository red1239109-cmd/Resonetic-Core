# Enterprise AI Operation System

**Explainability-first incident management for AI/ML systems.**

## Structure
- `src/`: Pure operation logic (holy ground)
- `dashboard/`: Human-readable views
- `runs/`: Audit trail (DO NOT GITIGNORE)
- `scripts/`: Simulations & tooling
- `docs/`: Architecture decisions
- `tests/`: Critical path coverage

## Quick Start
```bash
pip install -r requirements.txt
python scripts/demo_simulation.py

Core Concept
Every action is recorded, validated, and auditable by default.

## License
This project is dual-licensed under AGPL-3.0 or Commercial.  
See [LICENSING.md](LICENSING.md) for details.
