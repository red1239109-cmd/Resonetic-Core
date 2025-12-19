# Enterprise AI Operation System

**Explainability-first incident management for AI/ML systems.**

## Structure
- `src/`: Pure operation logic (holy ground)
- `dashboard/`: Human-readable views
- `runs/`: Audit trail (DO NOT GITIGNORE)
- `scripts/`: Simulations & tooling
- `docs/`: Architecture decisions
- `tests/`: Critical path coverage

## Demo Screenshots

### Dashboard Overview

![Dashboard](https://via.placeholder.com/800x400?text=Operational+Dashboard+(OPEN+0,+MITIGATING+0,+RESOLVED+1))
<img width="1761" height="644" alt="fe0c6c85-7b10-49ba-a8e2-b638da0a3e66" src="https://github.com/user-attachments/assets/c31db8a2-99e8-48bd-8461-e2312097609e" />

### Timeline with Veto Events
![Timeline Veto](https://via.placeholder.com/800x600?text=Kant+and+Rawls+Veto+Events)
<img width="1844" height="2105" alt="6acabf34-6f41-4f74-a06c-d5c0c97fd468" src="https://github.com/user-attachments/assets/a3d85d79-48e1-43b5-bbfd-07bd97ead6f9" />

### Postmortem Report
![Postmortem](https://via.placeholder.com/800x500?text=Automated+Postmortem+with+Verdict)
<img width="1506" height="1841" alt="fbb0f845-f68e-475e-a4c3-0e3149070b54" src="https://github.com/user-attachments/assets/9c99e82c-341f-4c00-a9ff-4edd5dc5e7ac" />

## Quick Start
```bash
pip install -r requirements.txt
python scripts/demo_simulation.py

Core Concept
Every action is recorded, validated, and auditable by default.

## License
This project is dual-licensed under AGPL-3.0 or Commercial.  
See [LICENSING.md](LICENSING.md) for details.
