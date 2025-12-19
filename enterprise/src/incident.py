# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from . import fmt_ts

@dataclass
class IncidentRecord:
    incident_id: str
    status: str
    severity: str
    title: str
    created_ts: float
    last_ts: float
    last_step: int
    stable_steps: int = 0
    required_stable_steps: int = 10
    stability_score: float = 0.0
    subscores: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    action_ids: List[str] = field(default_factory=list)

class IncidentRegistry:
    def __init__(self):
        self.by_id: Dict[str, IncidentRecord] = {}

    def create_or_update(
        self,
        incident_id: Optional[str] = None,
        severity: str = "info",
        title: str = "",
        step: int = 0,
        tags: List[str] = None,
        action_ids: List[str] = None,
        status: str = None
    ) -> IncidentRecord:
        if incident_id is None:
            incident_id = f"inc_{int(time.time())}_{str(uuid.uuid4())[:4]}"

        now = time.time()
        if incident_id in self.by_id:
            rec = self.by_id[incident_id]
            rec.last_ts = now
            rec.last_step = step
            if status:
                rec.status = status
            if tags:
                rec.tags.extend(tags)
                rec.tags = list(set(rec.tags))
            if action_ids:
                rec.action_ids.extend(action_ids)
                rec.action_ids = list(set(rec.action_ids))

            if rec.status == "RESOLVED" and severity in ["warn", "high"]:
                rec.status = "OPEN"
                rec.severity = severity
                rec.stable_steps = 0
        else:
            rec = IncidentRecord(
                incident_id=incident_id,
                status=status or "OPEN",
                severity=severity,
                title=title,
                created_ts=now,
                last_ts=now,
                last_step=step,
                tags=tags or [],
                action_ids=action_ids or []
            )
            self.by_id[incident_id] = rec
        return rec

    def update_stability(self, incident_id: str, is_stable: bool) -> None:
        if incident_id not in self.by_id:
            return
        rec = self.by_id[incident_id]

        if is_stable:
            rec.stable_steps += 1
        else:
            rec.stable_steps = 0

        if rec.required_stable_steps > 0:
            rec.stability_score = min(1.0, rec.stable_steps / rec.required_stable_steps)

        if rec.status in ["OPEN", "MITIGATING"]:
            if rec.stability_score >= 1.0:
                rec.status = "RESOLVED"
            elif rec.stability_score >= 0.1:
                if rec.status == "OPEN":
                    rec.status = "MITIGATING"

    def summary(self, host: str = "localhost", port: int = 8080) -> Dict[str, Any]:
        records = list(self.by_id.values())
        return {
            "summary": {
                "OPEN": sum(1 for r in records if r.status == "OPEN"),
                "MITIGATING": sum(1 for r in records if r.status == "MITIGATING"),
                "RESOLVED": sum(1 for r in records if r.status == "RESOLVED"),
            },
            "items": [
                {
                    **asdict(r),
                    "created_time": fmt_ts(r.created_ts),
                    "drilldown": f"http://{host}:{port}/timeline?incident_id={r.incident_id}",
                }
                for r in sorted(records, key=lambda r: r.last_ts, reverse=True)
            ],
        }
