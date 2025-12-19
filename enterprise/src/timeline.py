# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json
import os
from collections import deque
from typing import Any, Dict, List, Optional
import uuid
import time

def now_ts() -> float:
    return float(time.time())

def fmt_ts(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return str(ts)

@dataclass
class TimelineEvent:
    ts: float
    step: int
    kind: str          # anomaly, action_apply, action_effect, resolve, postmortem
    severity: str      # info, warn, high
    title: str
    detail: Dict[str, Any] = field(default_factory=dict)
    incident_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

class IncidentTimeline:
    def __init__(self, maxlen: int = 5000, jsonl_path: str = "runs/timeline.jsonl"):
        self.buf = deque(maxlen=int(maxlen))
        self.jsonl_path = str(jsonl_path)
        os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)

    def add(self, ev: TimelineEvent) -> str:
        ev_dict = asdict(ev)
        self.buf.append(ev_dict)
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev_dict, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return "evt_" + str(uuid.uuid4())[:8]

    def list_recent(self, limit: int = 50, incident_id: Optional[str] = None) -> List[Dict[str, Any]]:
        limit = int(limit)
        events = list(self.buf)
        if incident_id:
            events = [e for e in events if e.get("incident_id") == incident_id]
        return events[-limit:]

    def list_all(self, incident_id: Optional[str] = None) -> List[Dict[str, Any]]:
        events = list(self.buf)
        if incident_id:
            events = [e for e in events if e.get("incident_id") == incident_id]
        return events
