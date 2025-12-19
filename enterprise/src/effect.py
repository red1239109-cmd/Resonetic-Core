# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
import statistics
from typing import Dict, Optional, Any
from collections import deque
from . import now_ts
from .timeline import IncidentTimeline, TimelineEvent

class ActionEffectAnalyzer:
    def __init__(self, timeline: IncidentTimeline, window_steps: int = 10):
        self.timeline = timeline
        self.window_steps = int(window_steps)
        self.buffers: Dict[str, Dict[str, Any]] = {}

    def start(self, incident_id: str, action_id: str, step: int, baseline_metrics: Dict[str, float]):
        self.buffers[action_id] = {
            "incident_id": incident_id,
            "action_id": action_id,
            "start_step": int(step),
            "baseline": dict(baseline_metrics or {}),
            "samples": deque(maxlen=self.window_steps),
        }

    def collect(self, action_id: str, metrics: Dict[str, float]):
        if action_id not in self.buffers:
            return
        buf = self.buffers[action_id]
        buf["samples"].append(dict(metrics or {}))

        # Real-time Verdict
        if len(buf["samples"]) >= self.window_steps:
            self.finalize(action_id, step=buf["start_step"] + self.window_steps)

    def finalize(self, action_id: str, step: int) -> Optional[Dict[str, Any]]:
        buf = self.buffers.pop(action_id, None)
        if not buf:
            return None
        samples = list(buf["samples"])
        if not samples:
            return None

        baseline = buf["baseline"]

        def avg(key: str):
            vals = [s.get(key) for s in samples if isinstance(s.get(key), (int, float))]
            return float(statistics.mean(vals)) if vals else None

        after = {k: avg(k) for k in ["risk", "loss", "stability"]}
        delta = {}
        for k, v_after in after.items():
            v_base = baseline.get(k)
            if isinstance(v_after, (int, float)) and isinstance(v_base, (int, float)):
                delta[k] = float(v_after - v_base)

        score = 0.0
        score += (-delta.get("risk", 0.0)) * 1.0
        score += (-delta.get("loss", 0.0)) * 0.5
        score += (delta.get("stability", 0.0)) * 1.0

        verdict = "ineffective"
        if score > 0.10:
            verdict = "effective"
        elif score > 0.02:
            verdict = "partial"

        ev_detail = {
            "kind": "action_effect",
            "verdict": verdict,
            "baseline": baseline,
            "after_avg": after,
            "delta": delta,
            "window_steps": self.window_steps,
            "score": score,
            "action_id": action_id,
        }
        self.timeline.add(
            TimelineEvent(
                ts=now_ts(),
                step=int(step),
                kind="action_effect",
                severity="info" if verdict == "effective" else "warn",
                title=f"Action Effect: {verdict.upper()} (score={score:.3f})",
                incident_id=buf["incident_id"],
                tags=["effect", verdict],
                detail=ev_detail,
            )
        )
        return ev_detail
