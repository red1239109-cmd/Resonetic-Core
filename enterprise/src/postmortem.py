# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
from typing import Dict, Any
from . import now_ts, fmt_ts
from .timeline import IncidentTimeline, TimelineEvent

class PostmortemGenerator:
    def __init__(self, timeline: IncidentTimeline):
        self.timeline = timeline

    def generate(self, incident_id: str, step: int) -> Dict[str, Any]:
        events = self.timeline.list_all(incident_id=incident_id)
        opens = [e for e in events if e.get("kind") == "anomaly"]
        actions = [e for e in events if e.get("kind") == "action_apply"]
        effects = [e for e in events if e.get("kind") == "action_effect"]
        resolves = [e for e in events if e.get("kind") == "resolve"]

        summary = {
            "incident_id": incident_id,
            "opened_at": opens[0]["ts"] if opens else None,
            "resolved_at": resolves[-1]["ts"] if resolves else None,
            "num_actions": len(actions),
            "final_verdict": effects[-1].get("detail", {}).get("verdict") if effects else "UNKNOWN",
        }

        timeline_lines = []
        if opens:
            timeline_lines.append(f"[OPEN] {fmt_ts(opens[0]['ts'])} - {opens[0].get('title')}")
        for a in actions:
            tags = a.get("tags", [])
            actor = tags[1] if len(tags) > 1 else "operator"
            timeline_lines.append(f"[ACTION] {fmt_ts(a['ts'])} - {a.get('title')} (Actor: {actor})")
        for e in effects:
            d = e.get("detail", {})
            timeline_lines.append(f"[EFFECT] Verdict: {d.get('verdict')} (Score: {d.get('score', 0):.2f})")
        if resolves:
            timeline_lines.append(f"[RESOLVE] {fmt_ts(resolves[-1]['ts'])} - {resolves[-1].get('title')}")

        pm = {"summary": summary, "timeline": timeline_lines}
        self.timeline.add(
            TimelineEvent(
                ts=now_ts(),
                step=int(step),
                kind="postmortem",
                severity="info",
                title=f"Postmortem Generated: {incident_id}",
                incident_id=incident_id,
                tags=["postmortem", "rca"],
                detail=pm,
            )
        )
        return pm
