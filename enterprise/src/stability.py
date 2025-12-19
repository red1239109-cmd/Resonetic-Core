# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
from __future__ import annotations
from typing import Dict, Optional
from .timeline import TimelineEvent, now_ts

class StabilityTracker:
    def __init__(self, registry, timeline,
                 effect_analyzer: Optional = None,
                 postmortem: Optional = None):
        self.registry = registry
        self.timeline = timeline
        self.effect_analyzer = effect_analyzer
        self.postmortem = postmortem

    def observe(self, incident_id: str, step: int, stability: float, signal: Dict) -> None:
        is_stable = stability >= 0.8
        self.registry.update_stability(incident_id, is_stable)
        rec = self.registry.by_id.get(incident_id)

        # Trigger Resolution
        if rec and rec.status == "RESOLVED" and rec.stable_steps == rec.required_stable_steps:
            self.timeline.add(TimelineEvent(
                ts=now_ts(), step=step, kind="resolve", severity="info",
                title=f"Incident Resolved (Stable {rec.required_stable_steps} steps)",
                incident_id=incident_id, detail={"final_score": stability, "signal": signal}
            ))
            # [HOOK] Finalize Effects
            if self.effect_analyzer:
                for aid in list(rec.action_ids or []):
                    self.effect_analyzer.finalize(aid, step=step)
            # [HOOK] Generate Postmortem
            if self.postmortem:
                self.postmortem.generate(incident_id, step=step)
