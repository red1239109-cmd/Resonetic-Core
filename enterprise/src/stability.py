# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
from typing import Optional, Dict
from . import now_ts
from .timeline import IncidentTimeline, TimelineEvent
from .incident import IncidentRegistry
from .effect import ActionEffectAnalyzer
from .postmortem import PostmortemGenerator

class StabilityTracker:
    def __init__(
        self,
        registry: IncidentRegistry,
        timeline: IncidentTimeline,
        effect_analyzer: Optional[ActionEffectAnalyzer] = None,
        postmortem: Optional[PostmortemGenerator] = None,
    ):
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
            self.timeline.add(
                TimelineEvent(
                    ts=now_ts(),
                    step=step,
                    kind="resolve",
                    severity="info",
                    title=f"Incident Resolved (Stable {rec.required_stable_steps} steps)",
                    incident_id=incident_id,
                    detail={"final_score": stability, "signal": signal},
                )
            )
            # Generate Postmortem automatically on resolve
            if self.postmortem:
                self.postmortem.generate(incident_id, step=step)
