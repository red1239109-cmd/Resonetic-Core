# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional
from .timeline import TimelineEvent, now_ts

@dataclass
class ActionPlan:
    action_id: str
    incident_id: str
    title: str
    knobs: Dict[str, Any]
    actor: str = "operator"
    reason: str = ""

class ActionApplicator:
    def __init__(self, registry, timeline, 
                 effect_analyzer: Optional = None, 
                 get_metrics: Optional[Callable[[], Dict[str,float]]] = None):
        self.registry = registry
        self.timeline = timeline
        self.effect_analyzer = effect_analyzer
        self.get_metrics = get_metrics

    def apply(self, plan: ActionPlan, get_state: Callable, set_state: Callable, step: int) -> bool:
        try:
            before = get_state()
        except Exception:
            return False

        # [HOOK] Snapshot Baseline
        if self.effect_analyzer and self.get_metrics:
            try: base_metrics = self.get_metrics()
            except: base_metrics = {}
            self.effect_analyzer.start(plan.incident_id, plan.action_id, step, base_metrics)

        after = before.copy()
        after.update(plan.knobs)

        self.registry.create_or_update(
            incident_id=plan.incident_id, status="MITIGATING", step=step,
            tags=["intervention", plan.actor], action_ids=[plan.action_id]
        )
        set_state(after)

        self.timeline.add(TimelineEvent(
            ts=now_ts(), step=step, kind="action_apply", severity="info",
            title=f"Action Applied: {plan.title}", incident_id=plan.incident_id,
            tags=["action", plan.actor],
            detail={"before": before, "after": after, "reason": plan.reason, "action_id": plan.action_id}
        ))
        return True
