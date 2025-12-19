# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Callable
from . import now_ts
from .timeline import IncidentTimeline, TimelineEvent
from .incident import IncidentRegistry
from .effect import ActionEffectAnalyzer

# Schema Definitions
KNOB_SCHEMA = {
    "learning_rate": float,
    "reality_weight": float,
    "dropout": float,
}

METRIC_SCHEMA = {
    "stability": float,
    "risk": float,
    "loss": float,
}

def validate_knob_keys(d: Dict[str, Any]) -> bool:
    return all(k in KNOB_SCHEMA for k in d.keys())

@dataclass
class Judgement:
    approved: bool
    ruling: str
    reason: str
    violates: Optional[str] = None

@dataclass
class ActionPlan:
    action_id: str
    incident_id: str
    title: str
    knobs: Dict[str, Any]
    actor: str = "operator"
    reason: str = ""

class SupremeCourt:
    def __init__(self):
        self.constitution = [
            {
                "id": "ARTICLE_1",
                "desc": "Knob Schema Violation (Security)",
                "check": lambda plan, ctx: not validate_knob_keys(plan.knobs),
            },
            {
                "id": "ARTICLE_2",
                "desc": "Unsafe Manipulation during Instability (< 0.2) [LR only]",
                "check": lambda plan, ctx: (ctx.get("stability", 1.0) < 0.2) and ("learning_rate" in plan.knobs),
            },
            {
                "id": "ARTICLE_3",
                "desc": "Learning Rate Ceiling Exceeded (> 0.01)",
                "check": lambda plan, ctx: plan.knobs.get("learning_rate", ctx.get("learning_rate", 0)) > 0.01,
            },
        ]

    def review(self, plan: ActionPlan, context: Dict[str, Any]) -> Judgement:
        for article in self.constitution:
            try:
                if article["check"](plan, context):
                    return Judgement(
                        approved=False,
                        ruling="UNCONSTITUTIONAL",
                        reason=article["desc"],
                        violates=article["id"],
                    )
            except Exception as e:
                return Judgement(False, "MISTRIAL", f"Verification failure: {e}")
        return Judgement(True, "CONSTITUTIONAL", "No violations")

class ActionApplicator:
    def __init__(
        self,
        registry: IncidentRegistry,
        timeline: IncidentTimeline,
        effect_analyzer: Optional[ActionEffectAnalyzer] = None,
        get_metrics: Optional[Callable[[], Dict[str, float]]] = None,
    ):
        self.registry = registry
        self.timeline = timeline
        self.effect_analyzer = effect_analyzer
        self.get_metrics = get_metrics
        self.court = SupremeCourt()

    def apply(self, plan: ActionPlan, get_state: Callable, set_state: Callable, step: int) -> bool:
        try:
            before = get_state()
            context = before.copy()
            if self.get_metrics:
                context.update(self.get_metrics())
        except Exception:
            return False

        judgement = self.court.review(plan, context)
        if not judgement.approved:
            self.timeline.add(
                TimelineEvent(
                    ts=now_ts(),
                    step=step,
                    kind="action_vetoed",
                    severity="warn",
                    title=f"Action VETOED: {plan.title}",
                    incident_id=plan.incident_id,
                    tags=["veto", "constitutional", judgement.violates],
                    detail={
                        "plan": asdict(plan),
                        "ruling": judgement.ruling,
                        "reason": judgement.reason,
                        "context": context,
                    },
                )
            )
            print(f" 🚫 [SupremeCourt] Action Blocked: {judgement.reason}")
            return False

        if self.effect_analyzer and self.get_metrics:
            try:
                base_metrics = self.get_metrics()
            except Exception:
                base_metrics = {}
            self.effect_analyzer.start(plan.incident_id, plan.action_id, step, base_metrics)

        after = before.copy()
        after.update(plan.knobs)

        self.registry.create_or_update(
            incident_id=plan.incident_id,
            status="MITIGATING",
            step=step,
            tags=["intervention", plan.actor],
            action_ids=[plan.action_id],
        )
        set_state(after)

        self.timeline.add(
            TimelineEvent(
                ts=now_ts(),
                step=step,
                kind="action_apply",
                severity="info",
                title=f"Action Applied: {plan.title}",
                incident_id=plan.incident_id,
                tags=["action", plan.actor],
                detail={
                    "before": before,
                    "after": after,
                    "reason": plan.reason,
                    "action_id": plan.action_id,
                },
            )
        )
        return True
