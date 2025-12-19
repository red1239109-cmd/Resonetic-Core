# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
import sys
import os
import time

# Add root directory to sys.path to allow imports from src and dashboard
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.timeline import IncidentTimeline, TimelineEvent
from src.incident import IncidentRegistry
from src.effect import ActionEffectAnalyzer
from src.postmortem import PostmortemGenerator
from src.stability import StabilityTracker
from src.action import ActionApplicator, ActionPlan
from src import now_ts
from dashboard.app import create_app

def demo():
    # 1. Initialize
    timeline = IncidentTimeline()
    registry = IncidentRegistry()
    effect = ActionEffectAnalyzer(timeline, window_steps=5)
    pm = PostmortemGenerator(timeline)
    tracker = StabilityTracker(registry, timeline, effect_analyzer=effect, postmortem=pm)

    # Kernel State (Mock)
    current_metrics = {"loss": 2.8, "risk": 0.9, "stability": 0.15}
    def collect_metrics(): return current_metrics.copy()

    kernel_knobs = {"learning_rate": 0.005, "reality_weight": 1.0, "dropout": 0.1}
    def get_state(): return kernel_knobs.copy()
    def set_state(s):
        print(f" >>> Kernel Update: {s}")
        kernel_knobs.update(s)

    applicator = ActionApplicator(registry, timeline, effect_analyzer=effect, get_metrics=collect_metrics)

    # 2. Scenario
    print("🔥 [Step 100] Anomaly Detected!")
    inc = registry.create_or_update(severity="high", title="Critical Risk Spike", step=100, tags=["anomaly"])
    timeline.add(TimelineEvent(ts=now_ts(), step=100, kind="anomaly", severity="high", title="Risk > 0.8", incident_id=inc.incident_id))

    # PATCH: Demo guarantees RESOLVE
    registry.by_id[inc.incident_id].required_stable_steps = 3

    # A) Schema Violation
    print("😈 [Step 101] Autopilot tries to cheat (Set Risk=0)...")
    cheat_plan = ActionPlan(action_id="act_cheat", incident_id=inc.incident_id, title="Cheat Mode", knobs={"risk": 0.0}, actor="autopilot", reason="Quick fix")
    applicator.apply(cheat_plan, get_state, set_state, 101)

    # B) Unsafe LR during instability
    print("😈 [Step 102] Autopilot tries unsafe action (LR change) while unstable...")
    unsafe_plan = ActionPlan(action_id="act_unsafe", incident_id=inc.incident_id, title="Boost LR", knobs={"learning_rate": 0.02}, actor="autopilot", reason="Boost")
    applicator.apply(unsafe_plan, get_state, set_state, 102)

    # C) Safe Action
    print("🛡️ [Step 103] Applying Safe Action (Dropout)...")
    safe_plan = ActionPlan(action_id="act_safe", incident_id=inc.incident_id, title="Increase Dropout", knobs={"dropout": 0.5}, actor="autopilot", reason="Dampen")
    applicator.apply(safe_plan, get_state, set_state, 103)

    print("⏳ [Step 106-115] Stabilization...")
    for i in range(10):
        step = 106 + i
        current_metrics["loss"] = max(0.7, current_metrics["loss"] - 0.12)
        current_metrics["risk"] = max(0.05, current_metrics["risk"] - 0.07)
        current_metrics["stability"] = min(1.0, current_metrics["stability"] + 0.12)

        effect.collect("act_safe", collect_metrics())
        tracker.observe(inc.incident_id, step, current_metrics["stability"], current_metrics)
        time.sleep(0.05)

    print("✅ [Step 120] Resolution check...")
    tracker.observe(inc.incident_id, 120, 0.95, current_metrics)

    # 3. Start Dashboard
    print(f"🚀 Dashboard running at http://localhost:8080")
    app = create_app(timeline, registry)
    app.run(host="0.0.0.0", port=8080, debug=False)

if __name__ == "__main__":
    demo()
