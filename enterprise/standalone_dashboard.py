#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: standalone_dashboard.py_v1.2.py
# Version: v1.2 (Enterprise Final - Patched)
# Description: 
#   Enterprise-grade AI Operation System with Refined Governance
#   
#   Changelog (v1.2):
#     1. [Schema] Separated KNOBS (Input) from METRICS (Output) to prevent injection.
#     2. [Governance] Fixed "Deadlock Paradox" in Supreme Court (Allow safe actions during instability).
#     3. [Analysis] Action Analyzer now emits verdicts immediately upon window completion.
#   
#   [PATCHES APPLIED]:
#     ✅ Patch 1: DEMO now forces RESOLVED + POSTMORTEM (required_stable_steps = 3)
#     ✅ Patch 2: PostmortemGenerator actor extraction bug fixed
#     ✅ Patch 3: Reality weight allowed during instability (only LR blocked)
# ==============================================================================
from __future__ import annotations
import json
import os
import time
import uuid
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable
from collections import deque
from datetime import datetime

# ==============================================================================
# 0. Utilities & Constants
# ==============================================================================
def now_ts() -> float:
    return float(time.time())

def fmt_ts(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return str(ts)

# [v1.2] Schema Separation (Single Source of Truth)
# Inputs we can control
KNOB_SCHEMA = {
    "learning_rate": float,
    "reality_weight": float,
    "dropout": float,
}

# Outputs we observe
METRIC_SCHEMA = {
    "stability": float,
    "risk": float,
    "loss": float,
}

def validate_knob_keys(d: Dict[str, Any]) -> bool:
    # Only allow keys that exist in KNOB_SCHEMA
    return all(k in KNOB_SCHEMA for k in d.keys())

# ==============================================================================
# 1. Timeline & Incident Data Structures
# ==============================================================================
@dataclass
class TimelineEvent:
    ts: float
    step: int
    kind: str          # anomaly, action_apply, action_vetoed, action_effect, resolve, postmortem
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

    def create_or_update(self, incident_id: Optional[str] = None, severity: str = "info", title: str = "", step: int = 0, tags: List[str] = None, action_ids: List[str] = None, status: str = None) -> IncidentRecord:
        if incident_id is None:
            incident_id = f"inc_{int(time.time())}_{str(uuid.uuid4())[:4]}"
        
        now = time.time()
        if incident_id in self.by_id:
            rec = self.by_id[incident_id]
            rec.last_ts = now
            rec.last_step = step
            if status: rec.status = status
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
        if incident_id not in self.by_id: return
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
                if rec.status == "OPEN": rec.status = "MITIGATING"

    def summary(self, host: str = "localhost", port: int = 8080) -> Dict[str, Any]:
        records = list(self.by_id.values())
        return {
            "summary": {
                "OPEN": sum(1 for r in records if r.status == "OPEN"),
                "MITIGATING": sum(1 for r in records if r.status == "MITIGATING"),
                "RESOLVED": sum(1 for r in records if r.status == "RESOLVED")
            },
            "items": [
                {
                    **asdict(r), 
                    "created_time": fmt_ts(r.created_ts),
                    "drilldown": f"http://{host}:{port}/timeline?incident_id={r.incident_id}"
                } 
                for r in sorted(records, key=lambda r: r.last_ts, reverse=True)
            ]
        }

# ==============================================================================
# 2. Logic Engines (Analyzers, Trackers, and The Court)
# ==============================================================================

# [v1.2] The Supreme Court (Refined)
@dataclass
class Judgement:
    approved: bool
    ruling: str              # CONSTITUTIONAL | UNCONSTITUTIONAL | MISTRIAL
    reason: str
    violates: Optional[str] = None

class SupremeCourt:
    def __init__(self):
        self.constitution = [
            {
                "id": "ARTICLE_1",
                "desc": "Knob Schema Violation (Security)",
                # [v1.2] Use validate_knob_keys (Fail-Closed if unknown keys present)
                "check": lambda plan, ctx: not validate_knob_keys(plan.knobs)
            },
            {
                "id": "ARTICLE_2",
                "desc": "Unsafe Manipulation during Instability (< 0.2)",
                # [v1.2 PATCH 3] Reality weight allowed, only LR blocked during instability
                "check": lambda plan, ctx: (ctx.get("stability", 1.0) < 0.2) and ("learning_rate" in plan.knobs)
            },
            {
                "id": "ARTICLE_3",
                "desc": "Learning Rate Ceiling Exceeded (> 0.01)",
                "check": lambda plan, ctx: plan.knobs.get("learning_rate", ctx.get("learning_rate", 0)) > 0.01
            }
        ]

    def review(self, plan: ActionPlan, context: Dict[str, Any]) -> Judgement:
        for article in self.constitution:
            try:
                if article["check"](plan, context):
                    return Judgement(
                        approved=False,
                        ruling="UNCONSTITUTIONAL",
                        reason=article["desc"],
                        violates=article["id"]
                    )
            except Exception as e:
                return Judgement(False, "MISTRIAL", f"Verification failure: {e}")
        return Judgement(True, "CONSTITUTIONAL", "No violations")

# [v1.2] Action Effect Analyzer (Real-time Verdict)
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
        if action_id not in self.buffers: return
        buf = self.buffers[action_id]
        buf["samples"].append(dict(metrics or {}))
        
        # [v1.2] Immediate Finalization Check
        if len(buf["samples"]) >= self.window_steps:
            self.finalize(action_id, step=buf["start_step"] + self.window_steps)

    def finalize(self, action_id: str, step: int) -> Optional[Dict[str, Any]]:
        buf = self.buffers.pop(action_id, None)
        if not buf: return None
        samples = list(buf["samples"])
        if not samples: return None

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
        if score > 0.10: verdict = "effective"
        elif score > 0.02: verdict = "partial"

        ev_detail = {
            "kind": "action_effect", "verdict": verdict, "baseline": baseline,
            "after_avg": after, "delta": delta, "window_steps": self.window_steps,
            "score": score, "action_id": action_id
        }
        self.timeline.add(TimelineEvent(
            ts=now_ts(), step=int(step), kind="action_effect",
            severity="info" if verdict == "effective" else "warn",
            title=f"Action Effect: {verdict.upper()} (score={score:.3f})",
            incident_id=buf["incident_id"], tags=["effect", verdict], detail=ev_detail
        ))
        return ev_detail

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
        if opens: timeline_lines.append(f"[OPEN] {fmt_ts(opens[0]['ts'])} - {opens[0].get('title')}")
        for a in actions:
            tags = a.get("tags", [])
            actor = tags[1] if len(tags) > 1 else "operator"  # [PATCH 2] Fixed actor extraction
            timeline_lines.append(f"[ACTION] {fmt_ts(a['ts'])} - {a.get('title')} (Actor: {actor})")
        for e in effects:
            d = e.get("detail", {})
            timeline_lines.append(f"[EFFECT] Verdict: {d.get('verdict')} (Score: {d.get('score', 0):.2f})")
        if resolves: timeline_lines.append(f"[RESOLVE] {fmt_ts(resolves[-1]['ts'])} - {resolves[-1].get('title')}")

        pm = {"summary": summary, "timeline": timeline_lines}
        self.timeline.add(TimelineEvent(
            ts=now_ts(), step=int(step), kind="postmortem", severity="info",
            title=f"Postmortem Generated: {incident_id}", incident_id=incident_id,
            tags=["postmortem", "rca"], detail=pm
        ))
        return pm

class StabilityTracker:
    def __init__(self, registry: IncidentRegistry, timeline: IncidentTimeline,
                 effect_analyzer: Optional[ActionEffectAnalyzer] = None,
                 postmortem: Optional[PostmortemGenerator] = None):
        self.registry = registry
        self.timeline = timeline
        self.effect_analyzer = effect_analyzer
        self.postmortem = postmortem

    def observe(self, incident_id: str, step: int, stability: float, signal: Dict) -> None:
        is_stable = stability >= 0.8
        self.registry.update_stability(incident_id, is_stable)
        rec = self.registry.by_id.get(incident_id)
        if rec and rec.status == "RESOLVED" and rec.stable_steps == rec.required_stable_steps:
            self.timeline.add(TimelineEvent(
                ts=now_ts(), step=step, kind="resolve", severity="info",
                title=f"Incident Resolved (Stable {rec.required_stable_steps} steps)",
                incident_id=incident_id, detail={"final_score": stability, "signal": signal}
            ))
            if self.postmortem: self.postmortem.generate(incident_id, step=step)

@dataclass
class ActionPlan:
    action_id: str
    incident_id: str
    title: str
    knobs: Dict[str, Any]
    actor: str = "operator"
    reason: str = ""

class ActionApplicator:
    def __init__(self, registry: IncidentRegistry, timeline: IncidentTimeline, 
                 effect_analyzer: Optional[ActionEffectAnalyzer] = None, 
                 get_metrics: Optional[Callable[[], Dict[str,float]]] = None):
        self.registry = registry
        self.timeline = timeline
        self.effect_analyzer = effect_analyzer
        self.get_metrics = get_metrics
        self.court = SupremeCourt()

    def apply(self, plan: ActionPlan, get_state: Callable, set_state: Callable, step: int) -> bool:
        try:
            before = get_state()
            context = before.copy()
            if self.get_metrics: context.update(self.get_metrics())
        except Exception:
            return False

        # ⚖️ Supreme Court Review
        judgement = self.court.review(plan, context)
        if not judgement.approved:
            self.timeline.add(TimelineEvent(
                ts=now_ts(), step=step, kind="action_vetoed", severity="warn",
                title=f"Action VETOED: {plan.title}", incident_id=plan.incident_id,
                tags=["veto", "constitutional", judgement.violates],
                detail={"plan": asdict(plan), "ruling": judgement.ruling, "reason": judgement.reason, "context": context}
            ))
            print(f" 🚫 [SupremeCourt] Action Blocked: {judgement.reason}")
            return False

        # --- Constitutional Path Only ---
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

# ==============================================================================
# 3. HTML Rendering (Dashboard)
# ==============================================================================
try:
    from flask import Flask, Response, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

class EnterpriseDashboard:
    def __init__(self, timeline: IncidentTimeline, registry: IncidentRegistry, host="0.0.0.0", port=8080):
        if not FLASK_AVAILABLE: return
        self.timeline = timeline
        self.registry = registry
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/timeline")
        def timeline():
            iid = request.args.get("incident_id")
            events = self.timeline.list_recent(limit=100, incident_id=iid)
            return self._render_timeline(events, iid)

        @self.app.route("/incidents")
        @self.app.route("/")
        def incidents():
            data = self.registry.summary(self.host, self.port)
            return self._render_incidents(data)

    def _render_incidents(self, data: Dict) -> Response:
        items = data.get("items", [])
        summ = data.get("summary", {})
        html = f"""<html><head><meta charset="utf-8"><title>Incidents</title>
        <style>
            body {{font-family: system-ui; margin: 0; padding: 40px; background: #f9fafb; color: #111;}}
            .summary {{display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 40px;}}
            .card {{background: white; padding: 24px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}}
            .card h2 {{margin: 0; font-size: 36px;}}
            .list-item {{background: white; padding: 20px; border-radius: 12px; margin-bottom: 16px; border-left: 6px solid #ddd; box-shadow: 0 1px 2px rgba(0,0,0,0.05); display: flex; justify-content: space-between; align-items: center;}}
            .OPEN {{border-left-color: #ef4444;}} .MITIGATING {{border-left-color: #f59e0b;}} .RESOLVED {{border-left-color: #10b981;}}
            .badge {{padding: 4px 8px; border-radius: 6px; font-size: 12px; background: #eee; margin-right: 8px;}}
            a.btn {{text-decoration: none; background: #4f46e5; color: white; padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 14px;}}
        </style></head><body>
        <h1>📋 Operational Dashboard</h1>
        <div class="summary">
            <div class="card"><label>OPEN</label><h2 style="color:#ef4444">{summ.get('OPEN',0)}</h2></div>
            <div class="card"><label>MITIGATING</label><h2 style="color:#f59e0b">{summ.get('MITIGATING',0)}</h2></div>
            <div class="card"><label>RESOLVED</label><h2 style="color:#10b981">{summ.get('RESOLVED',0)}</h2></div>
        </div>
        <h3>Recent Incidents</h3>"""
        for item in items:
            html += f"""
            <div class="list-item {item['status']}">
                <div>
                    <div><span class="badge" style="background:#333;color:white">{item['status']}</span><span class="badge">{item['severity'].upper()}</span><span style="color:#666;font-size:14px">{item['created_time']}</span></div>
                    <h3 style="margin: 8px 0 4px 0">{item['title']}</h3>
                    <div style="font-size:14px;color:#666">ID: {item['incident_id']} | Stability: {int(item['stability_score']*100)}%</div>
                </div>
                <a href="{item['drilldown']}" class="btn">View Timeline →</a>
            </div>"""
        return Response(html + "</body></html>", mimetype="text/html")

    def _render_timeline(self, events: List, iid: str) -> Response:
        html = f"""<html><head><meta charset="utf-8"><title>Timeline</title>
        <style>
            body {{font-family: system-ui; margin: 40px auto; max-width: 900px; background: #f9fafb; color: #111;}}
            .event {{background: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; border-left: 4px solid #ccc; box-shadow: 0 1px 3px rgba(0,0,0,0.05);}}
            .action_apply {{border-left-color: #4f46e5;}} .anomaly {{border-left-color: #ef4444;}} 
            .resolve {{border-left-color: #10b981;}} .action_effect {{border-left-color: #0ea5e9;}} .postmortem {{border-left-color: #111827;}}
            .action_vetoed {{border-left-color: #000; background: #fff1f2;}} 
            .meta {{font-size: 13px; color: #666; margin-bottom: 8px; display: flex; gap: 10px;}}
            table.diff {{width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px;}}
            table.diff th {{text-align: left; background: #f3f4f6; padding: 8px; border-bottom: 1px solid #ddd;}}
            table.diff td {{padding: 8px; border-bottom: 1px solid #eee; font-family: monospace;}}
            .val-old {{color: #999; text-decoration: line-through; margin-right: 8px;}}
            .val-new {{color: #059669; font-weight: bold;}}
            pre {{background: #f3f4f6; padding: 10px; border-radius: 8px; overflow-x: auto;}}
        </style></head><body>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;">
            <div><h1>📊 Timeline View</h1><div style="color:#666">Incident: {iid or 'All Recent'}</div></div>
            <a href="/incidents" style="text-decoration:none;color:#4f46e5;font-weight:bold">← Back</a>
        </div>"""
        
        for ev in reversed(events):
            detail = ev.get("detail", {})
            diff_html, effect_html, pm_html, veto_html = "", "", "", ""
            
            # Diff Table
            if "before" in detail and "after" in detail:
                rows = ""
                before, after = detail["before"], detail["after"]
                for k in sorted(set(before.keys()) | set(after.keys())):
                    if before.get(k) != after.get(k):
                        rows += f"<tr><td>{k}</td><td><span class='val-old'>{before.get(k)}</span> <span class='val-new'>{after.get(k)}</span></td></tr>"
                if rows: diff_html = f"<table class='diff'><tr><th>Parameter</th><th>Change (Before &rarr; After)</th></tr>{rows}</table>"
                detail = {k:v for k,v in detail.items() if k not in ["before", "after"]}

            if ev['kind'] == "action_vetoed":
                veto_html = f"<div style='margin-top:10px;padding:10px;background:#000;color:white;border-radius:6px;font-family:monospace'>🚫 RULING: {detail.get('ruling')} <br> REASON: {detail.get('reason')}</div>"

            if ev['kind'] == "action_effect":
                verdict = detail.get("verdict", "UNKNOWN")
                bg, col = ("#d1fae5", "#065f46") if verdict == "effective" else ("#fee2e2", "#991b1b")
                effect_html = f"<div style='margin-top:12px;padding:12px;background:{bg};border-radius:8px;color:{col};font-weight:bold'>⚖️ Verdict: {verdict.upper()} (Score: {detail.get('score',0):.2f})</div>"

            if ev['kind'] == "postmortem":
                summ = detail.get("summary", {})
                lines = "".join([f"<li style='margin-bottom:6px'>{line}</li>" for line in detail.get("timeline", [])])
                pm_html = f"<div style='background:#fffbeb;border:1px solid #fcd34d;padding:16px;border-radius:8px;margin-top:12px;color:#92400e'><div style='font-weight:bold;margin-bottom:8px'>📝 Postmortem Summary</div><div>Duration: {summ.get('num_actions',0)} Actions / Verdict: {summ.get('final_verdict','N/A')}</div><ul style='margin-top:12px;padding-left:20px;line-height:1.5'>{lines}</ul></div>"

            html += f"""<div class="event {ev['kind']}"><div class="meta"><span>{fmt_ts(ev['ts'])}</span><span style="font-weight:bold;color:#444">{ev['kind'].upper()}</span><span>Step {ev['step']}</span></div><h3 style="margin: 0 0 10px 0">{ev['title']}</h3>{veto_html}{diff_html}{effect_html}{pm_html}<pre>{json.dumps(detail, indent=2)}</pre></div>"""
        return Response(html + "</body></html>", mimetype="text/html")

    def start(self):
        print(f"🚀 Dashboard running at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False)

# ==============================================================================
# 4. Demo Simulation (Includes Veto & Schema Check - PATCHED)
# ==============================================================================
def demo():
    # 1. Initialize
    timeline = IncidentTimeline()
    registry = IncidentRegistry()
    effect = ActionEffectAnalyzer(timeline, window_steps=5)
    pm = PostmortemGenerator(timeline)
    tracker = StabilityTracker(registry, timeline, effect_analyzer=effect, postmortem=pm)
    
    # Kernel State (Mock)
    current_metrics = {"loss": 2.8, "risk": 0.9, "stability": 0.15} # Unstable start
    def collect_metrics(): return current_metrics.copy()
    
    kernel_knobs = {"learning_rate": 0.005, "reality_weight": 1.0, "dropout": 0.1}
    def get_state(): return kernel_knobs.copy()
    def set_state(s): print(f" >>> Kernel Update: {s}"); kernel_knobs.update(s)

    applicator = ActionApplicator(registry, timeline, effect_analyzer=effect, get_metrics=collect_metrics)

    # 2. Scenario
    print("🔥 [Step 100] Anomaly Detected! (Unstable: 0.15)")
    inc = registry.create_or_update(severity="high", title="Critical Risk Spike", step=100, tags=["anomaly"])
    # [PATCH 1] Force RESOLVED in demo by lowering required stable steps
    registry.by_id[inc.incident_id].required_stable_steps = 3
    timeline.add(TimelineEvent(ts=now_ts(), step=100, kind="anomaly", severity="high", title="Risk > 0.8", incident_id=inc.incident_id))

    # [SCENARIO A] Schema Violation (Setting 'risk' directly)
    print("😈 [Step 101] Autopilot tries to cheat (Set Risk=0)...")
    cheat_plan = ActionPlan(action_id="act_cheat", incident_id=inc.incident_id, title="Cheat Mode", knobs={"risk": 0.0}, reason="Quick fix")
    applicator.apply(cheat_plan, get_state, set_state, 101) # Should fail by ARTICLE_1

    # [SCENARIO B] Unsafe Action during Instability (LR change)
    print("😈 [Step 102] Autopilot tries unsafe action (LR change) while unstable...")
    unsafe_plan = ActionPlan(action_id="act_unsafe", incident_id=inc.incident_id, title="Boost LR", knobs={"learning_rate": 0.02}, reason="Boost")
    applicator.apply(unsafe_plan, get_state, set_state, 102) # Should fail by ARTICLE_2

    # [SCENARIO C] Safe Action (Dropout) - Should Pass! (Paradox Fix)
    print("🛡️ [Step 103] Applying Safe Action (Dropout)...")
    safe_plan = ActionPlan(action_id="act_safe", incident_id=inc.incident_id, title="Increase Dropout", knobs={"dropout": 0.5}, reason="Dampen")
    applicator.apply(safe_plan, get_state, set_state, 103)

    # [SCENARIO D] Reality Weight during Instability - Should Pass with PATCH 3!
    print("🛡️ [Step 104] Applying Reality Weight during instability (PATCH 3 allows)...")
    reality_plan = ActionPlan(action_id="act_reality", incident_id=inc.incident_id, title="Increase Reality Weight", knobs={"reality_weight": 2.0}, reason="Ground to reality")
    applicator.apply(reality_plan, get_state, set_state, 104)

    # Simulate Stabilization
    print("⏳ [Step 106-115] Stabilization...")
    for i in range(10):
        step = 106 + i
        current_metrics["loss"] -= 0.1
        current_metrics["risk"] -= 0.05
        current_metrics["stability"] += 0.08
        
        effect.collect("act_safe", collect_metrics()) # Real-time check
        effect.collect("act_reality", collect_metrics()) # For second action
        tracker.observe(inc.incident_id, step, current_metrics["stability"], current_metrics)
        time.sleep(0.1)

    print("✅ [Step 120] Resolution...")
    tracker.observe(inc.incident_id, 120, 0.95, current_metrics)

    # 3. Start Dashboard
    dash = EnterpriseDashboard(timeline, registry)
    dash.start()

if __name__ == "__main__":
    demo()
