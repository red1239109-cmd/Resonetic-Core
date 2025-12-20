#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: gdr_v2_1.py
# Product: Governed Data Refinery (GDR) v2.1 (Hardened Production Edition)
# Changelog (v2.1):
#   + Log Rotation: Automatically rotates timeline.jsonl when > 10MB.
#   + Memory Guard: Monitors RSS usage and triggers 'High Memory' incidents.
#   + Port Check: Gracefully handles port conflicts on dashboard startup.
# ==============================================================================

from __future__ import annotations
import json
import os
import time
import uuid
import sqlite3
import shutil
import statistics
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable
from collections import deque
from pathlib import Path
from scipy.stats import entropy

# --- Dependencies ---
try:
    from flask import Flask, request, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask not found. Dashboard disabled.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not found. Memory monitoring disabled.")

# ==============================================================================
# 0. Config & Utilities
# ==============================================================================
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)
DB_PATH = RUNS_DIR / "governance.db"
LOG_PATH = RUNS_DIR / "timeline.jsonl"
MAX_LOG_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit for log rotation
MEMORY_LIMIT_MB = 1024  # 1GB soft limit for warning

def now_ts() -> float:
    return float(time.time())

def fmt_ts(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return str(ts)

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and (x == x) and not np.isinf(x)

KNOB_SCHEMA = {"learning_rate": float, "reality_weight": float, "dropout": float}
METRIC_SCHEMA = {"stability": float, "risk": float, "loss": float}

def validate_knob_keys(d: Dict[str, Any]) -> bool:
    return all(k in KNOB_SCHEMA for k in d.keys())

# ==============================================================================
# 1. Timeline & Persistence (with Log Rotation)
# ==============================================================================
@dataclass
class TimelineEvent:
    ts: float
    step: int
    kind: str
    severity: str
    title: str
    detail: Dict[str, Any] = field(default_factory=dict)
    incident_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

class IncidentTimeline:
    def __init__(self, maxlen: int = 5000, jsonl_path: str = str(LOG_PATH)):
        self.buf = deque(maxlen=int(maxlen))
        self.jsonl_path = Path(jsonl_path)

    def _rotate_logs(self):
        """Rotates logs if size exceeds limit."""
        if not self.jsonl_path.exists():
            return
        try:
            if self.jsonl_path.stat().st_size > MAX_LOG_SIZE_BYTES:
                timestamp = int(time.time())
                backup_path = self.jsonl_path.with_name(f"timeline_{timestamp}.jsonl.bak")
                shutil.move(str(self.jsonl_path), str(backup_path))
                print(f"🔄 [Log] Rotated {self.jsonl_path.name} -> {backup_path.name}")
        except Exception as e:
            print(f"⚠️ [Log] Rotation failed: {e}")

    def add(self, ev: TimelineEvent) -> str:
        ev_dict = asdict(ev)
        self.buf.append(ev_dict)
        try:
            self._rotate_logs() # Check rotation before write
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev_dict, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return "evt_" + str(uuid.uuid4())[:8]

    def list_recent(self, limit: int = 50, incident_id: Optional[str] = None) -> List[Dict[str, Any]]:
        events = list(self.buf)
        if incident_id:
            events = [e for e in events if e.get("incident_id") == incident_id]
        return events[-int(limit):]

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
    tags: List[str] = field(default_factory=list)
    action_ids: List[str] = field(default_factory=list)

class PersistentIncidentRegistry:
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self.by_id: Dict[str, IncidentRecord] = {}
        self._init_db()
        self._load_from_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    data TEXT,
                    last_ts REAL
                )
            """)

    def _load_from_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT data FROM incidents")
                for row in cursor:
                    rec_dict = json.loads(row[0])
                    self.by_id[rec_dict["incident_id"]] = IncidentRecord(**rec_dict)
            print(f"📦 [DB] Loaded {len(self.by_id)} incidents")
        except Exception as e:
            print(f"⚠️ [DB] Load failed: {e}")

    def _save_to_db(self, rec: IncidentRecord):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO incidents (incident_id, data, last_ts) VALUES (?, ?, ?)",
                    (rec.incident_id, json.dumps(asdict(rec)), rec.last_ts)
                )
        except Exception:
            pass

    def create_or_update(self, incident_id: Optional[str] = None, severity: str = "info", title: str = "", step: int = 0, tags: Optional[List[str]] = None, action_ids: Optional[List[str]] = None, status: Optional[str] = None) -> IncidentRecord:
        if incident_id is None:
            incident_id = f"inc_{int(time.time())}_{str(uuid.uuid4())[:4]}"
        now = time.time()
        
        if incident_id in self.by_id:
            rec = self.by_id[incident_id]
            rec.last_ts = now
            rec.last_step = step
            if status: rec.status = status
            if tags: rec.tags = list(set(rec.tags + tags))
            if action_ids: rec.action_ids = list(set(rec.action_ids + action_ids))
            if rec.status == "RESOLVED" and severity in ["warn", "high"]:
                rec.status = "OPEN"
                rec.severity = severity
                rec.stable_steps = 0
        else:
            rec = IncidentRecord(incident_id, status or "OPEN", severity, title, now, now, step, 0, 10, 0.0, tags or [], action_ids or [])
            self.by_id[incident_id] = rec
        
        self._save_to_db(rec)
        return rec

    def update_stability(self, incident_id: str, is_stable: bool):
        if incident_id not in self.by_id: return
        rec = self.by_id[incident_id]
        rec.stable_steps = rec.stable_steps + 1 if is_stable else 0
        if rec.required_stable_steps > 0:
            rec.stability_score = min(1.0, rec.stable_steps / rec.required_stable_steps)
        if rec.status in ["OPEN", "MITIGATING"]:
            if rec.stability_score >= 1.0: rec.status = "RESOLVED"
            elif rec.stability_score >= 0.1 and rec.status == "OPEN": rec.status = "MITIGATING"
        self._save_to_db(rec)

    def summary(self) -> Dict[str, Any]:
        records = list(self.by_id.values())
        return {
            "summary": {
                "OPEN": sum(1 for r in records if r.status == "OPEN"),
                "MITIGATING": sum(1 for r in records if r.status == "MITIGATING"),
                "RESOLVED": sum(1 for r in records if r.status == "RESOLVED"),
            },
            "items": [asdict(r) for r in sorted(records, key=lambda r: r.last_ts, reverse=True)]
        }

# ==============================================================================
# 2. Logic Engines (Kant, Rawls, ResourceMonitor)
# ==============================================================================
class ResourceMonitor:
    """Watches memory/CPU usage and triggers alerts."""
    def __init__(self, timeline: IncidentTimeline, limit_mb: int = MEMORY_LIMIT_MB):
        self.timeline = timeline
        self.limit_mb = limit_mb
        self.process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
        self.last_check = 0.0

    def check(self, step: int):
        if not self.process or (time.time() - self.last_check < 5.0):
            return # Check every 5s
        
        self.last_check = time.time()
        mem_mb = self.process.memory_info().rss / (1024 * 1024)
        
        if mem_mb > self.limit_mb:
            self.timeline.add(TimelineEvent(
                now_ts(), step, "resource_warning", "warn",
                f"High Memory Usage: {mem_mb:.1f}MB",
                detail={"rss_mb": mem_mb, "limit": self.limit_mb},
                tags=["resource", "memory"]
            ))
            # Optional: Call Python GC
            import gc; gc.collect()

@dataclass
class Judgement:
    approved: bool
    ruling: str
    reason: str
    violates: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

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
            {"id": "ART_1", "desc": "Schema Violation", "check": lambda p, c: not validate_knob_keys(p.knobs)},
            {"id": "ART_2", "desc": "Unsafe Ops (Instability)", "check": lambda p, c: (c.get("stability", 1.0) < 0.2) and any(k in p.knobs for k in ["learning_rate", "reality_weight"])},
            {"id": "ART_3", "desc": "LR Ceiling (>0.01)", "check": lambda p, c: p.knobs.get("learning_rate", 0.0) > 0.01},
        ]
    def review(self, plan: ActionPlan, context: Dict[str, Any]) -> Judgement:
        for art in self.constitution:
            if art["check"](plan, context):
                return Judgement(False, "UNCONSTITUTIONAL", art["desc"], violates=art["id"])
        return Judgement(True, "CONSTITUTIONAL", "Pass")

class RawlsCouncil:
    def __init__(self):
        self.max_harm = {"risk": 0.10, "loss": 0.30, "stability": -0.10}
        self.maximin_floor = -0.05
        self.utility_w = {"stability": 1.0, "risk": -1.0, "loss": -0.5}

    def review(self, plan: ActionPlan, base: Dict, proj: Dict) -> Judgement:
        delta = {k: float(proj.get(k,0) - base.get(k,0)) for k in METRIC_SCHEMA if k in base and k in proj}
        
        # No-Harm
        harms = []
        for k, thr in self.max_harm.items():
            if k not in delta: continue
            if (k in ["risk","loss"] and delta[k]>thr) or (k=="stability" and delta[k]<thr):
                harms.append((k, delta[k]))
        if harms: return Judgement(False, "UNJUST", "Harm Violation", violates="RAWLS_HARM", extra={"harms": harms})

        # Maximin
        u = {k: delta.get(k,0)*w for k,w in self.utility_w.items() if k in delta}
        worst = min(u.values()) if u else 0.0
        if worst < self.maximin_floor:
            return Judgement(False, "UNJUST", "Maximin Violation", violates="RAWLS_MAXIMIN", extra={"worst": worst})
        
        return Judgement(True, "JUST", "Pass", extra={"delta": delta})

class ExplainCardBuilder:
    def build(self, batch_id: str, col: str, decision: str, scores: Dict, threshold: float, reason: str = ""):
        return {
            "batch_id": batch_id, "column": col, "decision": decision,
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "threshold": round(threshold, 3),
            "headline": f"[{decision}] '{col}' (Gold: {scores['gold_score']:.2f}) {reason}",
            "ts": now_ts()
        }

# ==============================================================================
# 3. Engines: Refinery & Governor
# ==============================================================================
class DataRefineryEngine:
    def __init__(self, timeline: IncidentTimeline, target_col: Optional[str] = None):
        self.timeline = timeline
        self.target_col = target_col
        self.threshold = 0.55
        self.target_retention = 0.35
        self.kp, self.ki = 0.25, 0.05
        self.card_builder = ExplainCardBuilder()

    def _calc_scores(self, s: pd.Series, target_s: Optional[pd.Series]) -> Dict[str, float]:
        missing = s.isna().mean()
        nunique = s.nunique()
        qual = (1.0 - missing) * min(1.0, nunique / max(2.0, np.sqrt(len(s)))) if len(s)>0 else 0.0
        vc = s.dropna().astype(str).value_counts(normalize=True)
        ent = entropy(vc.values) / np.log(max(2, len(vc))) if len(vc)>1 else 0.0
        rel = 0.0
        if target_s is not None:
            try: rel = abs(pd.to_numeric(s, errors='coerce').corr(pd.to_numeric(target_s, errors='coerce')))
            except: pass
            if np.isnan(rel): rel = 0.0
        gold = (0.45 * ent) + (0.40 * qual) + (0.15 * rel)
        return {"gold_score": gold, "quality": qual, "entropy": ent, "relevance": rel}

    def refine(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
        batch_id = "batch_" + str(uuid.uuid4())[:6]
        kept, cards = [], []
        target_s = df[self.target_col] if self.target_col and self.target_col in df else None
        
        for col in df.columns:
            scores = self._calc_scores(df[col], target_s)
            decision = "KEEP" if (col == self.target_col or scores['gold_score'] >= self.threshold) else "DROP"
            if decision == "KEEP": kept.append(col)
            cards.append(self.card_builder.build(batch_id, col, decision, scores, self.threshold))

        retention = len(kept) / max(1, len(df.columns))
        self.threshold = np.clip(self.threshold + (self.kp * (retention - self.target_retention)), 0.1, 0.9)
        self.timeline.add(TimelineEvent(now_ts(), 0, "refine_batch", "info", f"Refined: {len(kept)} kept", detail={"retention": retention}))
        return df[kept], {"retention": retention}, cards

class TrainingGovernor:
    def __init__(self, registry: PersistentIncidentRegistry, timeline: IncidentTimeline):
        self.registry = registry
        self.timeline = timeline
        self.kant = SupremeCourt()
        self.rawls = RawlsCouncil()
        # Effect Analyzer stub (simplified)
        self.history = []

    def apply(self, plan: ActionPlan, get_state: Callable, set_state: Callable, metrics: Dict) -> bool:
        state = get_state()
        ctx = {**state, **metrics}
        
        # 1. Kant (Hard)
        j_k = self.kant.review(plan, ctx)
        if not j_k.approved:
            self._log_veto(plan, j_k, "KANT")
            return False
            
        # 2. Rawls (Fairness Prediction)
        proj = metrics.copy()
        if "learning_rate" in plan.knobs:
            f = plan.knobs["learning_rate"] / state.get("learning_rate", 0.001)
            proj["loss"] *= (1.0 - (f-1)*0.1)
            proj["risk"] *= (1.0 + (f-1)*0.2)
            
        j_r = self.rawls.review(plan, metrics, proj)
        if not j_r.approved:
            self._log_veto(plan, j_r, "RAWLS")
            return False
            
        # 3. Apply
        set_state({**state, **plan.knobs})
        self.registry.create_or_update(plan.incident_id, status="MITIGATING", action_ids=[plan.action_id])
        self.timeline.add(TimelineEvent(now_ts(), 0, "action_apply", "info", f"Applied: {plan.title}", incident_id=plan.incident_id, detail=plan.knobs))
        return True

    def _log_veto(self, plan: ActionPlan, j: Judgement, auth: str):
        self.timeline.add(TimelineEvent(now_ts(), 0, f"veto_{auth.lower()}", "warn", f"{auth} VETO: {plan.title}", incident_id=plan.incident_id, detail={"reason": j.reason}))
        print(f"🚫 [{auth}] Blocked: {j.reason}")

# ==============================================================================
# 4. Dashboard (Flask)
# ==============================================================================
class Dashboard:
    def __init__(self, registry: PersistentIncidentRegistry, timeline: IncidentTimeline, port=8080):
        self.registry = registry
        self.timeline = timeline
        self.port = port
        self.app = Flask(__name__)
        
        @self.app.route("/")
        def index():
            summ = self.registry.summary()
            html = "<h1>GDR Dashboard</h1>"
            html += f"<div>Open: {summ['summary']['OPEN']} | Mitigating: {summ['summary']['MITIGATING']}</div><hr>"
            for item in summ['items']:
                html += f"<div><b>[{item['status']}] {item['title']}</b> (ID: {item['incident_id']}) <a href='/timeline?id={item['incident_id']}'>Trace</a></div>"
            return html

        @self.app.route("/timeline")
        def timeline_view():
            iid = request.args.get("id")
            events = self.timeline.list_recent(100, iid)
            html = f"<h1>Timeline: {iid or 'All'}</h1><a href='/'>Back</a><hr>"
            for ev in reversed(events):
                color = "red" if ev['severity'] == "warn" else "black"
                html += f"<div style='color:{color}; margin-bottom:10px; border-left:4px solid #ccc; padding-left:10px'>"
                html += f"<small>{fmt_ts(ev['ts'])}</small> <b>{ev['kind'].upper()}</b>: {ev['title']}<br>"
                html += f"<pre>{json.dumps(ev['detail'], indent=2)}</pre></div>"
            return html

    def run(self):
        if not FLASK_AVAILABLE: return
        print(f"🚀 Dashboard: http://localhost:{self.port}")
        # Graceful port handling
        try:
            self.app.run(port=self.port, debug=False, use_reloader=False)
        except OSError:
            print(f"⚠️ Port {self.port} busy. Dashboard failed to start.")

# ==============================================================================
# 5. Demo Simulation
# ==============================================================================
def demo():
    print("🤖 GDR v2.1 Starting...")
    reg = PersistentIncidentRegistry()
    tl = IncidentTimeline()
    res_mon = ResourceMonitor(tl)
    refinery = DataRefineryEngine(tl, target_col="income")
    governor = TrainingGovernor(reg, tl)
    
    # 1. Refine Data
    print("\n--- [Phase 1] Data Refinery ---")
    raw_df = pd.DataFrame({
        "income": [5000, 5100, 4900, 5050, 5200],
        "uuid": [str(uuid.uuid4()) for _ in range(5)],
        "signal": [0.1, 0.2, 0.1, 0.3, 0.2],
        "noise": [99, 99, 99, 99, 99]
    })
    clean_df, meta, cards = refinery.refine(raw_df)
    print(f"Refined Config: Threshold={meta['retention']:.2f}")
    for c in cards: print(f"  {c['headline']}")

    # 2. Governance Loop
    print("\n--- [Phase 2] Governance Loop ---")
    state = {"learning_rate": 0.005, "reality_weight": 1.0}
    metrics = {"loss": 2.5, "risk": 0.8, "stability": 0.15} # Unstable
    inc = reg.create_or_update(severity="high", title="Unstable Training detected")
    
    # Scenarios
    plan_a = ActionPlan("act_1", inc.incident_id, "Force Metrics", {"risk": 0.0})
    governor.apply(plan_a, lambda: state, lambda x: state.update(x), metrics)
    
    plan_b = ActionPlan("act_2", inc.incident_id, "Boost LR", {"learning_rate": 0.02})
    governor.apply(plan_b, lambda: state, lambda x: state.update(x), metrics)
    
    metrics["stability"] = 0.9 # Stabilized
    plan_c = ActionPlan("act_3", inc.incident_id, "Fair Adjust", {"learning_rate": 0.004})
    if governor.apply(plan_c, lambda: state, lambda x: state.update(x), metrics):
        print("✅ Fair action applied.")

    # 3. Resource Check
    res_mon.check(step=1)

    # 4. Dashboard
    Dashboard(reg, tl).run()

if __name__ == "__main__":
    demo()
