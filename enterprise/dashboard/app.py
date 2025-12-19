# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
try:
    from flask import Flask, Response, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

import json
import time
from typing import Dict, List, Any

def fmt_ts(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return str(ts)

class EnterpriseDashboard:
    def __init__(self, timeline, registry, host="0.0.0.0", port=8080):
        if not FLASK_AVAILABLE: 
            print("❌ Flask not available")
            return
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
            diff_html, effect_html, pm_html = "", "", ""
            
            # Diff Table
            if "before" in detail and "after" in detail:
                rows = ""
                before, after = detail["before"], detail["after"]
                for k in sorted(set(before.keys()) | set(after.keys())):
                    if before.get(k) != after.get(k):
                        rows += f"<tr><td>{k}</td><td><span class='val-old'>{before.get(k)}</span> <span class='val-new'>{after.get(k)}</span></td></tr>"
                if rows: diff_html = f"<table class='diff'><tr><th>Parameter</th><th>Change (Before &rarr; After)</th></tr>{rows}</table>"
                detail = {k:v for k,v in detail.items() if k not in ["before", "after"]}

            # Effect Badge
            if ev['kind'] == "action_effect":
                verdict = detail.get("verdict", "UNKNOWN")
                bg, col = ("#d1fae5", "#065f46") if verdict == "effective" else ("#fee2e2", "#991b1b")
                effect_html = f"<div style='margin-top:12px;padding:12px;background:{bg};border-radius:8px;color:{col};font-weight:bold'>⚖️ Verdict: {verdict.upper()} (Score: {detail.get('score',0):.2f})</div>"

            # Postmortem Report
            if ev['kind'] == "postmortem":
                summ = detail.get("summary", {})
                lines = "".join([f"<li style='margin-bottom:6px'>{line}</li>" for line in detail.get("timeline", [])])
                pm_html = f"<div style='background:#fffbeb;border:1px solid #fcd34d;padding:16px;border-radius:8px;margin-top:12px;color:#92400e'><div style='font-weight:bold;margin-bottom:8px'>📝 Postmortem Summary</div><div>Duration: {summ.get('num_actions',0)} Actions / Verdict: {summ.get('final_verdict','N/A')}</div><ul style='margin-top:12px;padding-left:20px;line-height:1.5'>{lines}</ul></div>"

            html += f"""<div class="event {ev['kind']}"><div class="meta"><span>{fmt_ts(ev['ts'])}</span><span style="font-weight:bold;color:#444">{ev['kind'].upper()}</span><span>Step {ev['step']}</span></div><h3 style="margin: 0 0 10px 0">{ev['title']}</h3>{diff_html}{effect_html}{pm_html}<pre>{json.dumps(detail, indent=2)}</pre></div>"""
        return Response(html + "</body></html>", mimetype="text/html")

    def start(self):
        print(f"🚀 Dashboard running at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False)
