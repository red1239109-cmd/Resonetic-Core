# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
import json
from typing import List, Any

def fmt_ts(ts):
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def render_timeline_html(events: List[Any], iid: str) -> str:
    html = f"""<html><head><meta charset="utf-8"><title>Timeline</title>
    <link rel="stylesheet" href="/static/style.css">
    </head><body>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;">
        <div><h1>📊 Timeline View</h1><div style="color:#666">Incident: {iid or 'All Recent'}</div></div>
        <a href="/incidents" style="text-decoration:none;color:#4f46e5;font-weight:bold">← Back</a>
    </div>"""
    
    for ev in reversed(events):
        detail = ev.get("detail", {})
        diff_html, effect_html, pm_html, veto_html = "", "", "", ""

        if "before" in detail and "after" in detail:
            rows = ""
            before, after = detail["before"], detail["after"]
            for k in sorted(set(before.keys()) | set(after.keys())):
                if before.get(k) != after.get(k):
                    rows += f"<tr><td>{k}</td><td><span class='val-old'>{before.get(k)}</span> <span class='val-new'>{after.get(k)}</span></td></tr>"
            if rows:
                diff_html = f"<table class='diff'><tr><th>Parameter</th><th>Change (Before &rarr; After)</th></tr>{rows}</table>"
            detail = {k: v for k, v in detail.items() if k not in ["before", "after"]}

        if ev["kind"] == "action_vetoed":
            veto_html = f"<div style='margin-top:10px;padding:10px;background:#000;color:white;border-radius:6px;font-family:monospace'>🚫 RULING: {detail.get('ruling')}<br>REASON: {detail.get('reason')}</div>"

        if ev["kind"] == "action_effect":
            verdict = detail.get("verdict", "UNKNOWN")
            bg, col = ("#d1fae5", "#065f46") if verdict == "effective" else ("#fee2e2", "#991b1b")
            effect_html = f"<div style='margin-top:12px;padding:12px;background:{bg};border-radius:8px;color:{col};font-weight:bold'>⚖️ Verdict: {verdict.upper()} (Score: {detail.get('score',0):.2f})</div>"

        if ev["kind"] == "postmortem":
            summ = detail.get("summary", {})
            lines = "".join([f"<li style='margin-bottom:6px'>{line}</li>" for line in detail.get("timeline", [])])
            pm_html = f"<div style='background:#fffbeb;border:1px solid #fcd34d;padding:16px;border-radius:8px;margin-top:12px;color:#92400e'><div style='font-weight:bold;margin-bottom:8px'>📝 Postmortem Summary</div><div>Actions: {summ.get('num_actions',0)} / Verdict: {summ.get('final_verdict','N/A')}</div><ul style='margin-top:12px;padding-left:20px;line-height:1.5'>{lines}</ul></div>"

        html += f"""<div class="event {ev['kind']}">
            <div class="meta">
                <span>{fmt_ts(ev['ts'])}</span>
                <span style="font-weight:bold;color:#444">{ev['kind'].upper()}</span>
                <span>Step {ev['step']}</span>
            </div>
            <h3 style="margin: 0 0 10px 0">{ev['title']}</h3>
            {veto_html}{diff_html}{effect_html}{pm_html}
            <pre>{json.dumps(detail, indent=2)}</pre>
        </div>"""
    return html + "</body></html>"
