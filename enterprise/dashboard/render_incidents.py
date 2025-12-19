# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
from typing import Dict, Any

def render_incidents_html(data: Dict[str, Any]) -> str:
    items = data.get("items", [])
    summ = data.get("summary", {})
    html = f"""<html><head><meta charset="utf-8"><title>Incidents</title>
    <link rel="stylesheet" href="/static/style.css">
    </head><body>
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
                <div style="font-size:14px;color:#666">ID: {item['incident_id']} | Stability: {int(item['stability_score']*100)}% ({item['stable_steps']} steps)</div>
            </div>
            <a href="{item['drilldown']}" class="btn">View Timeline →</a>
        </div>"""
    return html + "</body></html>"
