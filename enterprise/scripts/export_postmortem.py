# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.timeline import IncidentTimeline

def export(incident_id, output_file="postmortem.md"):
    timeline = IncidentTimeline()
    events = timeline.list_all(incident_id)
    
    # 마지막 postmortem 이벤트 찾기
    pm_event = next((e for e in reversed(events) if e['kind'] == 'postmortem'), None)
    
    if not pm_event:
        print(f"❌ No postmortem generated for {incident_id}")
        return

    data = pm_event['detail']
    summary = data.get('summary', {})
    lines = data.get('timeline', [])

    md = f"""# 📝 Postmortem Report: {incident_id}
**Generated At:** {pm_event['ts']}

## 📊 Summary
- **Opened At:** {summary.get('opened_at')}
- **Resolved At:** {summary.get('resolved_at')}
- **Total Actions:** {summary.get('num_actions')}
- **Final Verdict:** {summary.get('final_verdict')}

## ⏳ Timeline Narrative
"""
    for line in lines:
        md += f"- {line}\n"

    with open(output_file, "w") as f:
        f.write(md)
    
    print(f"✅ Exported to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_postmortem.py <incident_id>")
    else:
        export(sys.argv[1])
