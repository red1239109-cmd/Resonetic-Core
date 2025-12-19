# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def replay(jsonl_path="runs/timeline.jsonl"):
    if not os.path.exists(jsonl_path):
        print("❌ No timeline found.")
        return

    print(f"🎬 Replaying timeline from {jsonl_path}...\n")
    with open(jsonl_path, 'r') as f:
        for line in f:
            ev = json.loads(line)
            ts = ev.get('ts')
            kind = ev.get('kind').upper()
            title = ev.get('title')
            step = ev.get('step')
            
            # 시각화 출력
            icon = "🔹"
            if kind == "ANOMALY": icon = "🔥"
            if kind == "ACTION_APPLY": icon = "🛡️"
            if kind == "ACTION_VETOED": icon = "🚫"
            if kind == "ACTION_EFFECT": icon = "⚖️"
            if kind == "RESOLVE": icon = "✅"

            print(f"{icon} [Step {step}] {kind}: {title}")
            
            # Diff가 있으면 출력
            if "detail" in ev and "before" in ev['detail']:
                before = ev['detail']['before']
                after = ev['detail']['after']
                diff = {k: f"{before[k]} -> {after[k]}" for k in after if before.get(k) != after.get(k)}
                if diff:
                    print(f"      📝 Diff: {diff}")
            
            # Verdict 출력
            if kind == "ACTION_EFFECT":
                print(f"      Outcome: {ev['detail'].get('verdict')}")

if __name__ == "__main__":
    replay()
