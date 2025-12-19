# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.incident import IncidentRegistry
from src.stability import StabilityTracker
from src.timeline import IncidentTimeline

class TestStability(unittest.TestCase):
    def setUp(self):
        self.registry = IncidentRegistry()
        self.timeline = IncidentTimeline(jsonl_path="runs/test_timeline.jsonl")
        self.tracker = StabilityTracker(self.registry, self.timeline)

    def test_stability_score_increase(self):
        inc = self.registry.create_or_update(title="Stability Test", severity="warn")
        
        # 안정적인 상태(0.9)를 5번 주입
        for i in range(5):
            self.tracker.observe(inc.incident_id, step=i, stability=0.9, signal={})
        
        updated = self.registry.by_id[inc.incident_id]
        self.assertEqual(updated.stable_steps, 5)
        self.assertEqual(updated.stability_score, 0.5) # 5/10 steps

    def tearDown(self):
        # 테스트용 파일 삭제
        if os.path.exists("runs/test_timeline.jsonl"):
            os.remove("runs/test_timeline.jsonl")

if __name__ == '__main__':
    unittest.main()
