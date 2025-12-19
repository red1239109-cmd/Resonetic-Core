# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.action import ActionApplicator, ActionPlan, SupremeCourt
from src.incident import IncidentRegistry
from src.timeline import IncidentTimeline

class TestActionDiff(unittest.TestCase):
    def setUp(self):
        self.registry = IncidentRegistry()
        self.timeline = IncidentTimeline(jsonl_path="runs/test_timeline.jsonl")
        self.applicator = ActionApplicator(self.registry, self.timeline)
        
        # Mock State
        self.state = {"learning_rate": 0.005, "reality_weight": 1.0, "dropout": 0.1}

    def test_constitutional_action(self):
        # Constitutional change (Valid)
        plan = ActionPlan("act_1", "inc_1", "Safe Change", {"dropout": 0.5})
        success = self.applicator.apply(
            plan, 
            get_state=lambda: self.state, 
            set_state=lambda s: self.state.update(s), 
            step=1
        )
        self.assertTrue(success)
        self.assertEqual(self.state["dropout"], 0.5)

    def test_unconstitutional_action(self):
        # Unconstitutional change (LR > 0.01)
        plan = ActionPlan("act_2", "inc_1", "Dangerous Change", {"learning_rate": 0.05})
        success = self.applicator.apply(
            plan, 
            get_state=lambda: self.state, 
            set_state=lambda s: self.state.update(s), 
            step=2
        )
        self.assertFalse(success)
        self.assertEqual(self.state["learning_rate"], 0.005) # Should remain unchanged

    def tearDown(self):
        if os.path.exists("runs/test_timeline.jsonl"):
            os.remove("runs/test_timeline.jsonl")

if __name__ == '__main__':
    unittest.main()
