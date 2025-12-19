# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
import unittest
import sys
import os

# src 모듈을 찾기 위한 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.incident import IncidentRegistry

class TestIncidentRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = IncidentRegistry()

    def test_create_incident(self):
        inc = self.registry.create_or_update(title="Test Incident", severity="high")
        self.assertIsNotNone(inc.incident_id)
        self.assertEqual(inc.status, "OPEN")
        self.assertEqual(inc.severity, "high")

    def test_status_transition(self):
        inc = self.registry.create_or_update(title="Transition Test")
        self.registry.create_or_update(incident_id=inc.incident_id, status="MITIGATING")
        
        updated = self.registry.by_id[inc.incident_id]
        self.assertEqual(updated.status, "MITIGATING")

if __name__ == '__main__':
    unittest.main()
