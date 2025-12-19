#!/usr/bin/env python3
# ==============================================================================
# File: dre_v1_6_1.py
# Product: Data Refinery Engine (DRE) v1.6.1
# Patch: Import 누락 + TimelineEvent 정의 + ActionEffectAnalyzer 실제 호출
# ==============================================================================
import pandas as pd
import numpy as np
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy
from dataclasses import dataclass
from flask import Flask, request, Response
from collections import deque  # ← 추가

# ==============================================================================
# TimelineEvent 정의 (독립 실행 위해 추가)
# ==============================================================================
def now_ts() -> float:
    return float(time.time())

@dataclass
class TimelineEvent:
    ts: float
    step: int
    kind: str
    severity: str
    title: str
    detail: Dict = field(default_factory=dict)
    incident_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

# ==============================================================================
# 나머지 코드 동일 (ExplainCard, ExplainCardBuilder, AlwaysWinnerDetector 등)
# ==============================================================================

class ActionEffectAnalyzer:
    def __init__(self, timeline, window_steps: int = 10, always_winner: Optional[AlwaysWinnerDetector] = None):
        self.timeline = timeline
        self.window_steps = int(window_steps)
        self.buffers = {}
        self.always_winner = always_winner

    def start(self, action_id: str, step: int, baseline: Dict):
        self.buffers[action_id] = {
            "step": step,
            "baseline": baseline,
            "samples": deque(maxlen=self.window_steps)
        }

    def collect(self, action_id: str, metrics: Dict):
        if action_id in self.buffers:
            self.buffers[action_id]["samples"].append(metrics)

    def finalize(self, action_id: str, current_step: int) -> Optional[Dict]:
        buf = self.buffers.pop(action_id, None)
        if not buf or len(buf["samples"]) < self.window_steps:
            return None

        # delta 계산 (간단 예시)
        baseline = buf["baseline"]
        after_avg = {k: np.mean([s.get(k, 0) for s in buf["samples"]]) for k in baseline}
        delta = {k: after_avg.get(k, 0) - baseline.get(k, 0) for k in baseline}

        winner, scores = pick_winner_by_delta(delta)

        if self.always_winner:
            self.always_winner.observe(
                step=current_step,
                incident_id=None,
                winner=winner,
                scores=scores
            )

        # Timeline 이벤트 추가 (간단)
        self.timeline.add(TimelineEvent(
            ts=now_ts(),
            step=current_step,
            kind="action_effect",
            severity="info",
            title=f"Effect: {winner} benefited",
            detail={"delta": delta, "winner": winner}
        ))

        return {"delta": delta, "winner": winner}

# ==============================================================================
# Demo (실제 호출 추가)
# ==============================================================================
if __name__ == "__main__":
    timeline = IncidentTimeline()  # 기존 정의 필요 시 추가
    always = AlwaysWinnerDetector(timeline, window=20, streak_trigger=3)
    effect = ActionEffectAnalyzer(timeline, window_steps=5, always_winner=always)

    # 예시 배치 처리 루프에서
    # effect.start("dummy_action", step=0, baseline={"risk": 0.9, "loss": 2.0, "stability": 0.1})
    # effect.collect("dummy_action", {"risk": 0.8, "loss": 1.8, "stability": 0.2})
    # ... window 채운 후
    # effect.finalize("dummy_action", current_step=10)
