from config.config import load_config
cfg = load_config()
W = cfg.kernel.weights()
EPS = cfg.kernel.eps
PERIOD = cfg.kernel.structure_period


# src/resonetics_os/kernel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Dict, Any

# -----------------------------
# 1) StateMachine "인터페이스" (Protocol)
# -----------------------------
class StateMachineLike(Protocol):
    state: str
    def step(self, signals: Dict[str, Any]) -> str: ...


# -----------------------------
# 2) 기본 StateMachine (있다면 기존꺼 유지해도 됨)
# -----------------------------
@dataclass
class StateMachine:
    state: str = "CRUISE"

    def step(self, signals: Dict[str, Any]) -> str:
        # TODO: 너의 히스테리시스/threshold 로직으로 교체
        risk = float(signals.get("risk", 0.0))
        if risk > 0.5:
            self.state = "PANIC"
        elif risk > 0.2:
            self.state = "ALERT"
        elif risk > 0.1:
            self.state = "WARNING"
        else:
            self.state = "CRUISE"
        return self.state


# -----------------------------
# 3) KernelGovernor: StateMachine을 "주입" 받는다
# -----------------------------
class KernelGovernor:
    def __init__(self, sm: Optional[StateMachineLike] = None):
        self.sm: StateMachineLike = sm or StateMachine()

    def tick(self, signals: Dict[str, Any]) -> str:
        """signals 예: {'risk': 0.12, 'kernel_loss': 0.03, ...}"""
        return self.sm.step(signals)
