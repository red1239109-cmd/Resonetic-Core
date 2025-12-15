# config/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path
import os

# Optional YAML
try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    yaml = None
    YAML_AVAILABLE = False


# -----------------------------
# helpers
# -----------------------------
def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (update or {}).items():
        if isinstance(base.get(k), dict) and isinstance(v, dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if not YAML_AVAILABLE:
        raise RuntimeError(f"PyYAML is not available, but YAML file exists: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {path}")
    return data


def _as_path(p: Optional[str]) -> Optional[Path]:
    if p is None:
        return None
    return Path(p).expanduser().resolve()


# -----------------------------
# dataclasses (single source of truth)
# -----------------------------
@dataclass
class Thresholds:
    panic: float = 0.50
    alert: float = 0.20
    warning: float = 0.10
    cruise: float = 0.00  # optional; not used as a threshold typically

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Thresholds":
        d = d or {}
        return Thresholds(
            panic=float(d.get("panic", 0.50)),
            alert=float(d.get("alert", 0.20)),
            warning=float(d.get("warning", 0.10)),
            cruise=float(d.get("cruise", 0.00)),
        )


@dataclass
class KernelConfig:
    eps: float = 1e-2
    structure_period: float = 3.0
    w_R: float = 1.0
    w_F: float = 0.4
    w_S: float = 0.3
    w_T: float = 0.3

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "KernelConfig":
        d = d or {}
        w = d.get("w", {}) or {}
        return KernelConfig(
            eps=float(d.get("eps", 1e-2)),
            structure_period=float(d.get("structure_period", 3.0)),
            w_R=float(w.get("R", 1.0)),
            w_F=float(w.get("F", 0.4)),
            w_S=float(w.get("S", 0.3)),
            w_T=float(w.get("T", 0.3)),
        )

    def weights(self) -> Dict[str, float]:
        return {"R": self.w_R, "F": self.w_F, "S": self.w_S, "T": self.w_T}


@dataclass
class PIDConfig:
    # “자동 튜닝”이든 “고정”이든, 결국 이 세 수치로 수렴함
    kp: float = 0.20
    ki: float = 0.00
    kd: float = 0.05

    # 안전장치
    out_min: float = -1.0
    out_max: float = 1.0
    integral_min: float = -0.5
    integral_max: float = 0.5

    # 파생/필터링
    derivative_smoothing: float = 0.90  # 0..1 (높을수록 더 부드러움)
    dt: float = 1.0                     # step-based OS면 1.0이 기본

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PIDConfig":
        d = d or {}
        limits = d.get("limits", {}) or {}
        return PIDConfig(
            kp=float(d.get("kp", 0.20)),
            ki=float(d.get("ki", 0.00)),
            kd=float(d.get("kd", 0.05)),
            out_min=float(limits.get("out_min", -1.0)),
            out_max=float(limits.get("out_max", 1.0)),
            integral_min=float(limits.get("integral_min", -0.5)),
            integral_max=float(limits.get("integral_max", 0.5)),
            derivative_smoothing=float(d.get("derivative_smoothing", 0.90)),
            dt=float(d.get("dt", 1.0)),
        )


@dataclass
class ProphetConfig:
    seed: int = 42
    steps: int = 2000
    log_interval: int = 100
    batch_size: int = 1

    # risk mapping, smoothing
    risk_smoothing: float = 0.90
    risk_sigmoid_k: float = 3.0

    # saturation proxy (예: ΔEMA)
    delta_ema_beta: float = 0.95

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProphetConfig":
        d = d or {}
        return ProphetConfig(
            seed=int(d.get("seed", 42)),
            steps=int(d.get("steps", 2000)),
            log_interval=int(d.get("log_interval", 100)),
            batch_size=int(d.get("batch_size", 1)),
            risk_smoothing=float(d.get("risk_smoothing", 0.90)),
            risk_sigmoid_k=float(d.get("risk_sigmoid_k", 3.0)),
            delta_ema_beta=float(d.get("delta_ema_beta", 0.95)),
        )


@dataclass
class OSConfig:
    thresholds: Thresholds = field(default_factory=Thresholds)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    prophet: ProphetConfig = field(default_factory=ProphetConfig)

    # 원본(dict) 백업(디버깅/로그용)
    raw: Dict[str, Any] = field(default_factory=dict)

    def post_init_validate(self) -> None:
        # 범위 클램프 (오류를 "빨리" 터뜨리기)
        if not (0.0 <= self.prophet.risk_smoothing <= 0.999):
            raise ValueError("prophet.risk_smoothing must be in [0, 0.999]")
        if not (0.0 <= self.prophet.delta_ema_beta <= 0.999):
            raise ValueError("prophet.delta_ema_beta must be in [0, 0.999]")
        if self.pid.out_min >= self.pid.out_max:
            raise ValueError("pid.limits: out_min must be < out_max")
        if self.pid.integral_min >= self.pid.integral_max:
            raise ValueError("pid.limits: integral_min must be < integral_max")


# -----------------------------
# loader: prophet.yaml + thresholds.yaml + env overrides
# -----------------------------
DEFAULT_DIR = Path(__file__).resolve().parent


def load_config(
    base_dir: Optional[str] = None,
    prophet_yaml: str = "prophet.yaml",
    thresholds_yaml: str = "thresholds.yaml",
) -> OSConfig:
    cfg_dir = _as_path(base_dir) or DEFAULT_DIR

    merged: Dict[str, Any] = {}
    # 1) prophet.yaml
    merged = _deep_update(merged, _read_yaml(cfg_dir / prophet_yaml))
    # 2) thresholds.yaml (분리 운영 가능)
    merged = _deep_update(merged, {"thresholds": _read_yaml(cfg_dir / thresholds_yaml)})

    # ---- env overrides (필요한 것만 최소로) ----
    # 예: export RESONETICS_SEED=123
    if os.getenv("RESONETICS_SEED"):
        merged.setdefault("prophet", {})
        merged["prophet"]["seed"] = int(os.getenv("RESONETICS_SEED", "42"))

    if os.getenv("RESONETICS_PID_KP"):
        merged.setdefault("pid", {})
        merged["pid"]["kp"] = float(os.getenv("RESONETICS_PID_KP", "0.2"))

    # ---- build dataclasses ----
    thresholds = Thresholds.from_dict(merged.get("thresholds", {}) or {})
    kernel = KernelConfig.from_dict(merged.get("kernel", {}) or {})
    pid = PIDConfig.from_dict(merged.get("pid", {}) or {})
    prophet = ProphetConfig.from_dict(merged.get("prophet", {}) or {})

    cfg = OSConfig(
        thresholds=thresholds,
        kernel=kernel,
        pid=pid,
        prophet=prophet,
        raw=merged,
    )
    cfg.post_init_validate()
    return cfg

