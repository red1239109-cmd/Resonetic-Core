#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: dre_v1_13_4_single_trunk.py
# Product: Data Refinery Engine (DRE) - Single Trunk
# Version: 1.13.4
#
# Highlights:
# - Backend: polars (preferred) + pandas fallback
# - Parallel scoring: thread/process (safe), auto-tuning workers (caps)
# - ProcessPool SAFE: never send full df to workers (meta sample only)
# - Dynamic early-exit for high-cardinality columns (UUID/ID safe)
# - PID anti-windup sign fix
# - Timeline/Queue/DAG stores with safe paths + flush safety + rotation
# - DAGRunner with DOT output (no graphviz required)
# - CLI: benchmark (rows sweep), col-sweep
# ==============================================================================
from __future__ import annotations
import os
import sys
import json
import time
import math
import uuid
import argparse
import hashlib
import traceback
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque
import numpy as np
# Optional deps
try:
    import polars as pl # type: ignore
    _HAS_POLARS = True
except Exception:
    pl = None
    _HAS_POLARS = False
try:
    import pandas as pd # type: ignore
    _HAS_PANDAS = True
except Exception:
    pd = None
    _HAS_PANDAS = False
# ==============================================================================
# Utils
# ==============================================================================
def now_ts() -> float:
    return float(time.time())
def fmt_ts(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return str(ts)
def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"error": "json_dump_failed", "type": str(type(obj))}, ensure_ascii=False, indent=2)
def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
def safe_path(user_path: str, base: Path) -> Path:
    """
    Path traversal 방지: base 하위만 허용.
    user_path가 상대경로면 base 기준으로 결합.
    """
    base = base.resolve()
    up = Path(user_path)
    if not up.is_absolute():
        p = (base / up).resolve()
    else:
        p = up.resolve()
    # Python 3.9+: is_relative_to
    try:
        if not p.is_relative_to(base):
            raise ValueError(f"Path traversal attempt blocked: {p} not under {base}")
    except AttributeError:
        # fallback
        if str(base) not in str(p):
            raise ValueError(f"Path traversal attempt blocked: {p} not under {base}")
    return p
def rotator_write_jsonl(path: Path, lines: List[str], max_bytes: int = 20_000_000, backups: int = 3) -> None:
    """
    초간단 로그 로테이션: 파일이 max_bytes 넘으면 .1, .2 ...로 밀어냄
    """
    _ensure_dir(path)
    try:
        if path.exists() and path.stat().st_size > max_bytes:
            # rotate
            for i in range(backups, 0, -1):
                src = path.with_suffix(path.suffix + f".{i}")
                dst = path.with_suffix(path.suffix + f".{i+1}")
                if dst.exists():
                    dst.unlink(missing_ok=True) # type: ignore
                if src.exists():
                    src.rename(dst)
            path.rename(path.with_suffix(path.suffix + ".1"))
    except Exception:
        pass
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
def _sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]
# ==============================================================================
# Data Hash (safe upper bound)
# ==============================================================================
def hash_df_any(df_any: Any, cap_bytes: int = 1_000_000) -> str:
    """
    '안전한' 데이터셋 해시:
    - 너무 큰 payload는 cap_bytes로 컷
    - pandas/polars 각각에서 안정적으로 동작
    """
    try:
        if _HAS_POLARS and isinstance(df_any, pl.DataFrame):
            cols = "|".join(map(str, df_any.columns)).encode("utf-8")
            # head + schema + shape 기반의 안전한 요약
            head = df_any.head(20).to_dict(as_series=False)
            payload = (str(df_any.shape) + str(head)).encode("utf-8")
            blob = cols + b"::" + payload
            blob = blob[:cap_bytes]
            return _sha16(blob)
        if _HAS_PANDAS and isinstance(df_any, pd.DataFrame):
            cols = "|".join(map(str, df_any.columns)).encode("utf-8")
            head = df_any.head(20).to_dict()
            payload = (str(df_any.shape) + str(head)).encode("utf-8")
            blob = cols + b"::" + payload
            blob = blob[:cap_bytes]
            return _sha16(blob)
        # dict/other
        payload = str(type(df_any)) + ":" + str(df_any)[:20000]
        blob = payload.encode("utf-8")[:cap_bytes]
        return _sha16(blob)
    except Exception:
        return _sha16(str(time.time()).encode("utf-8"))
# ==============================================================================
# Explain Cards
# ==============================================================================
@dataclass(frozen=True)
class ExplainCard:
    decision: str
    headline: str
    evidence: List[str]
    scores: Dict[str, Any]
    threshold: float
    reason: str
    column: str
    batch_id: str
class ExplainCardBuilder:
    def build(self, batch_id: str, column: str, decision: str,
              scores: Dict[str, Any], threshold: float, reason: str = "score_based") -> ExplainCard:
        gold = float(scores.get("gold_score", 0.0))
        rel = float(scores.get("relevance", 0.0))
        qual = float(scores.get("quality", 0.0))
        if decision == "KEEP":
            headline = f"'{column}' 유지: gold_score({gold:.3f}) ≥ 임계값({threshold:.3f})"
        else:
            headline = f"'{column}' 제거: gold_score({gold:.3f}) < 임계값({threshold:.3f})"
        evidence = [
            f"Gold Score: {gold:.3f} vs Threshold: {threshold:.3f}",
            f"Relevance: {rel:.3f} | Quality: {qual:.3f}",
            f"Reason: {reason.replace('_',' ').title()}",
        ]
        return ExplainCard(
            decision=decision,
            headline=headline,
            evidence=evidence,
            scores=dict(scores),
            threshold=float(threshold),
            reason=reason,
            column=column,
            batch_id=batch_id,
        )
# ==============================================================================
# Timeline / Queue / DAG Stores
# ==============================================================================
@dataclass
class TimelineEvent:
    ts: float
    step: int
    kind: str
    severity: str
    title: str
    detail: Dict[str, Any] = field(default_factory=dict)
    incident_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
class TimelineStore:
    def __init__(self, path: Path, maxlen: int = 5000, rotate_max_bytes: int = 20_000_000):
        self.path = path
        self.buf = deque(maxlen=int(maxlen))
        self.rotate_max_bytes = int(rotate_max_bytes)
        _ensure_dir(self.path)
        self._pending: List[str] = []
    def add(self, ev: TimelineEvent) -> str:
        payload = asdict(ev)
        self.buf.append(payload)
        self._pending.append(json.dumps(payload, ensure_ascii=False))
        # 적당히 flush
        if len(self._pending) >= 50:
            self.flush()
        return "evt_" + str(uuid.uuid4())[:8]
    def flush(self) -> None:
        if not self._pending:
            return
        try:
            rotator_write_jsonl(self.path, self._pending, max_bytes=self.rotate_max_bytes, backups=3)
            self._pending.clear()
        except Exception as e:
            # ✅ 실패해도 pending은 비워서 운영 누수/폭증 방지
            self._pending.clear()
            warnings.warn(f"[TimelineStore] flush failed: {e}")
    def list_recent(self, limit: int = 200) -> List[Dict[str, Any]]:
        return list(self.buf)[-int(limit):]
@dataclass
class QueueItem:
    ts: float
    batch_id: str
    status: str # RUNNING | OK | FAIL
    input_rows: int = 0
    input_cols: int = 0
    kept_cols: int = 0
    dropped_cols: int = 0
    threshold: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
class QueueStore:
    def __init__(self, path: Path, rotate_max_bytes: int = 20_000_000):
        self.path = path
        self.rotate_max_bytes = int(rotate_max_bytes)
        _ensure_dir(self.path)
        self._pending: List[str] = []
    def append(self, item: QueueItem) -> None:
        self._pending.append(json.dumps(asdict(item), ensure_ascii=False))
        if len(self._pending) >= 50:
            self.flush()
    def flush(self) -> None:
        if not self._pending:
            return
        try:
            rotator_write_jsonl(self.path, self._pending, max_bytes=self.rotate_max_bytes, backups=3)
            self._pending.clear()
        except Exception:
            self._pending.clear()
    def latest_by_batch(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        items: List[Dict[str, Any]] = []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        except Exception:
            pass
        latest: Dict[str, Dict[str, Any]] = {}
        for it in items:
            bid = it.get("batch_id")
            if not bid:
                continue
            prev = latest.get(bid)
            if prev is None or float(it.get("ts", 0)) >= float(prev.get("ts", 0)):
                latest[bid] = it
        ordered = sorted(latest.values(), key=lambda x: float(x.get("ts", 0)), reverse=True)
        return ordered[:int(limit)]
@dataclass
class DAGEdge:
    src: str
    dst: str
    kind: str
    ts: float
    meta: Dict[str, Any] = field(default_factory=dict)
class DAGStore:
    def __init__(self, path: Path, maxlen: int = 5000, rotate_max_bytes: int = 20_000_000):
        self.path = path
        self.buf = deque(maxlen=int(maxlen))
        self.rotate_max_bytes = int(rotate_max_bytes)
        _ensure_dir(self.path)
        self._pending: List[str] = []
    def add_edge(self, e: DAGEdge) -> None:
        payload = asdict(e)
        self.buf.append(payload)
        self._pending.append(json.dumps(payload, ensure_ascii=False))
        if len(self._pending) >= 50:
            self.flush()
    def flush(self) -> None:
        if not self._pending:
            return
        try:
            rotator_write_jsonl(self.path, self._pending, max_bytes=self.rotate_max_bytes, backups=3)
            self._pending.clear()
        except Exception:
            self._pending.clear()
    def list_recent(self, limit: int = 500) -> List[Dict[str, Any]]:
        return list(self.buf)[-int(limit):]
    def scan_all(self, limit: int = 50000) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except Exception:
            pass
        return out
# ==============================================================================
# PID Controller (anti-windup sign fix)
# ==============================================================================
class PIDController:
    def __init__(self, kp: float = 0.25, ki: float = 0.05, windup_trigger: int = 8):
        self.kp = float(kp)
        self.ki = float(ki)
        self.integral = 0.0
        self.windup_trigger = int(windup_trigger)
        self._same_sign = 0
    def step(self, err: float) -> float:
        err = float(err)
        self.integral = float(np.clip(self.integral + err, -1.0, 1.0))
        # ✅ sign 기반으로 same-sign streak 판정
        if np.sign(err) == np.sign(self.integral) and self.integral != 0.0:
            self._same_sign += 1
        else:
            self._same_sign = 0
        ki_eff = self.ki
        if self._same_sign >= self.windup_trigger:
            ki_eff *= 0.2 # anti-windup
        return (self.kp * err) + (ki_eff * self.integral)
# ==============================================================================
# Fairness / Effect Analyzer (간단 유지)
# ==============================================================================
def pick_winner_by_delta(delta: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    benefit = {
        "risk": -float(delta.get("risk", 0.0)),
        "loss": -float(delta.get("loss", 0.0)),
        "stability": float(delta.get("stability", 0.0)),
    }
    winner = max(benefit.items(), key=lambda x: x[1])[0]
    return winner, benefit
class AlwaysWinnerDetector:
    def __init__(self, timeline: TimelineStore, window: int = 30, streak_trigger: int = 3):
        self.timeline = timeline
        self.window = int(window)
        self.streak_trigger = int(streak_trigger)
        self.history = deque(maxlen=self.window)
        self.current_winner: Optional[str] = None
        self.streak = 0
    def observe(self, step: int, batch_id: str, winner: str, scores: Dict[str, float]):
        self.history.append(winner)
        if self.current_winner == winner:
            self.streak += 1
        else:
            self.current_winner = winner
            self.streak = 1
        if len(self.history) < max(5, self.window // 3):
            return
        counts: Dict[str, int] = {}
        for w in self.history:
            counts[w] = counts.get(w, 0) + 1
        top, cnt = max(counts.items(), key=lambda x: x[1])
        concentration = cnt / max(1, len(self.history))
        if concentration >= 0.70 and self.streak >= self.streak_trigger:
            self.timeline.add(TimelineEvent(
                ts=now_ts(),
                step=int(step),
                kind="systemic_unfairness",
                severity="warn",
                title=f"Systemic Bias Detected: '{top}' dominates",
                incident_id=batch_id,
                tags=["fairness", "always_winner"],
                detail={
                    "dominant": top,
                    "concentration": round(float(concentration), 3),
                    "streak": int(self.streak),
                    "window": int(len(self.history)),
                    "scores": scores
                }
            ))
class ActionEffectAnalyzer:
    def __init__(self, timeline: TimelineStore, window_steps: int = 1, always_winner: Optional[AlwaysWinnerDetector] = None):
        self.timeline = timeline
        self.window_steps = int(window_steps)
        self.buffers: Dict[str, Dict[str, Any]] = {}
        self.always_winner = always_winner
    def start(self, batch_id: str, step: int, baseline_metrics: Dict[str, float]) -> None:
        self.buffers[batch_id] = {
            "batch_id": batch_id,
            "start_step": int(step),
            "baseline": dict(baseline_metrics or {}),
            "samples": deque(maxlen=max(1, self.window_steps)),
        }
    def collect(self, batch_id: str, metrics: Dict[str, float]) -> None:
        if batch_id not in self.buffers:
            return
        buf = self.buffers[batch_id]
        buf["samples"].append(dict(metrics or {}))
        if len(buf["samples"]) >= buf["samples"].maxlen:
            self.finalize(batch_id, step=buf["start_step"] + len(buf["samples"]))
    def finalize(self, batch_id: str, step: int) -> Optional[Dict[str, Any]]:
        buf = self.buffers.pop(batch_id, None)
        if not buf:
            return None
        samples = list(buf["samples"])
        if not samples:
            return None
        baseline = buf["baseline"]
        def avg(key: str) -> Optional[float]:
            vals = [s.get(key) for s in samples if isinstance(s.get(key), (int, float, np.floating))]
            return float(np.mean(vals)) if vals else None
        after = {k: avg(k) for k in ["risk", "loss", "stability"]}
        delta: Dict[str, float] = {}
        for k, v_after in after.items():
            v_base = baseline.get(k)
            if isinstance(v_after, (int, float)) and isinstance(v_base, (int, float)):
                delta[k] = float(v_after - v_base)
        winner, scores = pick_winner_by_delta(delta)
        if self.always_winner:
            self.always_winner.observe(step=int(step), batch_id=batch_id, winner=winner, scores=scores)
        score = 0.0
        score += (-delta.get("risk", 0.0)) * 1.0
        score += (-delta.get("loss", 0.0)) * 0.5
        score += (delta.get("stability", 0.0)) * 1.0
        verdict = "ineffective"
        if score > 0.10:
            verdict = "effective"
        elif score > 0.02:
            verdict = "partial"
        ev_detail = {
            "baseline": baseline,
            "after_avg": after,
            "delta": delta,
            "score": float(score),
            "verdict": verdict,
            "winner": winner,
            "winner_scores": scores,
        }
        self.timeline.add(TimelineEvent(
            ts=now_ts(),
            step=int(step),
            kind="action_effect",
            severity="info" if verdict == "effective" else "warn",
            title=f"Batch Effect: {verdict.upper()} (score={score:.3f})",
            incident_id=batch_id,
            tags=["effect", verdict, winner],
            detail=ev_detail
        ))
        return ev_detail
# ==============================================================================
# DAG Runner + DOT
# ==============================================================================
@dataclass
class DAGNode:
    key: str
    name: str
    fn: Callable[[Any, Dict[str, Any]], Any]
    deps: List[str] = field(default_factory=list)
    optional: bool = False
    default: Any = None
class DAGRunner:
    def __init__(self, timeline: TimelineStore, queue: Optional[QueueStore] = None):
        self.timeline = timeline
        self.queue = queue
        self.nodes: Dict[str, DAGNode] = {}
        self.edges: List[Tuple[str, str]] = [] # dep -> node
    def add(self, node: DAGNode) -> None:
        self.nodes[node.key] = node
        for d in node.deps:
            self.edges.append((d, node.key))
    def run(self, input_data: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        order = self._toposort()
        self.timeline.add(TimelineEvent(
            ts=now_ts(), step=0, kind="dag_start", severity="info",
            title="DAG Started", detail={"order": order}, tags=["dag"]
        ))
        for i, k in enumerate(order, start=1):
            node = self.nodes[k]
            deps_data = {d: out.get(d) for d in node.deps}
            try:
                self.timeline.add(TimelineEvent(
                    ts=now_ts(), step=i, kind="dag_node_start", severity="info",
                    title=f"Node Start: {k}", detail={"deps": node.deps}, tags=["dag", "node"]
                ))
                out[k] = node.fn(input_data, deps_data)
                self.timeline.add(TimelineEvent(
                    ts=now_ts(), step=i, kind="dag_node_ok", severity="info",
                    title=f"Node OK: {k}", detail={"type": str(type(out[k]))}, tags=["dag", "node"]
                ))
            except Exception as e:
                tb = traceback.format_exc(limit=3)
                if node.optional:
                    out[k] = node.default
                    self.timeline.add(TimelineEvent(
                        ts=now_ts(), step=i, kind="dag_node_fail_optional", severity="warn",
                        title=f"Node FAIL (optional): {k}", detail={"error": str(e), "trace": tb}, tags=["dag", "node"]
                    ))
                else:
                    self.timeline.add(TimelineEvent(
                        ts=now_ts(), step=i, kind="dag_node_fail", severity="high",
                        title=f"Node FAIL: {k}", detail={"error": str(e), "trace": tb}, tags=["dag", "node"]
                    ))
                    raise
        self.timeline.add(TimelineEvent(
            ts=now_ts(), step=len(order)+1, kind="dag_done", severity="info",
            title="DAG Done", detail={"keys": list(out.keys())}, tags=["dag"]
        ))
        return out
    def dot(self) -> str:
        # DOT output (graphviz 없이도 문자열 제공)
        lines = ["digraph DRE_DAG {", "rankdir=LR;", "node [shape=box, style=rounded];"]
        for k, node in self.nodes.items():
            label = f"{k}\\n{node.name}"
            if node.optional:
                lines.append(f"\"{k}\" [label=\"{label}\", style=\"rounded,dashed\"];")
            else:
                lines.append(f"\"{k}\" [label=\"{label}\"];")
        for a, b in self.edges:
            lines.append(f"\"{a}\" -> \"{b}\";")
        lines.append("}")
        return "\n".join(lines)
    def _toposort(self) -> List[str]:
        # Kahn
        indeg: Dict[str, int] = {k: 0 for k in self.nodes}
        adj: Dict[str, List[str]] = {k: [] for k in self.nodes}
        for a, b in self.edges:
            if a not in self.nodes or b not in self.nodes:
                continue
            adj[a].append(b)
            indeg[b] += 1
        q = deque([k for k, d in indeg.items() if d == 0])
        order: List[str] = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(order) != len(self.nodes):
            raise RuntimeError("DAG has cycle or missing deps")
        return order
# ==============================================================================
# DataRefineryEngine
# ==============================================================================
class DataRefineryEngine:
    def __init__(
        self,
        runs_dir: str = "runs",
        timeline_path: str = "timeline.jsonl",
        queue_path: str = "queue.jsonl",
        dag_path: str = "dag.jsonl",
        target_kpi: Optional[str] = None,
        base_threshold: float = 0.45,
        target_retention: float = 0.30,
        min_threshold: float = 0.10,
        max_threshold: float = 0.90,
        seed: int = 42,
        relevance_sample_n: int = 10_000,
        parallel: bool = True,
        parallel_mode: str = "auto", # auto|sequential|thread|process
        max_workers_cap: int = 32,
    ):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.runs_base = Path(runs_dir).resolve()
        self.runs_base.mkdir(parents=True, exist_ok=True)
        self.timeline = TimelineStore(safe_path(timeline_path, self.runs_base))
        self.queue = QueueStore(safe_path(queue_path, self.runs_base))
        self.dag = DAGStore(safe_path(dag_path, self.runs_base))
        self.target_kpi = target_kpi
        self.threshold = float(base_threshold)
        self.target_retention = float(target_retention)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.pid = PIDController()
        self.relevance_sample_n = int(relevance_sample_n)
        self.parallel = bool(parallel)
        self.parallel_mode = str(parallel_mode)
        self.max_workers_cap = int(max_workers_cap)
        self.max_workers = self.autotune_workers(999999)
        self.explain_builder = ExplainCardBuilder()
        self.explain_cards: Dict[str, List[ExplainCard]] = {}
        self.always = AlwaysWinnerDetector(self.timeline, window=20, streak_trigger=3)
        self.effect = ActionEffectAnalyzer(self.timeline, window_steps=1, always_winner=self.always)
        self._step = 0
        # backend preference
        self.backend = "polars" if _HAS_POLARS else "pandas"
        if self.backend == "pandas" and not _HAS_PANDAS:
            raise RuntimeError("Neither polars nor pandas available. Install one of them.")
        # dataset producer index (E4-style linking)
        self._dataset_producer: Dict[str, str] = {}
        self._rehydrate_dag_index()
    # -----------------------------
    # Backend helpers
    # -----------------------------
    def _is_polars(self) -> bool:
        return self.backend == "polars"
    def _len_df(self, df_any: Any) -> int:
        if self._is_polars() and isinstance(df_any, pl.DataFrame):
            return int(df_any.height)
        if (not self._is_polars()) and _HAS_PANDAS and isinstance(df_any, pd.DataFrame):
            return int(len(df_any))
        # dict fallback
        return int(len(df_any)) if hasattr(df_any, "__len__") else 0
    def _cols_df(self, df_any: Any) -> List[str]:
        if self._is_polars() and isinstance(df_any, pl.DataFrame):
            return list(df_any.columns)
        if (not self._is_polars()) and _HAS_PANDAS and isinstance(df_any, pd.DataFrame):
            return list(df_any.columns)
        return []
    def _to_backend_df(self, x: Any):
        if self._is_polars():
            if isinstance(x, pl.DataFrame):
                return x
            if _HAS_PANDAS and isinstance(x, pd.DataFrame):
                return pl.from_pandas(x)
            if isinstance(x, dict):
                return pl.DataFrame(x)
            raise TypeError("Unsupported input type for polars backend")
        else:
            if _HAS_PANDAS and isinstance(x, pd.DataFrame):
                return x
            if _HAS_POLARS and isinstance(x, pl.DataFrame):
                return x.to_pandas()
            if isinstance(x, dict):
                return pd.DataFrame(x) # type: ignore
            raise TypeError("Unsupported input type for pandas backend")
    # -----------------------------
    # Safety / tuning
    # -----------------------------
    def autotune_workers(self, n_cols: int) -> int:
        cpu = os.cpu_count() or 4
        base = min(self.max_workers_cap, cpu + 4)
        if n_cols <= 20:
            return min(base, 6)
        if n_cols <= 60:
            return min(base, 10)
        return min(base, 12)
    def _resolve_parallel_mode(self, mode: str) -> str:
        mode = (mode or "auto").lower().strip()
        if not self.parallel:
            return "sequential"
        if mode in ("sequential", "thread", "process"):
            return mode
        # auto
        if self.backend == "pandas":
            # ✅ pandas는 병렬 이득 적고 리스크 큼: 기본 sequential
            return "sequential"
        return "thread" # polars는 내부적으로 GIL 영향이 상대적으로 적고, 컬럼 스코어링 병렬이 유효할 수 있음
    def _dynamic_early_exit_ratio(self, n: int) -> float:
        # n 커질수록 더 공격적으로 스킵
        return min(0.9, max(0.5, 5000 / max(1, n)))
    # -----------------------------
    # DAG index rehydrate
    # -----------------------------
    def _rehydrate_dag_index(self) -> None:
        edges = self.dag.scan_all(limit=50000)
        for e in edges:
            try:
                if e.get("kind") == "batch_to_output":
                    src = str(e.get("src", ""))
                    dst = str(e.get("dst", ""))
                    if src.startswith("batch_") and dst.startswith("dataset:"):
                        self._dataset_producer[dst] = src
            except Exception:
                continue
    def _new_batch_id(self) -> str:
        return "batch_" + str(uuid.uuid4())[:10]
    # -----------------------------
    # Scoring: quality/relevance
    # -----------------------------
    def _column_quality_score(self, s_any: Any, n: int) -> float:
        # quality ~ non-null + low unique ratio
        try:
            if self._is_polars():
                s = s_any # pl.Series
                non_null = float(1.0 - (s.null_count() / max(1, n)))
                nunique = float(s.n_unique())
                uniq_ratio = nunique / max(1.0, float(n))
                penalty = float(np.clip(uniq_ratio, 0.0, 1.0))
                quality = 0.65 * non_null + 0.35 * (1.0 - penalty)
                return float(np.clip(quality, 0.0, 1.0))
            else:
                s = s_any # pd.Series
                non_null = float(s.notna().mean()) if n > 0 else 0.0
                nunique = float(s.nunique(dropna=True)) if n > 0 else 0.0
                uniq_ratio = nunique / max(1.0, float(n))
                penalty = float(np.clip(uniq_ratio, 0.0, 1.0))
                quality = 0.65 * non_null + 0.35 * (1.0 - penalty)
                return float(np.clip(quality, 0.0, 1.0))
        except Exception:
            return 0.0
    def _column_relevance_score(self, s_any: Any, n: int) -> float:
        """
        relevance ~ entropy-like. 핵심: high-cardinality면 즉시 0점.
        """
        try:
            if self._is_polars():
                s = s_any # pl.Series
                nunique = int(s.n_unique())
                uniq_ratio = nunique / max(1, n)
                # ✅ dynamic early-exit
                if uniq_ratio > self._dynamic_early_exit_ratio(n):
                    return 0.0
                # 샘플링
                if n > self.relevance_sample_n:
                    # polars sample on series: convert to DataFrame
                    df_tmp = pl.DataFrame({"x": s})
                    df_tmp = df_tmp.sample(n=min(self.relevance_sample_n, n), seed=self.seed)
                    s2 = df_tmp["x"]
                else:
                    s2 = s
                # value_counts + entropy
                vc = s2.value_counts()
                # vc columns: ["x", "count"] (polars version varies)
                count_col = "count" if "count" in vc.columns else vc.columns[-1]
                counts = vc[count_col].to_list()
                total = float(sum(counts)) if counts else 0.0
                if total <= 0:
                    return 0.0
                p = np.array(counts, dtype=np.float64) / total
                ent = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
                # normalize-ish via tanh
                rel = float(np.tanh(ent))
                return float(np.clip(rel, 0.0, 1.0))
            else:
                s = s_any # pd.Series
                nunique = int(s.nunique(dropna=True))
                uniq_ratio = nunique / max(1, n)
                if uniq_ratio > self._dynamic_early_exit_ratio(n):
                    return 0.0
                if n > self.relevance_sample_n:
                    s2 = s.sample(n=min(self.relevance_sample_n, n), random_state=self.seed)
                else:
                    s2 = s
                vc = s2.value_counts(dropna=True, normalize=True)
                if len(vc) <= 1:
                    return 0.0
                p = vc.values.astype(np.float64)
                ent = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
                rel = float(np.tanh(ent))
                return float(np.clip(rel, 0.0, 1.0))
        except Exception:
            return 0.0
    def _score_one(self, df_any: Any, col: str) -> Tuple[str, Dict[str, float]]:
        n = self._len_df(df_any)
        if self._is_polars():
            s = df_any[col]
        else:
            s = df_any[col]
        qual = self._column_quality_score(s, n)
        rel = self._column_relevance_score(s, n)
        gold = 0.5 * rel + 0.5 * qual
        return col, {"quality": float(qual), "relevance": float(rel), "gold_score": float(gold)}
    def _make_score_df_meta(self, df_any: Any, cols: List[str]) -> Any:
        n = self._len_df(df_any)
        sample_n = min(self.relevance_sample_n, n)
        if self._is_polars():
            d = df_any.select(cols)
            if sample_n < n:
                d = d.sample(n=sample_n, seed=self.seed)
            return d
        # pandas
        d = df_any[cols]
        if sample_n < n:
            d = d.sample(n=sample_n, random_state=self.seed)
        return d
    def _score_columns(self, df_any: Any, cols: List[str], mode: str = "auto") -> List[Tuple[str, Dict[str, float]]]:
        mode = self._resolve_parallel_mode(mode)
        workers = self.autotune_workers(len(cols))
        self.max_workers = workers
        if mode == "sequential" or len(cols) < 5:
            return [self._score_one(df_any, c) for c in cols]
        if mode == "thread":
            from concurrent.futures import ThreadPoolExecutor
            self.timeline.add(TimelineEvent(
                ts=now_ts(), step=self._step, kind="parallel_scoring",
                severity="info", title="Parallel scoring start",
                detail={"mode": "thread", "workers": workers, "cols": len(cols)},
                tags=["perf", "parallel"]
            ))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(lambda c: self._score_one(df_any, c), cols))
        if mode == "process":
            # ✅ df 전체 전송 금지: meta만
            df_meta = self._make_score_df_meta(df_any, cols)
            # ⚠️ Windows에서 lambda/process는 터질 수 있음 → 안전하게: process는 기본적으로 linux 권장
            if os.name == "nt":
                # fallback to sequential to avoid pickling hell
                return [self._score_one(df_meta, c) for c in cols]
            from concurrent.futures import ProcessPoolExecutor
            self.timeline.add(TimelineEvent(
                ts=now_ts(), step=self._step, kind="parallel_scoring",
                severity="info", title="Parallel scoring start",
                detail={"mode": "process", "workers": workers, "cols": len(cols), "note": "meta-only"},
                tags=["perf", "parallel"]
            ))
            # process에서 self 메서드 피클링을 피하려면 top-level 함수가 정석.
            # 여기선 'linux + fork' 환경을 가정하고 최소한으로 유지.
            def _worker_score(col: str) -> Tuple[str, Dict[str, float]]:
                # df_meta는 closure로 캡쳐되지만 fork면 복사비용이 적고,
                # 무엇보다 df_full이 아니라 sample meta라 폭탄이 아님.
                n = len(df_meta) if hasattr(df_meta, "__len__") else 0
                if _HAS_POLARS and isinstance(df_meta, pl.DataFrame):
                    s = df_meta[col]
                    # 빠르게 점수
                    # (engine 파라미터는 closure로)
                    nunique = int(s.n_unique())
                    uniq_ratio = nunique / max(1, n)
                    if uniq_ratio > min(0.9, max(0.5, 5000 / max(1, n))):
                        rel = 0.0
                    else:
                        vc = s.value_counts()
                        count_col = "count" if "count" in vc.columns else vc.columns[-1]
                        counts = vc[count_col].to_list()
                        total = float(sum(counts)) if counts else 0.0
                        if total <= 0:
                            rel = 0.0
                        else:
                            p = np.array(counts, dtype=np.float64) / total
                            ent = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
                            rel = float(np.tanh(ent))
                    non_null = float(1.0 - (s.null_count() / max(1, n)))
                    uniq_ratio2 = nunique / max(1.0, float(n))
                    penalty = float(np.clip(uniq_ratio2, 0.0, 1.0))
                    qual = float(np.clip(0.65 * non_null + 0.35 * (1.0 - penalty), 0.0, 1.0))
                    gold = 0.5 * rel + 0.5 * qual
                    return col, {"quality": float(qual), "relevance": float(rel), "gold_score": float(gold)}
                # pandas fallback in worker
                if _HAS_PANDAS and isinstance(df_meta, pd.DataFrame):
                    s = df_meta[col]
                    n2 = int(len(s))
                    nunique = int(s.nunique(dropna=True))
                    uniq_ratio = nunique / max(1, n2)
                    if uniq_ratio > min(0.9, max(0.5, 5000 / max(1, n2))):
                        rel = 0.0
                    else:
                        vc = s.value_counts(dropna=True, normalize=True)
                        if len(vc) <= 1:
                            rel = 0.0
                        else:
                            p = vc.values.astype(np.float64)
                            ent = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
                            rel = float(np.tanh(ent))
                    non_null = float(s.notna().mean()) if n2 > 0 else 0.0
                    uniq_ratio2 = nunique / max(1.0, float(n2))
                    penalty = float(np.clip(uniq_ratio2, 0.0, 1.0))
                    qual = float(np.clip(0.65 * non_null + 0.35 * (1.0 - penalty), 0.0, 1.0))
                    gold = 0.5 * rel + 0.5 * qual
                    return col, {"quality": float(qual), "relevance": float(rel), "gold_score": float(gold)}
                return col, {"quality": 0.0, "relevance": 0.0, "gold_score": 0.0}
            with ProcessPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(_worker_score, cols))
        # fallback
        return [self._score_one(df_any, c) for c in cols]
    # -----------------------------
    # Main refine
    # -----------------------------
    def refine(self, df_in: Any, enable_row_filter: bool = True, min_non_null_ratio: float = 0.4) -> Tuple[Any, Dict[str, Any], List[ExplainCard]]:
        self._step += 1
        batch_id = self._new_batch_id()
        df = self._to_backend_df(df_in)
        cols = self._cols_df(df)
        nrows = self._len_df(df)
        ncols = len(cols)
        input_hash = hash_df_any(df)
        in_node = f"dataset:{input_hash}"
        prev_batch = self._dataset_producer.get(in_node)
        self.queue.append(QueueItem(
            ts=now_ts(), batch_id=batch_id, status="RUNNING",
            input_rows=int(nrows), input_cols=int(ncols),
            meta={"target_kpi": self.target_kpi or "", "input_hash": input_hash, "backend": self.backend}
        ))
        self.timeline.add(TimelineEvent(
            ts=now_ts(), step=self._step, kind="batch_start", severity="info",
            title=f"Batch Started: {batch_id}",
            incident_id=batch_id, tags=["batch"],
            detail={"rows": int(nrows), "cols": int(ncols), "input_hash": input_hash, "backend": self.backend}
        ))
        try:
            baseline_metrics = {
                "risk": float(np.clip(1.0 - self.threshold, 0.0, 1.0)),
                "loss": float(np.clip(1.5 - self.threshold, 0.0, 5.0)),
                "stability": float(np.clip(self.threshold, 0.0, 1.0)),
            }
            self.effect.start(batch_id=batch_id, step=self._step, baseline_metrics=baseline_metrics)
            kept_cols: List[str] = []
            dropped_cols: List[str] = []
            cards: List[ExplainCard] = []
            if self.target_kpi and self.target_kpi in cols:
                kept_cols.append(self.target_kpi)
            candidate_cols = [c for c in cols if c != self.target_kpi]
            scored = self._score_columns(df, candidate_cols, mode=self.parallel_mode)
            # 결정
            for col, scores in scored:
                gold = float(scores.get("gold_score", 0.0))
                decision = "KEEP" if gold >= self.threshold else "DROP"
                if decision == "KEEP":
                    kept_cols.append(col)
                else:
                    dropped_cols.append(col)
                cards.append(self.explain_builder.build(
                    batch_id=batch_id, column=col, decision=decision,
                    scores=scores, threshold=self.threshold
                ))
            # output df
            if self._is_polars():
                df_kept = df.select([c for c in kept_cols if c in cols]) if kept_cols else df.slice(0, 0)
                if enable_row_filter and df_kept.width > 0 and df_kept.height > 0:
                    min_valid = int(math.ceil(df_kept.width * float(min_non_null_ratio)))
                    # not_null count per row
                    mask = (pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in df_kept.columns]) >= min_valid)
                    df_out = df_kept.filter(mask)
                else:
                    df_out = df_kept
            else:
                df_kept = df[kept_cols] if kept_cols else df.iloc[0:0]
                if enable_row_filter and len(df_kept.columns) > 0 and len(df_kept) > 0:
                    min_valid = int(math.ceil(len(df_kept.columns) * float(min_non_null_ratio)))
                    mask = df_kept.notna().sum(axis=1) >= min_valid
                    df_out = df_kept.loc[mask]
                else:
                    df_out = df_kept
            output_hash = hash_df_any(df_out)
            out_node = f"dataset:{output_hash}"
            # PID update
            retention = len(kept_cols) / max(1, ncols)
            err = float(retention - self.target_retention)
            adj = float(self.pid.step(err))
            self.threshold = float(np.clip(self.threshold + adj, self.min_threshold, self.max_threshold))
            lineage = {
                "batch_id": batch_id,
                "kept": kept_cols,
                "dropped": dropped_cols,
                "retention": float(retention),
                "pid_adjust": float(adj),
                "threshold": float(self.threshold),
                "input_hash": input_hash,
                "output_hash": output_hash,
                "backend": self.backend,
                "parallel_mode": self._resolve_parallel_mode(self.parallel_mode),
                "workers": int(self.max_workers),
            }
            self.explain_cards[batch_id] = cards
            # DAG edges
            if prev_batch:
                self.dag.add_edge(DAGEdge(
                    src=in_node, dst=batch_id, kind="dataset_to_next_batch",
                    ts=now_ts(), meta={"linked_from_batch": prev_batch}
                ))
            self.dag.add_edge(DAGEdge(src=in_node, dst=batch_id, kind="input_to_batch", ts=now_ts(), meta={"batch_id": batch_id}))
            self.dag.add_edge(DAGEdge(src=batch_id, dst=out_node, kind="batch_to_output", ts=now_ts(), meta={"batch_id": batch_id}))
            self._dataset_producer[out_node] = batch_id
            after_metrics = {
                "risk": float(np.clip(1.0 - self.threshold, 0.0, 1.0)),
                "loss": float(np.clip(1.5 - self.threshold, 0.0, 5.0)),
                "stability": float(np.clip(self.threshold, 0.0, 1.0)),
            }
            self.effect.collect(batch_id, after_metrics)
            self.timeline.add(TimelineEvent(
                ts=now_ts(), step=self._step, kind="refine_done", severity="info",
                title=f"Refine Done: kept={len(kept_cols)} dropped={len(dropped_cols)}",
                incident_id=batch_id, tags=["batch", "refine"], detail=lineage
            ))
            self.queue.append(QueueItem(
                ts=now_ts(), batch_id=batch_id, status="OK",
                input_rows=int(nrows), input_cols=int(ncols),
                kept_cols=int(len(kept_cols)), dropped_cols=int(len(dropped_cols)),
                threshold=float(self.threshold),
                meta={"target_kpi": self.target_kpi or "", "input_hash": input_hash, "output_hash": output_hash, "backend": self.backend}
            ))
            # flush stores
            self.timeline.flush()
            self.queue.flush()
            self.dag.flush()
            return df_out, lineage, cards
        except Exception as e:
            tb = traceback.format_exc(limit=3)
            self.timeline.add(TimelineEvent(
                ts=now_ts(), step=self._step, kind="error", severity="high",
                title=f"Batch Failed: {batch_id}",
                incident_id=batch_id, tags=["error"],
                detail={"error": str(e), "trace": tb}
            ))
            self.queue.append(QueueItem(
                ts=now_ts(), batch_id=batch_id, status="FAIL",
                input_rows=int(nrows), input_cols=int(ncols),
                error=str(e),
                meta={"target_kpi": self.target_kpi or "", "input_hash": input_hash, "backend": self.backend}
            ))
            self.timeline.flush()
            self.queue.flush()
            raise
# ==============================================================================
# Benchmark utilities
# ==============================================================================
def make_df(n: int, cols: int, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    base = {
        "income": rng.normal(5000, 120, n),
        "score": rng.normal(0, 1, n),
        "age": rng.integers(20, 70, n),
        "city": rng.choice(["Seoul", "Busan", "Incheon", "Daegu"], n),
        "noise": ["x"] * n,
        "id": np.arange(n),
    }
    # add extra columns
    for i in range(max(0, cols - len(base))):
        base[f"c{i:03d}"] = rng.normal(0, 1, n)
    # inject missing
    mask = rng.random(n) < 0.15
    arr = np.array(base["score"], copy=True)
    arr[mask] = np.nan
    base["score"] = arr
    return base
def bench_rows(engine: DataRefineryEngine, sizes: List[int], cols: int, repeats: int = 3) -> None:
    print(f"\n🚀 Benchmark backend={engine.backend} parallel={engine.parallel} mode={engine.parallel_mode} workers(cap)={engine.max_workers_cap}")
    for n in sizes:
        t_list = []
        thr0 = engine.threshold
        for r in range(repeats):
            data = make_df(n, cols, seed=engine.seed + r)
            df_out, _, _ = engine.refine(data)
            # crude timing: refine 자체의 내부 시간을 재려면 외부에서 perf_counter로 감싸면 됨
            # 여기선 간단히 전체를 재서 보고 싶으면 아래처럼:
            # (지금은 이미 엔진 내부 logging 위주로, 벤치는 main에서 perf_counter로 진행)
        # main에서 실제 측정함
def run_benchmark(engine: DataRefineryEngine, sizes: List[int], cols: int, repeats: int = 3) -> None:
    print(f"\n🚀 Benchmark backend={engine.backend} parallel={engine.parallel} mode={engine.parallel_mode} workers=auto(cap {engine.max_workers_cap})")
    for n in sizes:
        times = []
        thr_before = engine.threshold
        for r in range(repeats):
            data = make_df(n, cols, seed=engine.seed + r)
            t0 = time.perf_counter()
            engine.refine(data)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        mn = min(times)
        print(f"N={n:,} avg={avg:.3f}s min={mn:.3f}s thr={thr_before:.3f}")
def run_col_sweep(engine: DataRefineryEngine, rows_fixed: int, col_list: List[int], repeats: int = 3) -> None:
    print(f"\n🧪 Column Sweep backend={engine.backend} rows={rows_fixed:,} parallel={engine.parallel} mode={engine.parallel_mode}")
    for c in col_list:
        times = []
        for r in range(repeats):
            data = make_df(rows_fixed, c, seed=engine.seed + r)
            t0 = time.perf_counter()
            engine.refine(data)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        print(f"Cols={c:<4d} avg={avg:.3f}s")
# ==============================================================================
# CLI
# ==============================================================================
def parse_int_list(s: str) -> List[int]:
    out = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="benchmark", help="benchmark|col-sweep|demo")
    ap.add_argument("--backend", type=str, default="auto", help="auto|polars|pandas")
    ap.add_argument("--parallel", action="store_true", help="enable parallel scoring")
    ap.add_argument("--parallel-mode", type=str, default="auto", help="auto|sequential|thread|process")
    ap.add_argument("--sizes", type=str, default="10000,50000,100000,500000")
    ap.add_argument("--cols", type=int, default=50, help="number of columns for benchmark")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rows-fixed", type=int, default=100000)
    ap.add_argument("--col-sweep", type=str, default="20,50,100,200,500")
    ap.add_argument("--runs-dir", type=str, default="runs")
    args = ap.parse_args()
    engine = DataRefineryEngine(
        runs_dir=args.runs_dir,
        target_kpi="income",
        parallel=bool(args.parallel),
        parallel_mode=str(args.parallel_mode),
    )
    # backend override
    if args.backend == "polars":
        if not _HAS_POLARS:
            raise RuntimeError("polars not installed")
        engine.backend = "polars"
    elif args.backend == "pandas":
        if not _HAS_PANDAS:
            raise RuntimeError("pandas not installed")
        engine.backend = "pandas"
    else:
        engine.backend = "polars" if _HAS_POLARS else "pandas"
    if args.mode == "benchmark":
        sizes = parse_int_list(args.sizes)
        run_benchmark(engine, sizes=sizes, cols=int(args.cols), repeats=int(args.repeats))
        return
    if args.mode == "col-sweep":
        cols_list = parse_int_list(args.col_sweep)
        run_col_sweep(engine, rows_fixed=int(args.rows_fixed), col_list=cols_list, repeats=int(args.repeats))
        return
    if args.mode == "demo":
        data = make_df(1000, 50, seed=0)
        out, lineage, cards = engine.refine(data)
        print("Lineage:", safe_json(lineage))
        print("First cards:", safe_json([asdict(cards[i]) for i in range(min(3, len(cards)))]))
        # DAGRunner demo
        runner = DAGRunner(engine.timeline, engine.queue)
        runner.add(DAGNode("load", "Load", lambda inp, deps: data))
        runner.add(DAGNode("refine", "Refine", lambda inp, deps: engine.refine(deps["load"])[0], deps=["load"]))
        res = runner.run(None)
        print("DAG keys:", list(res.keys()))
        print("DOT:\n", runner.dot())
        return
    raise SystemExit(f"Unknown mode: {args.mode}")
if __name__ == "__main__":
    main()
