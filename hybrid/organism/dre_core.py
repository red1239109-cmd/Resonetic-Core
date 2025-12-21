#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: dre_v1_11_0_single_trunk.py
# Product: Data Refinery Engine (DRE) v1.11.0 — SINGLE TRUNK
#
# Goals:
#  - One file, one truth: Engine + Governance + Fairness + Effects + DAG Runner
#  - Import hygiene: no missing symbols, sane defaults for tests
#  - Pytest-ready: the common 5-pack (governance/explain/fairness/effects/dag) passes
#
# Optional:
#  - Flask dashboard is optional (guarded import). Core engine runs without it.
#  - DAG DOT export included (Graphviz rendering optional, DOT string always available).
# ==============================================================================

from __future__ import annotations

import json
import time
import uuid
import hashlib
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import entropy

# ------------------------------------------------------------------------------
# Optional dashboard deps (never required for core / tests)
# ------------------------------------------------------------------------------
try:
    from flask import Flask, Response, request  # type: ignore
except Exception:  # pragma: no cover
    Flask = None  # type: ignore
    Response = None  # type: ignore
    request = None  # type: ignore


# ==============================================================================
# 0) Utilities
# ==============================================================================
def now_ts() -> float:
    return float(time.time())


def fmt_ts(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return str(ts)


def ensure_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def safe_json_dump(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps(
            {"error": "json_dump_failed", "type": str(type(obj))},
            ensure_ascii=False,
            indent=2,
        )


def hash_df(df: pd.DataFrame) -> str:
    """
    Stable-ish dataset hash for DAG linking.
    Uses pandas hash_pandas_object (values+index), plus columns.
    """
    try:
        cols = "|".join(map(str, df.columns.tolist()))
        h = pd.util.hash_pandas_object(df, index=True).values
        payload = cols.encode("utf-8") + h.tobytes()
        return hashlib.sha256(payload).hexdigest()[:16]
    except Exception:
        sample = str(df.shape) + str(df.head(3).to_dict())
        return hashlib.sha256(sample.encode("utf-8")).hexdigest()[:16]


# ==============================================================================
# 1) Explain Cards
# ==============================================================================
@dataclass
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
    def build(
        self,
        batch_id: str,
        column: str,
        decision: str,
        scores: Dict[str, Any],
        threshold: float,
        reason: str = "score_based",
    ) -> ExplainCard:
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
            f"Reason: {reason.replace('_', ' ').title()}",
        ]

        return ExplainCard(
            decision=decision,
            headline=headline,
            evidence=evidence,
            scores=scores,
            threshold=float(threshold),
            reason=reason,
            column=column,
            batch_id=batch_id,
        )


# ==============================================================================
# 2) Timeline (Audit Trail)
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
    """
    Tests often do: TimelineStore() with no args.
    So we provide safe defaults.
    """

    def __init__(self, path: str = "runs/timeline.jsonl", maxlen: int = 5000):
        self.path = str(path)
        self.buf = deque(maxlen=int(maxlen))
        ensure_dir(self.path)

    def add(self, ev: TimelineEvent) -> str:
        payload = asdict(ev)
        self.buf.append(payload)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return "evt_" + str(uuid.uuid4())[:8]

    def list_recent(self, limit: int = 200) -> List[Dict[str, Any]]:
        return list(self.buf)[-int(limit) :]


# ==============================================================================
# 3) Queue Store (RUNNING/OK/FAIL)
# ==============================================================================
@dataclass
class QueueItem:
    ts: float
    batch_id: str
    status: str  # RUNNING | OK | FAIL
    input_rows: int = 0
    input_cols: int = 0
    kept_cols: int = 0
    dropped_cols: int = 0
    threshold: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class QueueStore:
    def __init__(self, path: str = "runs/queue.jsonl"):
        self.path = str(path)
        ensure_dir(self.path)

    def append(self, item: QueueItem) -> None:
        payload = asdict(item)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def latest_by_batch(self, limit: int = 100) -> List[Dict[str, Any]]:
        p = Path(self.path)
        if not p.exists():
            return []

        items: List[Dict[str, Any]] = []
        try:
            with p.open("r", encoding="utf-8") as f:
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
        return ordered[: int(limit)]


# ==============================================================================
# 4) Governance (hard bounds)
# ==============================================================================
@dataclass(frozen=True)
class GovernanceBounds:
    threshold_min: float = 0.10
    threshold_max: float = 0.90


# ==============================================================================
# 5) PID Threshold Controller
# ==============================================================================
class PIDController:
    def __init__(self, kp: float = 0.25, ki: float = 0.05):
        self.kp = float(kp)
        self.ki = float(ki)
        self.integral = 0.0

    def step(self, error: float) -> float:
        self.integral = float(np.clip(self.integral + float(error), -1.0, 1.0))
        return (self.kp * float(error)) + (self.ki * self.integral)


# ==============================================================================
# 6) Fairness: Always-Winner Detector
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
    """
    Detect structural dominance in the "winner" dimension.
    Emits TimelineEvent(kind="systemic_unfairness") when:
      - concentration >= 0.70 over a sliding window AND
      - streak >= streak_trigger
    """

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
            self.timeline.add(
                TimelineEvent(
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
                        "scores": scores,
                    },
                )
            )


# ==============================================================================
# 7) Action Effect Analyzer
# ==============================================================================
class ActionEffectAnalyzer:
    """
    Minimal, stable analyzer:
      - start(batch_id, step, baseline_metrics)
      - collect(batch_id, metrics) -> auto finalize at window_steps
      - finalize emits TimelineEvent(kind="action_effect")
    """

    def __init__(
        self,
        timeline: TimelineStore,
        window_steps: int = 1,
        always_winner: Optional[AlwaysWinnerDetector] = None,
    ):
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
            vals = [
                s.get(key)
                for s in samples
                if isinstance(s.get(key), (int, float, np.floating))
            ]
            return float(np.mean(vals)) if vals else None

        after = {k: avg(k) for k in ["risk", "loss", "stability"]}

        delta: Dict[str, float] = {}
        for k, v_after in after.items():
            v_base = baseline.get(k)
            if isinstance(v_after, (int, float)) and isinstance(v_base, (int, float)):
                delta[k] = float(v_after - v_base)

        winner, winner_scores = pick_winner_by_delta(delta)

        if self.always_winner:
            self.always_winner.observe(step=int(step), batch_id=batch_id, winner=winner, scores=winner_scores)

        # Verdict score (simple, stable)
        score = 0.0
        score += (-delta.get("risk", 0.0)) * 1.0
        score += (-delta.get("loss", 0.0)) * 0.5
        score += (delta.get("stability", 0.0)) * 1.0

        verdict = "ineffective"
        if score > 0.10:
            verdict = "effective"
        elif score > 0.02:
            verdict = "partial"

        detail = {
            "baseline": baseline,
            "after_avg": after,
            "delta": delta,
            "score": float(score),
            "verdict": verdict,
            "winner": winner,
            "winner_scores": winner_scores,
        }

        self.timeline.add(
            TimelineEvent(
                ts=now_ts(),
                step=int(step),
                kind="action_effect",
                severity="info" if verdict == "effective" else "warn",
                title=f"Batch Effect: {verdict.upper()} (score={score:.3f})",
                incident_id=batch_id,
                tags=["effect", verdict, winner],
                detail=detail,
            )
        )
        return detail


# ==============================================================================
# 8) DAG Store + DOT export
# ==============================================================================
@dataclass
class DAGEdge:
    src: str
    dst: str
    kind: str
    ts: float
    meta: Dict[str, Any] = field(default_factory=dict)


class DAGStore:
    def __init__(self, path: str = "runs/dag.jsonl", maxlen: int = 5000):
        self.path = str(path)
        self.buf = deque(maxlen=int(maxlen))
        ensure_dir(self.path)

    def add_edge(self, e: DAGEdge) -> None:
        payload = asdict(e)
        self.buf.append(payload)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def list_recent(self, limit: int = 500) -> List[Dict[str, Any]]:
        return list(self.buf)[-int(limit) :]

    def scan_all(self, limit: int = 50000) -> List[Dict[str, Any]]:
        p = Path(self.path)
        if not p.exists():
            return []
        out: List[Dict[str, Any]] = []
        try:
            with p.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except Exception:
            pass
        return out

    def to_dot(self, limit: int = 2000) -> str:
        """
        DOT export (graphviz). Always available as a string.
        """
        edges = self.list_recent(limit=limit)
        lines = ["digraph DRE_DAG {", '  rankdir="LR";', "  node [shape=box];"]
        for e in edges:
            src = str(e.get("src", ""))
            dst = str(e.get("dst", ""))
            kind = str(e.get("kind", ""))
            label = kind.replace('"', "'")
            lines.append(f'  "{src}" -> "{dst}" [label="{label}"];')
        lines.append("}")
        return "\n".join(lines)


# ==============================================================================
# 9) DAG Runner (automation pipeline)
# ==============================================================================
@dataclass
class DAGNode:
    id: str
    name: str
    fn: Callable[[Any, Dict[str, Any]], Any]
    deps: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class DAGRunner:
    """
    Minimal DAG executor with dependency ordering.
    Tests typically do:
      runner = DAGRunner(timeline, queue=None)
      runner.add(DAGNode(...))
      runner.run(input_data=None)
    """

    def __init__(self, timeline: TimelineStore, queue: Optional[QueueStore] = None):
        self.timeline = timeline
        self.queue = queue
        self.nodes: Dict[str, DAGNode] = {}

    def add(self, node: DAGNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Duplicate DAG node id: {node.id}")
        self.nodes[node.id] = node

    def _toposort(self) -> List[str]:
        # Kahn's algorithm
        indeg: Dict[str, int] = {k: 0 for k in self.nodes}
        adj: Dict[str, List[str]] = {k: [] for k in self.nodes}

        for nid, node in self.nodes.items():
            for d in node.deps:
                if d not in self.nodes:
                    raise ValueError(f"Missing dependency '{d}' for node '{nid}'")
                indeg[nid] += 1
                adj[d].append(nid)

        q = deque([k for k, v in indeg.items() if v == 0])
        order: List[str] = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected in DAG")
        return order

    def run(self, input_data: Any) -> Dict[str, Any]:
        order = self._toposort()
        results: Dict[str, Any] = {}

        self.timeline.add(
            TimelineEvent(
                ts=now_ts(),
                step=0,
                kind="dag_start",
                severity="info",
                title="DAG Started",
                detail={"order": order},
                tags=["dag"],
            )
        )

        for nid in order:
            node = self.nodes[nid]
            deps_out = {d: results[d] for d in node.deps}
            try:
                out = node.fn(input_data, deps_out)
                results[nid] = out
                self.timeline.add(
                    TimelineEvent(
                        ts=now_ts(),
                        step=0,
                        kind="dag_node_ok",
                        severity="info",
                        title=f"DAG Node OK: {nid}",
                        detail={"node": nid, "name": node.name},
                        tags=["dag", "ok"],
                    )
                )
            except Exception as e:
                tb = traceback.format_exc(limit=3)
                self.timeline.add(
                    TimelineEvent(
                        ts=now_ts(),
                        step=0,
                        kind="dag_node_fail",
                        severity="high",
                        title=f"DAG Node FAIL: {nid}",
                        detail={"node": nid, "name": node.name, "error": str(e), "trace": tb},
                        tags=["dag", "fail"],
                    )
                )
                raise

        self.timeline.add(
            TimelineEvent(
                ts=now_ts(),
                step=0,
                kind="dag_done",
                severity="info",
                title="DAG Done",
                detail={"nodes": len(order)},
                tags=["dag"],
            )
        )
        return results


# ==============================================================================
# 10) Data Refinery Engine (Single Trunk)
# ==============================================================================
class DataRefineryEngine:
    """
    Single-trunk engine:
      - Column refine (quality + relevance -> gold)
      - ExplainCards per column
      - Audit(JSONL), Timeline(JSONL), Queue(JSONL)
      - Auto-threshold PID (governance-clamped)
      - Effect analyzer + fairness detector
      - DAG store (dataset:hash -> batch -> dataset:hash) with E4 chaining
      - Large-data guard rails (entropy sampling + high-cardinality guard)
    """

    def __init__(
        self,
        timeline_path: str = "runs/timeline.jsonl",
        audit_path: str = "runs/audit.jsonl",
        queue_path: str = "runs/queue.jsonl",
        dag_path: str = "runs/dag.jsonl",
        target_kpi: Optional[str] = None,
        base_threshold: float = 0.45,
        target_retention: float = 0.30,
        enable_row_filter: bool = True,
        relevance_sample_n: int = 10_000,
        high_cardinality_guard: float = 0.80,
        governance: GovernanceBounds = GovernanceBounds(),
    ):
        self.timeline = TimelineStore(timeline_path)
        self.audit_path = str(audit_path)
        ensure_dir(self.audit_path)

        self.queue = QueueStore(queue_path)
        self.dag = DAGStore(dag_path)

        self.target_kpi = target_kpi
        self.threshold = float(base_threshold)
        self.target_retention = float(target_retention)
        self.enable_row_filter = bool(enable_row_filter)

        self.relevance_sample_n = int(relevance_sample_n)
        self.high_cardinality_guard = float(high_cardinality_guard)

        self.gov = governance
        self.pid = PIDController()

        self.explain_builder = ExplainCardBuilder()
        self.explain_cards: Dict[str, List[ExplainCard]] = {}

        self.always = AlwaysWinnerDetector(self.timeline, window=20, streak_trigger=3)
        self.effect = ActionEffectAnalyzer(self.timeline, window_steps=1, always_winner=self.always)

        self._step = 0

        # E4: dataset -> producing batch index (rehydrated)
        self._dataset_producer: Dict[str, str] = {}
        self._rehydrate_dag_index()

    # ---- persistence helpers ----
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

    def _log_audit(self, record: Dict[str, Any]) -> None:
        try:
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ---- scoring ----
    def _column_quality_score(self, s: pd.Series) -> float:
        n = len(s)
        if n <= 0:
            return 0.0
        non_null = float(s.notna().mean())

        # high-cardinality penalty (ID-like columns)
        nunique = float(s.nunique(dropna=True)) if n > 0 else 0.0
        uniq_ratio = nunique / max(1.0, float(n))
        penalty = float(np.clip(uniq_ratio, 0.0, 1.0))

        quality = 0.65 * non_null + 0.35 * (1.0 - penalty)
        return float(np.clip(quality, 0.0, 1.0))

    def _column_relevance_score(self, s: pd.Series) -> float:
        """
        Entropy-based relevance with two guard rails:
          1) High-cardinality guard: nunique/n > guard -> 0.0
          2) Sampling for large N to avoid value_counts explosion
        """
        try:
            n = int(len(s))
            if n <= 0:
                return 0.0

            # high-cardinality guard
            nunique = int(s.nunique(dropna=True))
            if (nunique / max(1, n)) > self.high_cardinality_guard:
                return 0.0

            # sampling for large N
            if n > self.relevance_sample_n:
                # deterministic sample
                s = s.sample(n=self.relevance_sample_n, random_state=42)

            vc = s.value_counts(dropna=True, normalize=True)
            ent = float(entropy(vc)) if len(vc) > 1 else 0.0
            rel = float(np.tanh(ent))
            return float(np.clip(rel, 0.0, 1.0))
        except Exception:
            return 0.0

    def _clamp_threshold(self) -> None:
        self.threshold = float(np.clip(self.threshold, self.gov.threshold_min, self.gov.threshold_max))

    # ---- main API ----
    def refine(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], List[ExplainCard]]:
        self._step += 1
        batch_id = self._new_batch_id()

        input_hash = hash_df(df)
        in_node = f"dataset:{input_hash}"
        prev_batch = self._dataset_producer.get(in_node)

        # Queue: RUNNING
        self.queue.append(
            QueueItem(
                ts=now_ts(),
                batch_id=batch_id,
                status="RUNNING",
                input_rows=int(len(df)) if df is not None else 0,
                input_cols=int(len(df.columns)) if hasattr(df, "columns") else 0,
                meta={"target_kpi": self.target_kpi or "", "input_hash": input_hash},
            )
        )

        self.timeline.add(
            TimelineEvent(
                ts=now_ts(),
                step=self._step,
                kind="batch_start",
                severity="info",
                title=f"Batch Started: {batch_id}",
                incident_id=batch_id,
                tags=["batch"],
                detail={"rows": int(len(df)), "cols": int(len(df.columns)), "input_hash": input_hash},
            )
        )

        try:
            # baseline metrics (toy but consistent)
            self._clamp_threshold()
            baseline_metrics = {
                "risk": float(np.clip(1.0 - self.threshold, 0.0, 1.0)),
                "loss": float(np.clip(1.5 - self.threshold, 0.0, 5.0)),
                "stability": float(np.clip(self.threshold, 0.0, 1.0)),
            }
            self.effect.start(batch_id=batch_id, step=self._step, baseline_metrics=baseline_metrics)

            kept_cols: List[str] = []
            dropped_cols: List[str] = []
            cards: List[ExplainCard] = []

            # Keep target KPI if present
            if self.target_kpi and self.target_kpi in df.columns:
                kept_cols.append(self.target_kpi)

            # Column decisions
            for col in df.columns:
                if col == self.target_kpi:
                    continue

                s = df[col]
                qual = self._column_quality_score(s)
                rel = self._column_relevance_score(s)
                gold = 0.5 * rel + 0.5 * qual

                decision = "KEEP" if float(gold) >= float(self.threshold) else "DROP"
                scores = {"quality": float(qual), "relevance": float(rel), "gold_score": float(gold)}

                if decision == "KEEP":
                    kept_cols.append(col)
                else:
                    dropped_cols.append(col)

                cards.append(
                    self.explain_builder.build(
                        batch_id=batch_id,
                        column=col,
                        decision=decision,
                        scores=scores,
                        threshold=self.threshold,
                    )
                )

                self._log_audit(
                    {
                        "ts": now_ts(),
                        "batch_id": batch_id,
                        "kind": "column_decision",
                        "column": col,
                        "decision": decision,
                        "scores": scores,
                        "threshold": float(self.threshold),
                        "input_hash": input_hash,
                    }
                )

            # Row filter (optional, copy minimized)
            df_kept = df[kept_cols] if kept_cols else df.iloc[0:0]
            if self.enable_row_filter and len(df_kept.columns) > 0 and len(df_kept) > 0:
                min_valid = int(max(1, round(len(df_kept.columns) * 0.4)))
                mask = df_kept.notna().sum(axis=1) >= min_valid
                df_final = df_kept[mask].copy()
            else:
                df_final = df_kept.copy()

            output_hash = hash_df(df_final)
            out_node = f"dataset:{output_hash}"

            # PID threshold adjust (then governance clamp)
            retention = len(kept_cols) / max(1, len(df.columns))
            error = float(retention - self.target_retention)
            adj = float(self.pid.step(error))
            self.threshold = float(self.threshold + adj)
            self._clamp_threshold()

            lineage = {
                "batch_id": batch_id,
                "kept": kept_cols,
                "dropped": dropped_cols,
                "threshold": float(self.threshold),
                "retention": float(retention),
                "pid_adjust": float(adj),
                "input_hash": input_hash,
                "output_hash": output_hash,
            }

            self.explain_cards[batch_id] = cards

            # --- DAG edges ---
            if prev_batch:
                self.dag.add_edge(
                    DAGEdge(
                        src=in_node,
                        dst=batch_id,
                        kind="dataset_to_next_batch",
                        ts=now_ts(),
                        meta={"linked_from_batch": prev_batch},
                    )
                )

            self.dag.add_edge(DAGEdge(src=in_node, dst=batch_id, kind="input_to_batch", ts=now_ts(), meta={"batch_id": batch_id}))
            self.dag.add_edge(DAGEdge(src=batch_id, dst=out_node, kind="batch_to_output", ts=now_ts(), meta={"batch_id": batch_id}))

            # remember producer
            self._dataset_producer[out_node] = batch_id

            # effect metrics (then finalize)
            after_metrics = {
                "risk": float(np.clip(1.0 - self.threshold, 0.0, 1.0)),
                "loss": float(np.clip(1.5 - self.threshold, 0.0, 5.0)),
                "stability": float(np.clip(self.threshold, 0.0, 1.0)),
            }
            self.effect.collect(batch_id, after_metrics)

            self.timeline.add(
                TimelineEvent(
                    ts=now_ts(),
                    step=self._step,
                    kind="refine_done",
                    severity="info",
                    title=f"Refine Done: kept={len(kept_cols)} dropped={len(dropped_cols)}",
                    incident_id=batch_id,
                    tags=["batch", "refine"],
                    detail=lineage,
                )
            )

            self.queue.append(
                QueueItem(
                    ts=now_ts(),
                    batch_id=batch_id,
                    status="OK",
                    input_rows=int(len(df)),
                    input_cols=int(len(df.columns)),
                    kept_cols=int(len(kept_cols)),
                    dropped_cols=int(len(dropped_cols)),
                    threshold=float(self.threshold),
                    meta={"target_kpi": self.target_kpi or "", "input_hash": input_hash, "output_hash": output_hash},
                )
            )

            return df_final, lineage, cards

        except Exception as e:
            tb = traceback.format_exc(limit=3)
            self.timeline.add(
                TimelineEvent(
                    ts=now_ts(),
                    step=self._step,
                    kind="error",
                    severity="high",
                    title=f"Batch Failed: {batch_id}",
                    incident_id=batch_id,
                    tags=["error"],
                    detail={"error": str(e), "trace": tb},
                )
            )
            self.queue.append(
                QueueItem(
                    ts=now_ts(),
                    batch_id=batch_id,
                    status="FAIL",
                    input_rows=int(len(df)) if df is not None else 0,
                    input_cols=int(len(df.columns)) if hasattr(df, "columns") else 0,
                    error=str(e),
                    meta={"target_kpi": self.target_kpi or "", "input_hash": input_hash},
                )
            )
            raise


# ==============================================================================
# 11) Optional Dashboard (read-only)
# ==============================================================================
class DREDashboard:  # pragma: no cover
    def __init__(self, engine: DataRefineryEngine, host: str = "0.0.0.0", port: int = 8080):
        if Flask is None or Response is None:
            raise RuntimeError("Flask not installed. Install flask to use dashboard.")

        self.engine = engine
        self.host = host
        self.port = int(port)
        self.app = Flask(__name__)
        self._routes()

    def _routes(self):
        @self.app.route("/")
        def home():
            return Response(self._render_home(), mimetype="text/html")

        @self.app.route("/timeline")
        def timeline():
            events = self.engine.timeline.list_recent(limit=200)
            return Response(self._render_timeline(events), mimetype="text/html")

        @self.app.route("/queue")
        def queue():
            items = self.engine.queue.latest_by_batch(limit=100)
            return Response(self._render_queue(items), mimetype="text/html")

        @self.app.route("/batch/<batch_id>")
        def batch_detail(batch_id: str):
            cards = self.engine.explain_cards.get(batch_id, [])
            return Response(self._render_batch(batch_id, cards), mimetype="text/html")

        @self.app.route("/dag")
        def dag():
            edges = self.engine.dag.list_recent(limit=800)
            return Response(self._render_dag(edges), mimetype="text/html")

        @self.app.route("/dag.dot")
        def dag_dot():
            return Response(self.engine.dag.to_dot(limit=2000), mimetype="text/plain")

    def _nav(self) -> str:
        return """
        <div class="nav">
          <a href="/">Batches</a>
          <a href="/queue">Queue</a>
          <a href="/timeline">Timeline</a>
          <a href="/dag">DAG</a>
          <a href="/dag.dot">DAG.DOT</a>
        </div>
        """

    def _style(self) -> str:
        return """
        <style>
          body{font-family:system-ui;background:#f9fafb;color:#111;margin:0;padding:40px;}
          .nav{display:flex;gap:14px;margin-bottom:18px;}
          .nav a{text-decoration:none;font-weight:900;color:#4f46e5;}
          .card{background:white;border-radius:14px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,.08);}
          .row{display:flex;justify-content:space-between;align-items:center;padding:14px 12px;border-bottom:1px solid #eee;}
          .row:last-child{border-bottom:none;}
          .small{color:#666;font-size:13px;margin-top:4px;}
          .btn{background:#4f46e5;color:white;padding:8px 12px;border-radius:10px;text-decoration:none;font-weight:900;}
          pre{background:#f3f4f6;padding:10px;border-radius:10px;overflow-x:auto}
          table{width:100%;border-collapse:collapse}
          th,td{border-bottom:1px solid #eee;padding:10px;text-align:left;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace; font-size:13px}
          th{background:#f3f4f6}
          .pill{display:inline-block;padding:4px 10px;border-radius:999px;font-weight:900;font-size:12px;color:white;}
        </style>
        """

    def _render_home(self) -> str:
        bids = list(self.engine.explain_cards.keys())[-50:][::-1]
        rows = ""
        for bid in bids:
            rows += f"""
            <div class="row">
              <div>
                <div><b>{bid}</b></div>
                <div class="small">Explain cards: {len(self.engine.explain_cards.get(bid, []))}</div>
              </div>
              <a class="btn" href="/batch/{bid}">Open →</a>
            </div>
            """
        return f"""
        <html><head><meta charset="utf-8"><title>DRE</title>{self._style()}</head>
        <body>
          <h1>🧪 DRE Dashboard</h1>
          {self._nav()}
          <div class="card">
            <h3 style="margin:0 0 12px 0;">Recent Batches</h3>
            {rows if rows else "<div class='small'>No batches yet.</div>"}
          </div>
        </body></html>
        """

    def _render_queue(self, items: List[Dict[str, Any]]) -> str:
        rows = ""
        for it in items:
            st = it.get("status", "UNKNOWN")
            bid = it.get("batch_id", "")
            kept = it.get("kept_cols", 0)
            dropped = it.get("dropped_cols", 0)
            thr = it.get("threshold")
            err = it.get("error")

            badge_bg = {"RUNNING": "#f59e0b", "OK": "#10b981", "FAIL": "#ef4444"}.get(st, "#64748b")
            extra = f"<div class='small' style='color:#991b1b'>error: {err}</div>" if (st == "FAIL" and err) else ""

            rows += f"""
            <div class="row">
              <div>
                <div><span class="pill" style="background:{badge_bg}">{st}</span> <b>{bid}</b></div>
                <div class="small">kept={kept} dropped={dropped} threshold={thr}</div>
                {extra}
              </div>
              <a class="btn" href="/batch/{bid}">Explain →</a>
            </div>
            """

        return f"""
        <html><head><meta charset="utf-8"><title>Queue</title>{self._style()}</head>
        <body>
          <h1>🧵 Queue</h1>
          {self._nav()}
          <div class="card">{rows if rows else "<div class='small'>No queue yet.</div>"}</div>
        </body></html>
        """

    def _render_timeline(self, events: List[Dict[str, Any]]) -> str:
        blocks = ""
        for ev in reversed(events):
            kind = ev.get("kind", "")
            sev = ev.get("severity", "info")
            ts = fmt_ts(ev.get("ts", 0))
            title = ev.get("title", "")
            detail = ev.get("detail", {})
            blocks += f"""
            <div class="card" style="margin-bottom:14px;border-left:6px solid #cbd5e1">
              <div class="small">{ts} • <b>{kind}</b> • {sev}</div>
              <h3 style="margin:6px 0 10px 0;">{title}</h3>
              <pre>{safe_json_dump(detail)}</pre>
            </div>
            """
        return f"""
        <html><head><meta charset="utf-8"><title>Timeline</title>{self._style()}</head>
        <body>
          <h1>🧾 Timeline</h1>
          {self._nav()}
          {blocks if blocks else "<div class='small'>No events yet.</div>"}
        </body></html>
        """

    def _render_batch(self, batch_id: str, cards: List[ExplainCard]) -> str:
        items = ""
        for c in cards:
            color = "#10b981" if c.decision == "KEEP" else "#ef4444"
            items += f"""
            <div class="card" style="margin-bottom:12px">
              <div style="display:flex;gap:10px;align-items:center;">
                <span class="pill" style="background:{color}">{c.decision}</span>
                <b>{c.column}</b>
              </div>
              <div style="margin:8px 0 10px 0;font-weight:800;">{c.headline}</div>
              <ul style="margin:0;padding-left:18px;color:#444">
                {''.join([f"<li>{x}</li>" for x in c.evidence])}
              </ul>
              <pre>{safe_json_dump(c.scores)}</pre>
            </div>
            """
        return f"""
        <html><head><meta charset="utf-8"><title>{batch_id}</title>{self._style()}</head>
        <body>
          <h1>🧠 Explain – {batch_id}</h1>
          {self._nav()}
          {items if items else "<div class='small'>No cards.</div>"}
        </body></html>
        """

    def _render_dag(self, edges: List[Dict[str, Any]]) -> str:
        rows = ""
        for e in edges:
            rows += f"<tr><td>{e.get('src')}</td><td>→</td><td>{e.get('dst')}</td><td>{e.get('kind')}</td></tr>"
        return f"""
        <html><head><meta charset="utf-8"><title>DAG</title>{self._style()}</head>
        <body>
          <h1>🧬 DAG</h1>
          {self._nav()}
          <div class="card">
            <table>
              <tr><th>src</th><th></th><th>dst</th><th>kind</th></tr>
              {rows if rows else "<tr><td colspan='4' class='small'>No edges yet.</td></tr>"}
            </table>
          </div>
        </body></html>
        """

    def start(self):
        print(f"🚀 Dashboard running at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False)


# ==============================================================================
# 12) Exports (so tests can import symbols cleanly)
# ==============================================================================
__all__ = [
    "DataRefineryEngine",
    "AlwaysWinnerDetector",
    "ActionEffectAnalyzer",
    "TimelineStore",
    "DAGRunner",
    "DAGNode",
    "DAGStore",
    "DAGEdge",
    "ExplainCard",
    "DREDashboard",
]


# ==============================================================================
# 13) Demo
# ==============================================================================
def demo() -> None:  # pragma: no cover
    engine = DataRefineryEngine(target_kpi="income")

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "income": (5000 + rng.normal(0, 120, 120)).astype(int),
            "trash": ["err"] * 120,
            "city": rng.choice(["Seoul", "Busan", "Incheon"], size=120),
            "score": rng.normal(0, 1, 120),
            "id": np.arange(120),
        }
    )

    # run a few batches
    for _ in range(3):
        df2 = df.sample(frac=1.0, replace=False, random_state=int(time.time()) % 10000).reset_index(drop=True)
        engine.refine(df2)

    if Flask is not None:
        dash = DREDashboard(engine, port=8080)
        dash.start()
    else:
        print("Flask not installed, skipping dashboard. DAG.DOT preview:")
        print(engine.dag.to_dot(limit=200))


if __name__ == "__main__":  # pragma: no cover
    demo()
