# ==============================================================================
# File: resonetics_researchnote_lineage_v1.py
# Project: Resonetics (Researchnote Automation)
# Version: 1.0
# License: MIT (You can change/remove as you prefer)
#
# Purpose
# -------
# - Automatic experiment tagging (branch A/B, experiment tag, ablation conditions)
# - Lineage logging (nodes/edges) + Mermaid export for GitHub README
# - Drop-in additions for your Idea Evolution / Paradox judge loop
#
# How to use
# ----------
# 1) Create ctx = ExperimentContext(...)
# 2) Pass ctx into IdeaEvolutionEngine and call engine.step(..., ctx=ctx)
# 3) Export: engine.export_mermaid(), engine.export_experiment_log_md()
#
# Notes
# -----
# - This file is "framework glue": it doesn't assume your whole codebase.
# - It’s designed to be easy to splice into your existing engine.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time
import uuid


# ------------------------------------------------------------------------------
# 1) Experiment Context (branch / tag / ablation)
# ------------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """
    A switchboard describing which features are ON/OFF for this run.
    Keep keys stable: they become part of lineage_tag and experiment logs.
    """
    flags: Dict[str, bool] = field(default_factory=dict)

    def to_short_string(self, max_items: int = 6) -> str:
        """
        Compact encoding for tags: +feat/-feat
        Example: +paradox -ether +humility
        """
        items = []
        for k in sorted(self.flags.keys()):
            sign = "+" if self.flags[k] else "-"
            items.append(f"{sign}{k}")
        if len(items) > max_items:
            items = items[:max_items] + ["..."]
        return " ".join(items) if items else "no_ablation"

    def to_dict(self) -> Dict[str, bool]:
        return dict(self.flags)


@dataclass
class ExperimentContext:
    """
    Run-level metadata injected into every Idea child node automatically.
    """
    experiment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    experiment_tag: str = "exp"
    branch: str = "A"  # "A" or "B" (or any label you want)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    notes: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def lineage_tag(self) -> str:
        """
        Single canonical string to store on Idea.lineage_tag
        GitHub-friendly and readable in Mermaid.
        """
        abl = self.ablation.to_short_string()
        return f"{self.experiment_tag}:{self.experiment_id} | branch:{self.branch} | abl:{abl}"

    def add_note(self, note: str) -> None:
        self.notes.append(note)


# ------------------------------------------------------------------------------
# 2) Idea + Lineage Graph
# ------------------------------------------------------------------------------

@dataclass
class Idea:
    text: str
    version: int = 0
    notes: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    idea_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    parent_id: Optional[str] = None
    lineage_tag: str = "root"


@dataclass
class Verdict:
    type: str   # creative_tension | bubble | collapse | stable
    energy: float
    action: str
    reason: str


@dataclass
class ParadoxState:
    tension: float
    coherence: float
    pressure_response: float
    self_protecting: bool


@dataclass
class LineageNode:
    idea_id: str
    version: int
    lineage_tag: str
    created_at: float
    verdict: str
    energy: float
    action: str
    summary: str


@dataclass
class LineageEdge:
    parent_id: str
    child_id: str
    label: str
    timestamp: float = field(default_factory=time.time)


class LineageGraph:
    def __init__(self):
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []

    def add_node(self, node: LineageNode) -> None:
        self.nodes[node.idea_id] = node

    def add_edge(self, parent_id: str, child_id: str, label: str) -> None:
        self.edges.append(LineageEdge(parent_id=parent_id, child_id=child_id, label=label))

    def to_mermaid(self, direction: str = "TD") -> str:
        lines = [f"flowchart {direction}"]
        for nid, n in self.nodes.items():
            safe_summary = (n.summary or "").replace('"', '\\"')
            safe_tag = (n.lineage_tag or "").replace('"', '\\"')
            label = (
                f'{nid}["v{n.version} | {n.verdict} | e={n.energy:.2f}'
                f'\\n{safe_summary}'
                f'\\n{safe_tag}"]'
            )
            lines.append(f"  {label}")

        for e in self.edges:
            safe_label = e.label.replace('"', '\\"')
            lines.append(f'  {e.parent_id} -->|"{safe_label}"| {e.child_id}')
        return "\n".join(lines)

    def stats(self) -> Dict[str, int]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}


# ------------------------------------------------------------------------------
# 3) Research Log Export (Markdown)
# ------------------------------------------------------------------------------

class ResearchLogger:
    """
    Produces a reviewer-friendly markdown log.
    """
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def log_step(self, *, ctx: ExperimentContext, idea_in: Idea, idea_out: Idea,
                 verdict: Verdict, ledger: Dict[str, float], step_idx: int) -> None:
        self.rows.append({
            "step": step_idx,
            "exp": ctx.experiment_id,
            "tag": ctx.experiment_tag,
            "branch": ctx.branch,
            "ablation": ctx.ablation.to_short_string(),
            "idea_in": f"{idea_in.idea_id}@v{idea_in.version}",
            "idea_out": f"{idea_out.idea_id}@v{idea_out.version}",
            "verdict": verdict.type,
            "energy": round(float(verdict.energy), 3),
            "action": verdict.action,
            "reason": verdict.reason,
            "ledger_creative": round(float(ledger.get("creative_energy", 0.0)), 3),
            "ledger_leaked": round(float(ledger.get("leaked_energy", 0.0)), 3),
            "ledger_blocked": round(float(ledger.get("blocked_energy", 0.0)), 3),
        })

    def to_markdown(self, title: str = "EXPERIMENT LOG") -> str:
        lines: List[str] = []
        lines.append(f"# {title}")
        lines.append("")
        lines.append("This log is auto-generated. Each row is one evolution step.")
        lines.append("")
        lines.append("## Run metadata")
        if self.rows:
            r0 = self.rows[0]
            lines.append(f"- Experiment: `{r0['tag']}:{r0['exp']}`")
            lines.append(f"- Branch: `{r0['branch']}`")
            lines.append(f"- Ablation: `{r0['ablation']}`")
        lines.append("")
        lines.append("## Steps")
        lines.append("")
        header = [
            "step", "idea_in", "idea_out", "verdict", "energy",
            "action", "ledger_creative", "ledger_leaked", "ledger_blocked"
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for r in self.rows:
            row = [
                str(r["step"]),
                f"`{r['idea_in']}`",
                f"`{r['idea_out']}`",
                f"`{r['verdict']}`",
                str(r["energy"]),
                f"`{r['action']}`",
                str(r["ledger_creative"]),
                str(r["ledger_leaked"]),
                str(r["ledger_blocked"]),
            ]
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("## Notes / Reasons")
        lines.append("")
        for r in self.rows:
            lines.append(f"- Step {r['step']} `{r['verdict']}`: {r['reason']}")
        lines.append("")
        return "\n".join(lines)


# ------------------------------------------------------------------------------
# 4) Minimal Engine Skeleton (plug your judge/evolve logic in here)
# ------------------------------------------------------------------------------

class EnergyLedger:
    def __init__(self):
        self.creative_energy: float = 0.0
        self.leaked_energy: float = 0.0
        self.blocked_energy: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "creative_energy": self.creative_energy,
            "leaked_energy": self.leaked_energy,
            "blocked_energy": self.blocked_energy,
        }


class ParadoxJudge:
    """
    Placeholder: plug in your actual rules.
    """
    def evaluate(self, state: ParadoxState) -> Verdict:
        if state.self_protecting and state.coherence < 0.35:
            return Verdict("collapse", energy=max(0.1, state.tension), action="QUARANTINE",
                           reason="Low coherence + self-protecting indicates collapse risk.")
        if state.tension >= 0.65 and state.coherence >= 0.55 and state.pressure_response >= 0.60:
            energy = (state.tension + state.coherence + state.pressure_response) / 3
            return Verdict("creative_tension", energy=min(1.0, energy), action="PRESERVE_AND_FEED",
                           reason="Sustained tension across layers generates creative energy.")
        if state.tension >= 0.75 and state.coherence < 0.45:
            return Verdict("bubble", energy=min(1.0, state.tension), action="DEMAND_EVIDENCE",
                           reason="High tension with weak coherence suggests hype/bubble.")
        return Verdict("stable", energy=min(1.0, (state.coherence + state.pressure_response) / 2),
                       action="LOG_AND_MONITOR", reason="Stable regime; monitor drift.")


class EvolutionOps:
    """
    Placeholder: call your real evolve() rules.
    """
    def evolve(self, idea: Idea, verdict: Verdict, ctx: ExperimentContext) -> Idea:
        new_text = idea.text
        new_notes = list(idea.notes)
        new_notes.append(f"Evolved under {ctx.lineage_tag()} due to {verdict.type} (e={verdict.energy:.2f}).")

        if verdict.type == "creative_tension":
            new_text = (
                new_text.strip()
                + "\n\n[Auto-Evolve]\n- Refine scope\n- Add ablation plan\n- Add falsifiable tests"
            )
        elif verdict.type == "bubble":
            new_text = (
                new_text.strip()
                + "\n\n[Bubble Mitigation]\n- Narrow claim\n- Add baselines + replication protocol"
            )
        elif verdict.type == "collapse":
            new_notes.append("Quarantined: requires axiom reset before next iteration.")

        return Idea(
            text=new_text,
            version=idea.version + 1,
            notes=new_notes,
            parent_id=idea.idea_id,
            lineage_tag=ctx.lineage_tag(),
        )


class IdeaEvolutionEngine:
    def __init__(self, evolve_threshold: float = 1.25, decay: float = 0.02):
        self.judge = ParadoxJudge()
        self.ops = EvolutionOps()
        self.ledger = EnergyLedger()
        self.lineage = LineageGraph()
        self.logger = ResearchLogger()
        self.threshold = float(evolve_threshold)
        self.decay = float(decay)
        self.step_index = 0

    def _decay_energy(self):
        self.ledger.creative_energy = max(0.0, self.ledger.creative_energy * (1.0 - self.decay))

    def _log_lineage_node(self, idea: Idea, verdict: Verdict):
        preview = idea.text.strip().replace("\n", " ")
        preview = preview[:80] + ("..." if len(preview) > 80 else "")
        self.lineage.add_node(LineageNode(
            idea_id=idea.idea_id,
            version=idea.version,
            lineage_tag=idea.lineage_tag,
            created_at=idea.created_at,
            verdict=verdict.type,
            energy=float(verdict.energy),
            action=verdict.action,
            summary=preview
        ))
        if idea.parent_id:
            self.lineage.add_edge(
                parent_id=idea.parent_id,
                child_id=idea.idea_id,
                label=f"{verdict.type} / e={verdict.energy:.2f}"
            )

    def step(self, idea: Idea, state: ParadoxState, ctx: ExperimentContext) -> Tuple[Idea, Verdict, Dict[str, float]]:
        if idea.lineage_tag == "root":
            idea.lineage_tag = ctx.lineage_tag()

        verdict = self.judge.evaluate(state)

        if verdict.type == "creative_tension":
            self.ledger.creative_energy += verdict.energy
        elif verdict.type == "bubble":
            self.ledger.leaked_energy += verdict.energy * 0.6
            self.ledger.creative_energy += verdict.energy * 0.2
        elif verdict.type == "collapse":
            self.ledger.blocked_energy += verdict.energy
        else:
            self.ledger.creative_energy += verdict.energy * 0.15

        self._decay_energy()

        self._log_lineage_node(idea, verdict)

        evolved = idea
        if verdict.type == "creative_tension" and self.ledger.creative_energy >= self.threshold:
            evolved = self.ops.evolve(idea, verdict, ctx)
            self.ledger.creative_energy -= self.threshold
            self._log_lineage_node(evolved, verdict)

        self.logger.log_step(
            ctx=ctx,
            idea_in=idea,
            idea_out=evolved,
            verdict=verdict,
            ledger=self.ledger.as_dict(),
            step_idx=self.step_index
        )
        self.step_index += 1

        return evolved, verdict, self.ledger.as_dict()

    def export_mermaid(self) -> str:
        return self.lineage.to_mermaid()

    def export_experiment_log_md(self) -> str:
        return self.logger.to_markdown()


if __name__ == "__main__":
    idea0 = Idea(text="AI can surpass humans in some tasks, but generalization is fragile.")
    engine = IdeaEvolutionEngine(evolve_threshold=1.10, decay=0.01)

    ctxA = ExperimentContext(
        experiment_tag="via-negativa",
        branch="A",
        ablation=AblationConfig(flags={
            "unlearning": True,
            "paradox": True,
            "humility": True,
            "unity": True,
        }),
        notes=["Full system ON"]
    )

    ctxB = ExperimentContext(
        experiment_tag="via-negativa",
        branch="B",
        ablation=AblationConfig(flags={
            "unlearning": True,
            "paradox": False,
            "humility": True,
            "unity": True,
        }),
        notes=["Paradox OFF (ablation)"]
    )

    ideaA = idea0
    for s in [
        ParadoxState(tension=0.72, coherence=0.85, pressure_response=0.88, self_protecting=False),
        ParadoxState(tension=0.80, coherence=0.70, pressure_response=0.75, self_protecting=False),
        ParadoxState(tension=0.78, coherence=0.62, pressure_response=0.72, self_protecting=False),
    ]:
        ideaA, v, _ = engine.step(ideaA, s, ctxA)

    ideaB = idea0
    for s in [
        ParadoxState(tension=0.85, coherence=0.42, pressure_response=0.40, self_protecting=False),
        ParadoxState(tension=0.78, coherence=0.28, pressure_response=0.35, self_protecting=True),
    ]:
        ideaB, v, _ = engine.step(ideaB, s, ctxB)

    print("\n--- Mermaid ---")
    print("```mermaid")
    print(engine.export_mermaid())
    print("```")

    print("\n--- Experiment Log ---")
    print(engine.export_experiment_log_md())
