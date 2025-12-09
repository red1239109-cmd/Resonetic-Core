# ==============================================================================
# Project Name : Paradox Refinement Engine (Resonetics Module)
# Description  : Text Refinement Engine with 8-Layer Sovereign Logic & Meta-Rules
# Author       : red1239109-cmd
# Contact      : red1239109@gmail.com
# Version      : 1.0 Alpha
#
# [LICENSE NOTICE]
# This project follows a DUAL-LICENSE model:
#
# 1. Open Source (AGPL-3.0)
#    - You are free to use, modify, and distribute this code.
#    - CONDITION: If you use this code (even over a network), you MUST open-source 
#      your entire project under AGPL-3.0.
#
# 2. Commercial License
#    - If you want to keep your source code private (Closed Source), 
#      you must purchase a Commercial License.
#    - Contact the author for pricing and details.
#
# Copyright (c) 2025 Resonetics Project. All rights reserved.
# ==============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Tuple
import re
import textwrap
import math


# =========================================================
# 1. Data Classes
# =========================================================

@dataclass
class Critique:
    """Structure to hold the self-critique results returned by the LLM."""
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    severity: str = "UNKNOWN"   # "HIGH" / "MEDIUM" / "LOW" / "UNKNOWN"
    raw: str = ""               # Raw response text (for debugging)
    improved_text: str = ""     # Extracted improved text


@dataclass
class RefinementResult:
    """Final result of the refine() process."""
    original_text: str
    final_text: str
    iterations: int
    stopped_reason: str
    history: List[str] = field(default_factory=list)
    critique_history: List[Critique] = field(default_factory=list)
    meta_scores: List[float] = field(default_factory=list)


# =========================================================
# 2. Meta-Rule Layer (SovereignLoss logic translated for Text)
# =========================================================

@dataclass
class MetaSignal:
    """Meta-signals extracted from a single iteration."""
    similarity: float       # L1: Closeness to the goal (previous text)
    delta: float            # Change magnitude = 1 - similarity
    structure_score: float  # L2/L5: Rule of 3, logical structure, formatting
    stability: float        # L7: Self-Consistency (Is the vibration reducing?)


class MetaRuleLayer:
    """
    Meta-Rule Layer for 'Convergence Judgment' in the Paradox Engine.
    - Simplifies the 8 layers of SovereignLoss into 4 key textual signals.
    """

    def __init__(self) -> None:
        self.prev_delta: Optional[float] = None

    # ---- Structure Score: Rule of 3, Numbering, Bullets ----
    def _structure_score(self, text: str) -> float:
        """
        Evaluates structural integrity.
        - '1.', '2.', '3.' numbered lists
        - Bullet points
        - High score if the 'Rule of 3' is observed.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return 0.0

        numbered = [ln for ln in lines if re.match(r"^\d+\.", ln)]
        bullets = [ln for ln in lines if ln.startswith(("-", "*", "+"))]

        score = 0.0

        # If numbered list has 3 or more items, it's considered well-structured (Rule of 3)
        if len(numbered) >= 3:
            score += 0.6
        elif len(numbered) >= 2:
            score += 0.4

        # Bullets act as a bonus
        if len(bullets) >= 3:
            score += 0.3
        elif len(bullets) >= 1:
            score += 0.1

        # Normalize score (Cap at 1.0)
        score = min(score, 1.0)
        return score

    def evaluate(
        self,
        prev_text: str,
        new_text: str,
        critique: Critique,
        iter_idx: int,
        max_iterations: int,
        similarity_fn: Callable[[str, str], float],
    ) -> Tuple[MetaSignal, bool]:
        """
        Evaluates a single step:
        - Calculates MetaSignals.
        - Decides whether to stop based on the signals (Sovereign Logic).
        """

        sim = similarity_fn(prev_text, new_text)
        delta = 1.0 - sim
        struct = self._structure_score(new_text)

        if self.prev_delta is None:
            stability = 0.5  # Neutral for the first iteration
        else:
            # Higher stability if the magnitude of change (delta) is decreasing (damping vibration)
            stability = 1.0 - min(abs(delta - self.prev_delta), 1.0)

        self.prev_delta = delta

        signal = MetaSignal(
            similarity=sim,
            delta=delta,
            structure_score=struct,
            stability=stability,
        )

        # ----- Synthesizing the Sovereign Score -----
        # L1: Similarity (Gravity towards target)
        # L5: Structure Score (Rule of 3 / Quantization)
        # L7: Stability (Self-Consistency)
        # L8: "Humility" is reflected via Severity weights
        
        sev_weight = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3, "UNKNOWN": 0.5}
        w_sev = sev_weight.get(critique.severity, 0.5)

        # Base Meta-Score (0.0 ~ 1.0)
        base_score = (
            0.45 * signal.similarity +
            0.35 * signal.structure_score +
            0.20 * signal.stability
        )

        # Penalize convergence if severity is HIGH (L8 Humility)
        meta_score = max(0.0, base_score * (1.0 - 0.5 * w_sev))

        # Dynamic Thresholding (Simulated Annealing)
        # Be loose at first (Exploration), strict at the end (Exploitation)
        progress = (iter_idx + 1) / max(1, max_iterations)
        dynamic_threshold = 0.65 + 0.25 * progress  # Starts at 0.65 -> Ends near 0.9

        meta_should_stop = meta_score >= dynamic_threshold

        return signal, meta_should_stop


# =========================================================
# 3. Paradox Refinement Engine (Integration)
# =========================================================

# Words that indicate strong logical boundaries.
# If these change, it suggests a "Logic Flip" (L3 Violation).
LOGIC_WORDS = {"all", "every", "must", "never", "some", "most", "possible", "always"}


class ParadoxRefinementEngine:
    """
    Engine that runs the Loop: Self-Critique -> Refine -> Convergence Check.
    - llm_fn: Callback function (prompt:str) -> str
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        max_iterations: int = 3,
        verbose: bool = False,
        enable_cache: bool = True,
    ) -> None:
        self.llm_fn = llm_fn
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.enable_cache = enable_cache
        self._cache: Dict[str, str] = {}
        self.meta_layer = MetaRuleLayer()

    # -----------------------------------------------------
    # LLM Call Wrapper (Caching included)
    # -----------------------------------------------------
    def _call_llm(self, prompt: str) -> str:
        if self.enable_cache and prompt in self._cache:
            return self._cache[prompt]

        try:
            resp = self.llm_fn(prompt)
        except Exception as e:
            # Simple fallback for errors
            return f"- WEAKNESSES:\n  - LLM_ERROR: {e}\n- SUGGESTIONS:\n  - Keep original.\n- SEVERITY: HIGH\n- IMPROVED_TEXT:\n{prompt}"

        if self.enable_cache:
            self._cache[prompt] = resp
        return resp

    # -----------------------------------------------------
    # Jaccard Similarity (Simple metric)
    # -----------------------------------------------------
    def _similarity(self, a: str, b: str) -> float:
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union

    # -----------------------------------------------------
    # Response Parsing
    # -----------------------------------------------------
    def _match_header(self, line: str, keyword: str) -> bool:
        # Allows variations like "- WEAKNESSES:", "WEAKNESSES :", "weaknesses:"
        return re.match(rf"-?\s*{re.escape(keyword)}\s*:?", line.strip(), re.IGNORECASE) is not None

    def _safe_parse_critique(self, raw: str) -> Critique:
        lines = raw.splitlines()
        weaknesses: List[str] = []
        suggestions: List[str] = []
        severity = "UNKNOWN"
        improved_lines: List[str] = []

        current = None

        for ln in lines:
            if self._match_header(ln, "WEAKNESSES"):
                current = "W"
                continue
            if self._match_header(ln, "SUGGESTIONS"):
                current = "S"
                continue
            if self._match_header(ln, "SEVERITY"):
                # Matches "SEVERITY: HIGH"
                m = re.search(r":\s*(\w+)", ln)
                if m:
                    severity = m.group(1).upper()
                current = None
                continue
            if self._match_header(ln, "IMPROVED_TEXT"):
                current = "I"
                continue

            if current == "W":
                if ln.strip().startswith(("-", "*")):
                    weaknesses.append(ln.strip()[1:].strip())
            elif current == "S":
                if ln.strip().startswith(("-", "*")):
                    suggestions.append(ln.strip()[1:].strip())
            elif current == "I":
                improved_lines.append(ln)

        crit = Critique(
            weaknesses=weaknesses,
            suggestions=suggestions,
            severity=severity,
            raw=raw,
        )
        crit.improved_text = "\n".join(improved_lines).strip()
        return crit

    # -----------------------------------------------------
    # Logic Flip Detection (L3: Phase Boundary)
    # -----------------------------------------------------
    def _logic_flip_detected(self, prev: str, new: str) -> bool:
        prev_lower = prev.lower()
        new_lower = new.lower()
        for w in LOGIC_WORDS:
            # If a key logic word disappears, it's a dangerous flip
            if (w in prev_lower) and (w not in new_lower):
                return True
        return False

    # -----------------------------------------------------
    # Convergence Check: Integration of Meta-Rules
    # -----------------------------------------------------
    def _has_converged(
        self,
        prev_text: str,
        new_text: str,
        critique: Critique,
        iter_idx: int,
    ) -> Tuple[bool, float]:
        """
        Returns: (should_stop, meta_score)
        """

        sim = self._similarity(prev_text, new_text)

        # 1) Absolute Barrier: If logic flips, never stop. (L3)
        if self._logic_flip_detected(prev_text, new_text):
            if self.verbose:
                print("  [DEBUG] Logic flip detected (L3 Violation) -> Keep refining")
            return False, 0.0

        # 2) Evaluate MetaRuleLayer
        signal, meta_stop = self.meta_layer.evaluate(
            prev_text=prev_text,
            new_text=new_text,
            critique=critique,
            iter_idx=iter_idx,
            max_iterations=self.max_iterations,
            similarity_fn=self._similarity,
        )

        meta_score = (
            0.45 * signal.similarity +
            0.35 * signal.structure_score +
            0.20 * signal.stability
        )

        if self.verbose:
            print(
                f"  [META] Sim={signal.similarity:.3f} | "
                f"Struct={signal.structure_score:.3f} | "
                f"Stab={signal.stability:.3f} | "
                f"Stop?={meta_stop}"
            )

        # 3) Safety Fallback: If Severity is HIGH, force continue unless max iter reached
        if critique.severity.upper() == "HIGH" and iter_idx + 1 < self.max_iterations:
            return False, meta_score

        return meta_stop, meta_score

    # -----------------------------------------------------
    # Prompt Building (English Template)
    # -----------------------------------------------------
    def _build_prompt(self, text: str, critique_history: List[Critique]) -> str:
        """
        Injects the 'Rule of Three' hint into the prompt.
        """
        past_weaknesses: List[str] = []
        for c in critique_history:
            past_weaknesses.extend(c.weaknesses)

        past_block = "\n".join(f"- {w}" for w in past_weaknesses[-6:]) or "None"

        tpl = """
You are a critical editor designed to refine thought processes.
Analyze the following text and respond in this STRICT format:

- WEAKNESSES:
  - (List specific logical flaws)
- SUGGESTIONS:
  - (Actionable advice)
- SEVERITY: LOW | MEDIUM | HIGH
- IMPROVED_TEXT:
...

Now apply a **Three-Step Structure** to your critique:
1. Logic: Identify logical contradictions.
2. Refute: Check for counter-examples or missed edge cases.
3. Define: Refine definitions and boundaries.

Previous weaknesses identified so far:
{past_block}

=== TEXT START ===
{text}
=== TEXT END ===
"""
        return textwrap.dedent(tpl).format(text=text, past_block=past_block)

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------
    def refine(self, text: str) -> RefinementResult:
        """
        Refines the text iteratively until 'Meta-Convergence' is reached.
        """
        current = text
        history = [text]
        critiques: List[Critique] = []
        meta_scores: List[float] = []

        stopped_reason = "MAX_ITERATIONS"

        for i in range(self.max_iterations):
            if self.verbose:
                print(f"\n[ITER {i+1}/{self.max_iterations}] ----------------------")

            prompt = self._build_prompt(current, critiques)
            raw = self._call_llm(prompt)
            crit = self._safe_parse_critique(raw)

            improved = getattr(crit, "improved_text", "").strip()
            if not improved:
                # If no improvement returned, stop safely
                stopped_reason = "NO_IMPROVEMENT_RETURNED"
                break

            # Convergence Check
            should_stop, meta_score = self._has_converged(
                prev_text=current,
                new_text=improved,
                critique=crit,
                iter_idx=i,
            )
            meta_scores.append(meta_score)

            history.append(improved)
            critiques.append(crit)
            current = improved

            if should_stop:
                stopped_reason = "CONVERGED"
                break

        return RefinementResult(
            original_text=text,
            final_text=current,
            iterations=len(history) - 1,
            stopped_reason=stopped_reason,
            history=history,
            critique_history=critiques,
            meta_scores=meta_scores,
        )


# =========================================================
# 4. Simulation Driver (Mock LLM for Testing)
# =========================================================

import random

def mock_llm_simulation(prompt: str) -> str:
    """Mock LLM to simulate gradual refinement for testing purposes."""
    try:
        input_text = prompt.split("=== TEXT START ===")[1].split("=== TEXT END ===")[0].strip()
    except:
        input_text = "Error parsing text"

    # Simulation Logic:
    # 1. If missing numbers, add them.
    if "1." not in input_text:
        improved = "1. " + input_text.replace("\n", "\n2. ")
        return f"""
- WEAKNESSES:
  - Lack of structure.
  - Vague definitions.
- SUGGESTIONS:
  - Add numbering.
  - Define terms clearly.
- SEVERITY: HIGH
- IMPROVED_TEXT:
{improved}
"""
    # 2. If missing the Rule of Three, add it.
    if "3." not in input_text:
        improved = input_text + "\n3. Conclusion and Synthesis: Integrating logic and physics."
        return f"""
- WEAKNESSES:
  - Missing conclusion.
  - Only two points found.
- SUGGESTIONS:
  - Apply the rule of three.
- SEVERITY: MEDIUM
- IMPROVED_TEXT:
{improved}
"""
    # 3. Final Polish
    return f"""
- WEAKNESSES:
  - Minor stylistic issues.
- SUGGESTIONS:
  - Polish the tone.
- SEVERITY: LOW
- IMPROVED_TEXT:
{input_text}
"""

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🧩 Paradox Engine (Resonetics) : Simulation Test")
    print(f"{'='*60}\n")

    # 1. Initialize Engine
    engine = ParadoxRefinementEngine(
        llm_fn=mock_llm_simulation,
        max_iterations=5,
        verbose=True
    )

    # 2. Raw Input
    raw_input = """
    AGI is coming. We need to prepare.
    Data is sufficient but logic is missing.
    """

    print(f"📌 [INPUT]:\n{raw_input.strip()}\n")

    # 3. Start Refinement
    result = engine.refine(raw_input)

    # 4. Report
    print(f"\n{'='*60}")
    print(f"🏁 [RESULT] Stopped Reason: {result.stopped_reason}")
    print(f"   Iterations: {result.iterations}")
    print(f"{'='*60}")
    
    print("\n📜 Final Refined Text:")
    print(result.final_text)

    print("\n📊 Meta-Scores History:")
    for i, score in enumerate(result.meta_scores):
        print(f"   Iter {i+1}: {score:.4f}")
