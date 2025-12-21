# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: resonetics_via_negativa_v1_2_weighted.py
# Project: Resonetics – Via Negativa
# Version: 1.2 (Confidence-Weighted Paradox Engine)
# ==============================================================================

from dataclasses import dataclass
from enum import Enum
import math


# ==============================================================================
# Paradox Classification Types
# ==============================================================================

class ParadoxType(str, Enum):
    CREATIVE_TENSION = "creative_tension"
    BUBBLE = "bubble"
    COLLAPSE = "collapse"


class ParadoxAction(str, Enum):
    PRESERVE_AND_FEED = "PRESERVE_AND_FEED"
    STRESS_TEST = "STRESS_TEST"
    CONTAIN_AND_PRUNE = "CONTAIN_AND_PRUNE"


# ==============================================================================
# Metrics
# ==============================================================================

@dataclass
class ParadoxMetrics:
    tension: float            # intrinsic contradiction energy (0~1)
    coherence: float          # logical / narrative coherence (0~1)
    pressure_response: float  # robustness under critique (0~1)
    self_protecting: bool     # defensive rationalization flag
    confidence: float         # epistemic confidence (0~1)


@dataclass
class ParadoxVerdict:
    type: ParadoxType
    energy: float
    action: ParadoxAction
    reason: str
    metrics: ParadoxMetrics


# ==============================================================================
# Paradox Engine (Confidence-Weighted)
# ==============================================================================

class ParadoxEngine:
    """
    Paradox Engine v1.2

    Core idea:
    - Not all contradictions are equal
    - Creative paradoxes survive pressure WITHOUT self-protection
    - Bubbles feel strong but collapse under stress
    - Collapses protect themselves and lose coherence

    Confidence:
    - Derived epistemic trust
    - Low confidence amplifies bubble risk
    """

    # ------------------------------
    # Energy Calculation
    # ------------------------------
    @staticmethod
    def compute_energy(m: ParadoxMetrics) -> float:
        """
        Unified energy equation (0~1)

        Weights are intentional:
        - tension + coherence are dominant
        - pressure_response ensures survivability
        - confidence moderates hallucination / overclaim
        """

        energy = (
            0.35 * m.tension +
            0.35 * m.coherence +
            0.2  * m.pressure_response +
            0.1  * m.confidence
        )

        # defensive rationalization penalty
        if m.self_protecting:
            energy -= 0.2

        return max(0.0, min(1.0, energy))

    # ------------------------------
    # Classification Logic
    # ------------------------------
    @staticmethod
    def classify(m: ParadoxMetrics) -> ParadoxVerdict:
        energy = ParadoxEngine.compute_energy(m)

        # --- COLLAPSE ---
        if (
            m.self_protecting and
            (m.coherence < 0.4 or m.pressure_response < 0.4)
        ):
            return ParadoxVerdict(
                type=ParadoxType.COLLAPSE,
                energy=energy,
                action=ParadoxAction.CONTAIN_AND_PRUNE,
                reason="Defensive self-protection with low coherence or pressure resilience",
                metrics=m
            )

        # --- BUBBLE ---
        if (
            m.tension > 0.7 and
            (m.coherence < 0.6 or m.pressure_response < 0.6) and
            m.confidence < 0.6
        ):
            return ParadoxVerdict(
                type=ParadoxType.BUBBLE,
                energy=energy,
                action=ParadoxAction.STRESS_TEST,
                reason="High tension without sufficient coherence, pressure resilience, or confidence",
                metrics=m
            )

        # --- CREATIVE TENSION ---
        if (
            m.tension >= 0.65 and
            m.coherence >= 0.65 and
            m.pressure_response >= 0.65 and
            not m.self_protecting
        ):
            return ParadoxVerdict(
                type=ParadoxType.CREATIVE_TENSION,
                energy=energy,
                action=ParadoxAction.PRESERVE_AND_FEED,
                reason="Sustained contradiction with coherence, resilience, and epistemic confidence",
                metrics=m
            )

        # --- Default conservative handling ---
        return ParadoxVerdict(
            type=ParadoxType.BUBBLE,
            energy=energy,
            action=ParadoxAction.STRESS_TEST,
            reason="Ambiguous paradox – treated conservatively as bubble pending validation",
            metrics=m
        )


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    example = ParadoxMetrics(
        tension=0.72,
        coherence=0.85,
        pressure_response=0.88,
        self_protecting=False,
        confidence=0.78
    )

    verdict = ParadoxEngine.classify(example)

    print("\n[Paradox Verdict]")
    print("Type   :", verdict.type.value)
    print("Energy :", round(verdict.energy, 3))
    print("Action :", verdict.action.value)
    print("Reason :", verdict.reason)
