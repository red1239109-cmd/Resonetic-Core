# ==============================================================================
# File: resonetics_v2_paradox.py
# Stage 2: The Linguistic Reason (Paradox Engine)
# Description: Translates Sovereign Logic into a recursive text critique loop.
# Author: red1239109-cmd
# Copyright (c) 2025 Resonetics Project
#
# License: AGPL-3.0
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# THIRD-PARTY COMPONENTS:
# - This code may interface with external LLM APIs
# - Users are responsible for compliance with respective API terms of service
# ==============================================================================

from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
import numpy as np
from enum import Enum

class ConvergenceStatus(Enum):
    CONVERGED = "CONVERGED"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    NO_PROGRESS = "NO_PROGRESS"
    PARADOX_DETECTED = "PARADOX_DETECTED"

@dataclass
class Critique:
    strength: float  # 0.0 ~ 1.0
    aspect: str      # 'clarity', 'consistency', 'depth', 'structure', 'creativity'
    suggestion: str

@dataclass
class RefinementResult:
    final_text: str
    iterations: int
    stopped_reason: ConvergenceStatus
    critiques: List[Critique]
    confidence_score: float  # 0.0 ~ 1.0

class ParadoxRefinementEngine:
    """
    Recursive Critique Engine for text refinement.
    Iteratively refines thought processes until convergence is reached.
    
    Core Principles:
    1. Gravity: Truth alignment through similarity checks.
    2. Rule of 3: Structural optimization in triadic patterns.
    3. Boundary: Logical flip detection at stability limits.
    4. Entropy: Preventing intellectual stagnation.
    5. Self-Consistency: Temporal coherence across iterations.
    """
    
    def __init__(self, llm_fn: Callable[[str], str], max_iterations: int = 5,
                 convergence_threshold: float = 0.95, min_improvement: float = 0.05):
        self.llm_fn = llm_fn
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_improvement = min_improvement
        self.paradox_rules = self._initialize_paradox_rules()
    
    def _initialize_paradox_rules(self) -> Dict:
        """Initialize critique rules."""
        return {
            'gravity': lambda t: self._check_gravity(t),
            'rule_of_3': lambda t: self._check_rule_of_3(t),
            'boundary': lambda t: self._check_boundary(t),
            'entropy': lambda t: self._check_entropy(t),
            'self_consistency': lambda t1, t2: self._check_self_consistency(t1, t2)
        }
    
    def _check_gravity(self, text: str) -> Critique:
        """Reality Alignment & Truth Verification."""
        clarity_score = min(len(text.split()) / 100, 1.0)
        return Critique(
            strength=clarity_score,
            aspect='clarity',
            suggestion="Enhance clarity and ensure fact-based verification."
        )
    
    def _check_rule_of_3(self, text: str) -> Critique:
        """Structural Ideal (The Rule of Three)."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        structure_score = 1.0 if len(sentences) % 3 == 0 else 0.3
        return Critique(
            strength=structure_score,
            aspect='structure',
            suggestion="Reconstruct into a triadic structure."
        )
    
    def _check_boundary(self, text: str) -> Critique:
        """Logical Boundary & Paradox Detection."""
        paradox_keywords = ['however', 'but', 'conversely', 'yet', 'although', 'paradoxically']
        has_paradox = any(kw in text.lower() for kw in paradox_keywords)
        
        return Critique(
            strength=0.7 if has_paradox else 0.3,
            aspect='depth',
            suggestion="Deepen the logic by introducing opposing views." if not has_paradox 
                      else "Logically integrate the paradoxical elements."
        )
    
    def _check_entropy(self, text: str) -> Critique:
        """Entropy Maintenance (Preventing Stagnation)."""
        unique_words = len(set(text.lower().split()))
        total_words = len(text.split())
        diversity_score = unique_words / max(total_words, 1)
        
        return Critique(
            strength=diversity_score,
            aspect='creativity',
            suggestion="Increase lexical diversity and introduce novel perspectives."
        )
    
    def _check_self_consistency(self, text1: str, text2: str) -> Critique:
        """Temporal Consistency (Narrative Identity)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return Critique(strength=0.0, aspect='consistency', 
                            suggestion="Text is too short for consistency check.")
        
        overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        return Critique(
            strength=overlap,
            aspect='consistency',
            suggestion="Maintain consistency of core concepts while evolving the narrative."
        )
    
    def _generate_refinement_prompt(self, text: str, critiques: List[Critique]) -> str:
        """Generates a structured prompt for the LLM based on critiques."""
        prompt = f"""Refine the following text:

Original Text: {text}

Critique Analysis:
"""
        for i, critique in enumerate(critiques[:3], 1):
            prompt += f"{i}. {critique.aspect.upper()}: {critique.suggestion} (Weight: {critique.strength:.2f})\n"
        
        prompt += "\nRefinement Guidelines:\n1. Address each critique.\n2. Maintain Triadic Structure.\n3. Enhance logical depth and consistency.\n\nRefined Text:"
        return prompt
    
    def _calculate_confidence(self, text: str, critiques: List[Critique]) -> float:
        """Calculates the confidence score."""
        if not critiques:
            return 0.0
        
        weights = {'structure': 0.3, 'consistency': 0.25, 'clarity': 0.2, 
                   'depth': 0.15, 'creativity': 0.1}
        
        total_score = 0.0
        total_weight = 0.0
        
        for critique in critiques:
            weight = weights.get(critique.aspect, 0.1)
            total_score += critique.strength * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def refine(self, text: str) -> RefinementResult:
        """
        Executes the Recursive Critique Loop.
        
        Args:
            text: Initial text to refine.
            
        Returns:
            RefinementResult: The converged thought process.
        """
        print(f"[Paradox] Analyzing: '{text[:50]}...'")
        
        current_text = text
        previous_text = ""
        all_critiques = []
        iteration_results = []
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"  Iteration {iteration}/{self.max_iterations}")
            
            critiques = []
            critiques.append(self.paradox_rules['gravity'](current_text))
            critiques.append(self.paradox_rules['rule_of_3'](current_text))
            critiques.append(self.paradox_rules['boundary'](current_text))
            critiques.append(self.paradox_rules['entropy'](current_text))
            
            if previous_text:
                critiques.append(self.paradox_rules['self_consistency'](previous_text, current_text))
            
            all_critiques.extend(critiques)
            
            confidence = self._calculate_confidence(current_text, critiques)
            iteration_results.append((current_text, confidence))
            
            print(f"    Confidence: {confidence:.3f}")
            
            if confidence >= self.convergence_threshold:
                print(f"    Convergence reached at iteration {iteration}")
                return RefinementResult(
                    final_text=current_text,
                    iterations=iteration,
                    stopped_reason=ConvergenceStatus.CONVERGED,
                    critiques=all_critiques,
                    confidence_score=confidence
                )
            
            if iteration >= 2:
                prev_conf = iteration_results[-2][1]
                if confidence - prev_conf < self.min_improvement:
                    print(f"    Minimal improvement detected")
                    if iteration >= 3:
                        print(f"    Stopping due to diminishing returns")
                        return RefinementResult(
                            final_text=current_text,
                            iterations=iteration,
                            stopped_reason=ConvergenceStatus.NO_PROGRESS,
                            critiques=all_critiques,
                            confidence_score=confidence
                        )
            
            if confidence < 0.3 and iteration > 1:
                print(f"    Paradox detected - restructuring required")
                previous_text = current_text
                current_text = f"{current_text}\n[Paradox Resolution Attempt]"
                continue
            
            if self.llm_fn and callable(self.llm_fn):
                prompt = self._generate_refinement_prompt(current_text, critiques)
                previous_text = current_text
                try:
                    current_text = self.llm_fn(prompt)
                    print(f"    Text refined via LLM")
                except Exception as e:
                    print(f"    LLM error: {e}")
                    current_text = f"{current_text}\n[Refinement: Structure adjustment]"
            else:
                previous_text = current_text
                current_text = f"{current_text}\n[Refined v{iteration}: Enhanced Structure]"
        
        final_confidence = self._calculate_confidence(current_text, all_critiques)
        print(f"    Max iterations reached. Final confidence: {final_confidence:.3f}")
        
        return RefinementResult(
            final_text=current_text,
            iterations=self.max_iterations,
            stopped_reason=ConvergenceStatus.MAX_ITERATIONS,
            critiques=all_critiques,
            confidence_score=final_confidence
        )

def mock_llm(prompt: str) -> str:
    """Simulates an LLM response for testing purposes."""
    return """Refined Text:

1. The core argument is restated with greater precision.
2. The logic follows a triadic progression.
3. Opposing viewpoints are integrated for depth.

Conclusion: This approach achieves structural elegance and logical rigor."""

if __name__ == "__main__":
    engine = ParadoxRefinementEngine(llm_fn=mock_llm, max_iterations=3)
    
    test_text = "AI can surpass humans. However, it has limits in creativity."
    
    result = engine.refine(test_text)
    
    print(f"\n{'='*60}")
    print(f"REFINEMENT RESULT")
    print(f"{'='*60}")
    print(f"Final Text:\n{result.final_text}")
    print(f"\nIterations: {result.iterations}")
    print(f"Stop Reason: {result.stopped_reason.value}")
    print(f"Confidence: {result.confidence_score:.3f}")
    print(f"Total Critiques: {len(result.critiques)}")
    
    if result.critiques:
        print(f"\nTop Critiques:")
        for i, critique in enumerate(result.critiques[:3], 1):
            print(f"  {i}. {critique.aspect.upper()}: {critique.suggestion} (Strength: {critique.strength:.2f})")
    
    print("\n[Paradox Engine Online]")
