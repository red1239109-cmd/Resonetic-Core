# ==============================================================================
# File: resonetics_core_v6_4.py
# Project: Resonetics Auditor - Philosophical Code Analysis Engine
# Version: 6.4 (Integrated Survival Edition)
# Author: Resonetics Lab
# License: AGPL-3.0
# ==============================================================================

"""
RESONETICS CORE v6.4
====================
"The Three Philosophical Laws of Code Survival"

This tool embodies the Resonetics philosophy by analyzing code through 
three fundamental principles derived from ancient wisdom:

1. STRUCTURAL HARMONY (Plato): Code should have mathematical elegance
2. ADAPTIVE FLOW (Heraclitus): Systems must evolve with their environment  
3. HUMBLE RESILIENCE (Socrates): True strength acknowledges limitations

The auditor doesn't just analyze syntax—it evaluates how well code
embodies these philosophical survival principles.
"""

import ast
import math
import statistics as stats
import sys
import io
import os
import json
from typing import Dict, List, Optional, Union
from datetime import datetime

print(f"\n{'='*70}")
print(f"🧠 RESONETICS AUDITOR v6.4 - PHILOSOPHICAL CODE ANALYSIS")
print(f"{'='*70}\n")

# ==============================================================================
# 1. PHILOSOPHICAL FOUNDATIONS: SURVIVAL MECHANISMS
# ==============================================================================

class PhilosophicalPrinciples:
    """
    THE THREE CORE SURVIVAL PRINCIPLES:
    
    1. STRUCTURAL HARMONY (Plato's Forms)
       - Code should align with mathematical ideals
       - Functions have optimal lengths based on context
       - Complexity follows natural patterns, not arbitrary rules
    
    2. ADAPTIVE FLOW (Heraclitus' Panta Rhei)
       - Systems must adapt to their environment
       - Platform differences are embraced, not fought
       - Missing dependencies are gracefully handled
    
    3. HUMBLE RESILIENCE (Socratic Wisdom)
       - Error handling acknowledges system fragility
       - Self-awareness: tool can analyze itself
       - Mercy Rule: Even imperfect code has potential
    """
    
    @staticmethod
    def ensure_survival() -> None:
        """
        [PRINCIPLE 2: ADAPTIVE FLOW]
        Adapt to hostile environments before attempting analysis.
        
        This is the first line of defense—the tool must survive
        its own execution environment before analyzing others.
        """
        # Windows UTF-8 adaptation
        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except AttributeError:
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, 
                    encoding='utf-8',
                    errors='replace'  # Graceful degradation
                )
        
        # Timezone awareness for reproducibility
        os.environ['TZ'] = 'UTC'
    
    @staticmethod
    def contextual_ideal(lengths: List[int]) -> tuple:
        """
        [PRINCIPLE 1: STRUCTURAL HARMONY]
        Find the mathematically ideal function length for THIS codebase.
        
        Unlike rigid rules (e.g., "functions must be < 30 lines"),
        this discovers the natural rhythm of each individual project.
        
        Returns: (ideal_length, standard_deviation, confidence_score)
        """
        if not lengths:
            return (15, 5, 0.0)  # Default with zero confidence
        
        # Multi-method consensus approach
        methods = []
        
        # Method 1: Median (robust to outliers)
        median_val = stats.median(lengths)
        methods.append(('median', median_val, 0.6))
        
        # Method 2: Mode (most common pattern)
        try:
            from collections import Counter
            counter = Counter(lengths)
            mode_val = counter.most_common(1)[0][0]
            frequency = counter[mode_val] / len(lengths)
            methods.append(('mode', mode_val, frequency * 0.8))
        except:
            pass
        
        # Method 3: Geometric mean (for multiplicative thinking)
        try:
            # Avoid log(0) issues
            positive_lengths = [l for l in lengths if l > 0]
            if positive_lengths:
                log_sum = sum(math.log(l) for l in positive_lengths)
                geo_mean = math.exp(log_sum / len(positive_lengths))
                methods.append(('geometric_mean', geo_mean, 0.5))
        except:
            pass
        
        # Select best method by confidence
        best_method = max(methods, key=lambda x: x[2])
        ideal_length = best_method[1]
        
        # Calculate standard deviation (with bounds)
        if len(lengths) > 1:
            try:
                std_dev = stats.stdev(lengths)
                std_dev = max(min(std_dev, ideal_length * 2), 1)
            except:
                std_dev = 5
        else:
            std_dev = 5
        
        confidence = best_method[2]
        
        return (ideal_length, std_dev, confidence)
    
    @staticmethod
    def mercy_score(try_blocks: int, total_functions: int) -> float:
        """
        [PRINCIPLE 3: HUMBLE RESILIENCE]
        Calculate error handling quality with philosophical nuance.
        
        The "Mercy Rule": Even code with no error handling deserves
        some credit—it represents *potential* for resilience, not
        complete failure. This acknowledges that all systems are
        works in progress.
        
        Returns: Score between 0.0 (fragile) and 1.0 (resilient)
        """
        if total_functions == 0:
            return 0.0
        
        # Base score: proportion of functions with error handling
        base_score = try_blocks / total_functions
        
        # Mercy bonus: acknowledge potential for improvement
        mercy_bonus = 0.1 if try_blocks == 0 else 0.0
        
        # Cap at 1.0
        return min(base_score + mercy_bonus, 1.0)

# ==============================================================================
# 2. THE COGNITIVE ARCHITECTURE: ANALYTICAL ENGINE
# ==============================================================================

class CognitiveAnalyzer:
    """
    THE ANALYTICAL MIND OF RESONETICS:
    
    This class transforms raw code into philosophical insights by
    applying the three principles through multiple cognitive layers.
    
    Each analysis method corresponds to a different aspect of
    understanding code as a living, evolving entity.
    """
    
    def __init__(self, source_code: str):
        """
        Initialize the cognitive engine with code to analyze.
        
        This is where the first philosophical transformation happens:
        raw text → abstract syntax tree → semantic understanding.
        """
        # Ensure survival first
        PhilosophicalPrinciples.ensure_survival()
        
        # Parse code into abstract thought
        self.tree = ast.parse(source_code)
        
        # Extract structural elements
        self.functions = [
            node for node in ast.walk(self.tree) 
            if isinstance(node, ast.FunctionDef)
        ]
        self.classes = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.ClassDef)
        ]
        
        # Calculate contextual ideals for this codebase
        lengths = [self._function_length(f) for f in self.functions]
        self.ideal_length, self.std_dev, self.confidence = \
            PhilosophicalPrinciples.contextual_ideal(lengths)
    
    def _function_length(self, func_node: ast.FunctionDef) -> int:
        """
        Measure the temporal extent of a function's existence.
        
        In philosophical terms: "How long does this thought persist?"
        """
        end_line = func_node.end_lineno if func_node.end_lineno is not None else func_node.lineno
        return end_line - func_node.lineno + 1
    
    def _harmonic_deviation(self, actual_length: int) -> float:
        """
        Calculate how far a function deviates from structural harmony.
        
        Uses Gaussian distribution: Functions near the ideal length
        receive high scores, with penalty increasing quadratically.
        
        This implements Plato's concept of mathematical perfection.
        """
        deviation = abs(actual_length - self.ideal_length)
        gaussian_score = math.exp(-(deviation ** 2) / (2 * self.std_dev ** 2))
        return gaussian_score
    
    def _cyclomatic_complexity(self, node: ast.AST) -> int:
        """
        Measure the cognitive weight of decision-making.
        
        Based on McCabe's theory but interpreted philosophically:
        "How many distinct paths must the mind consider?"
        
        Each decision point (if, while, for, etc.) adds to the
        cognitive burden of understanding.
        """
        complexity = 1  # Base complexity: the linear path
        stack = [node]
        
        while stack:
            current = stack.pop()
            
            # Decision points increase complexity
            if isinstance(current, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            
            # Boolean logic adds sub-decisions
            elif isinstance(current, ast.BoolOp):
                complexity += len(current.values) - 1
            
            # Explore the tree of thought
            for child in ast.iter_child_nodes(current):
                stack.append(child)
        
        return complexity
    
    def _docstring_presence(self, node: Union[ast.FunctionDef, ast.ClassDef]) -> bool:
        """
        Check for explanatory consciousness.
        
        Does this code element explain itself? This is the Socratic
        question of self-knowledge: "Does it know what it does?"
        """
        # Handle Python version differences gracefully
        if sys.version_info < (3, 8):
            return bool(
                node.body and 
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Str)
            )
        else:
            return ast.get_docstring(node) is not None
    
    def analyze_functional_layer(self) -> Dict:
        """
        Layer 1: Analysis of individual functions.
        
        Evaluates each function against the ideal of structural harmony
        while considering cognitive complexity and self-awareness.
        """
        details = []
        harmony_scores = []
        
        for func in self.functions:
            length = self._function_length(func)
            harmony = self._harmonic_deviation(length)
            complexity = self._cyclomatic_complexity(func)
            self_aware = self._docstring_presence(func)
            
            details.append({
                'name': func.name,
                'length': length,
                'harmony_score': round(harmony, 3),
                'cognitive_complexity': complexity,
                'self_aware': self_aware,
                'deviation': abs(length - self.ideal_length)
            })
            
            harmony_scores.append(harmony)
        
        avg_harmony = round(stats.mean(harmony_scores), 3) if harmony_scores else 0.0
        
        return {
            'total_functions': len(details),
            'average_harmony': avg_harmony,
            'ideal_length': self.ideal_length,
            'std_deviation': self.std_dev,
            'confidence': self.confidence,
            'functions': details
        }
    
    def analyze_organizational_layer(self) -> Dict:
        """
        Layer 2: Analysis of class structures.
        
        Examines how code organizes itself into conceptual units.
        Too many methods → cognitive overload (hubris)
        Too few methods → underutilization (sloth)
        """
        details = []
        
        for cls in self.classes:
            # Count methods and attributes
            methods = [
                node for node in cls.body 
                if isinstance(node, ast.FunctionDef)
            ]
            attributes = [
                node for node in cls.body
                if isinstance(node, (ast.AnnAssign, ast.Assign))
            ]
            
            # Golden mean calculation
            method_balance = 1.0 - min(len(methods) / 20, 1.0)  # 20 methods max
            attribute_balance = 1.0 - min(len(attributes) / 10, 1.0)  # 10 attrs max
            
            self_aware = self._docstring_presence(cls)
            awareness_bonus = 0.2 if self_aware else 0.0
            
            balance_score = (
                method_balance * 0.5 + 
                attribute_balance * 0.3 + 
                awareness_bonus
            )
            
            details.append({
                'name': cls.name,
                'method_count': len(methods),
                'attribute_count': len(attributes),
                'balance_score': round(balance_score, 3),
                'self_aware': self_aware
            })
        
        avg_balance = round(
            stats.mean([d['balance_score'] for d in details]), 3
        ) if details else 0.0
        
        return {
            'total_classes': len(details),
            'average_balance': avg_balance,
            'classes': details
        }
    
    def analyze_resilience_layer(self) -> Dict:
        """
        Layer 3: Analysis of error handling and recovery.
        
        Evaluates the code's humility in acknowledging its own
        potential for failure and its preparations for recovery.
        """
        # Count try-except blocks (explicit resilience)
        try_blocks = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.Try)
        ]
        
        # Apply the Mercy Rule philosophically
        resilience_score = PhilosophicalPrinciples.mercy_score(
            len(try_blocks), 
            len(self.functions)
        )
        
        return {
            'try_blocks': len(try_blocks),
            'total_functions': len(self.functions),
            'resilience_score': round(resilience_score, 3),
            'interpretation': self._interpret_resilience(resilience_score)
        }
    
    def _interpret_resilience(self, score: float) -> str:
        """Provide philosophical interpretation of resilience score."""
        if score == 0.0:
            return "Fragile innocence (no error handling, no mercy)"
        elif score < 0.3:
            return "Naive optimism (minimal acknowledgment of failure)"
        elif score < 0.6:
            return "Practical caution (some defensive measures)"
        elif score < 0.9:
            return "Experienced resilience (robust error handling)"
        else:
            return "Philosophical acceptance (comprehensive with mercy)"

# ==============================================================================
# 3. THE SYNTHESIS ENGINE: INTEGRATING PHILOSOPHICAL INSIGHTS
# ==============================================================================

class ResoneticsSynthesizer:
    """
    THE FINAL SYNTHESIS: FROM ANALYSIS TO WISDOM
    
    This class integrates insights from all cognitive layers and
    produces a holistic philosophical assessment of the code.
    
    It applies the Aristotelian "Golden Mean" to balance different
    aspects of code quality into a coherent whole.
    """
    
    # Philosophical weights for different quality aspects
    PHILOSOPHICAL_WEIGHTS = {
        'structural_harmony': 0.35,    # Plato: Mathematical perfection
        'organizational_balance': 0.25, # Aristotle: Golden mean
        'humble_resilience': 0.25,     # Socrates: Awareness of limits
        'self_knowledge': 0.15         # Combined: Docstring presence
    }
    
    def synthesize(self, analyzer: CognitiveAnalyzer) -> Dict:
        """
        Synthesize a comprehensive philosophical assessment.
        
        This is where isolated analyses become integrated wisdom,
        following the Hegelian dialectic of thesis-antithesis-synthesis.
        """
        # Gather insights from all cognitive layers
        functional = analyzer.analyze_functional_layer()
        organizational = analyzer.analyze_organizational_layer()
        resilience = analyzer.analyze_resilience_layer()
        
        # Calculate self-knowledge score
        total_elements = functional['total_functions'] + organizational['total_classes']
        if total_elements > 0:
            aware_functions = sum(1 for f in functional['functions'] if f['self_aware'])
            aware_classes = sum(1 for c in organizational['classes'] if c['self_aware'])
            self_knowledge = (aware_functions + aware_classes) / total_elements
        else:
            self_knowledge = 0.0
        
        # Apply philosophical weights
        weighted_score = (
            functional['average_harmony'] * self.PHILOSOPHICAL_WEIGHTS['structural_harmony'] +
            organizational['average_balance'] * self.PHILOSOPHICAL_WEIGHTS['organizational_balance'] +
            resilience['resilience_score'] * self.PHILOSOPHICAL_WEIGHTS['humble_resilience'] +
            self_knowledge * self.PHILOSOPHICAL_WEIGHTS['self_knowledge']
        )
        
        # Generate philosophical diagnosis
        diagnosis = self._philosophical_diagnosis(weighted_score)
        
        return {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'overall_wisdom_score': round(weighted_score, 3),
            'philosophical_diagnosis': diagnosis,
            'cognitive_layers': {
                'functional_analysis': functional,
                'organizational_analysis': organizational,
                'resilience_analysis': resilience
            },
            'self_knowledge_score': round(self_knowledge, 3),
            'methodology': {
                'ideal_length': functional['ideal_length'],
                'confidence': functional['confidence'],
                'philosophical_weights': self.PHILOSOPHICAL_WEIGHTS
            }
        }
    
    def _philosophical_diagnosis(self, score: float) -> Dict:
        """Provide deep philosophical interpretation of the overall score."""
        if score < 0.3:
            return {
                'state': 'Chaotic potential',
                'description': 'Raw creative energy awaiting structure',
                'philosopher': 'Heraclitus (pure flux)',
                'prescription': 'Introduce Plato\'s forms through refactoring'
            }
        elif score < 0.6:
            return {
                'state': 'Structured becoming',
                'description': 'Emerging patterns with room for refinement',
                'philosopher': 'Aristotle (potential becoming actual)',
                'prescription': 'Balance extremes through the golden mean'
            }
        elif score < 0.85:
            return {
                'state': 'Harmonious resilience',
                'description': 'Well-structured with adaptive capabilities',
                'philosopher': 'Plato (forms) + Heraclitus (flow)',
                'prescription': 'Maintain balance while embracing evolution'
            }
        else:
            return {
                'state': 'Philosophical excellence',
                'description': 'Embodies all three principles in harmony',
                'philosopher': 'Socrates (wisdom through self-knowledge)',
                'prescription': 'Share knowledge and mentor others'
            }

# ==============================================================================
# 4. MAIN EXECUTION: THE RESONETICS EXPERIENCE
# ==============================================================================

def execute_resonetics_analysis(source_code: str = None, 
                               filepath: str = None) -> Dict:
    """
    The complete Resonetics philosophical analysis journey.
    
    This function orchestrates the entire process from raw code
    to philosophical wisdom, embodying the Resonetics methodology.
    """
    print("🚀 INITIATING RESONETICS PHILOSOPHICAL ANALYSIS")
    print("-" * 60)
    
    # Auto-poiesis: If no code provided, analyze self
    if source_code is None and filepath is None:
        print("🌀 Auto-poiesis mode: Analyzing self-awareness...")
        with open(__file__, 'r', encoding='utf-8') as f:
            source_code = f.read()
        print("   (Philosophical note: True wisdom begins with self-knowledge)")
    
    # Or read from file
    elif filepath:
        print(f"📖 Analyzing: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            return {
                'error': f"Cannot read file: {str(e)}",
                'philosophical_note': 'Knowledge requires accessible sources'
            }
    
    # Ensure we have code to analyze
    if not source_code:
        return {
            'error': 'No source code provided',
            'philosophical_note': 'Analysis requires substance to analyze'
        }
    
    # The Three-Stage Philosophical Journey
    print("\n📚 PHILOSOPHICAL JOURNEY:")
    print("   1. COGNITIVE ANALYSIS: Parsing code into understanding")
    print("   2. PHILOSOPHICAL ASSESSMENT: Applying ancient wisdom")
    print("   3. HOLISTIC SYNTHESIS: Integrating insights into wisdom")
    
    # Stage 1: Cognitive Analysis
    print("\n🔍 STAGE 1: COGNITIVE ANALYSIS")
    analyzer = CognitiveAnalyzer(source_code)
    
    # Stage 2 & 3: Philosophical Assessment & Synthesis
    print("🧠 STAGE 2 & 3: PHILOSOPHICAL ASSESSMENT & SYNTHESIS")
    synthesizer = ResoneticsSynthesizer()
    results = synthesizer.synthesize(analyzer)
    
    # Display key insights
    print(f"\n📊 KEY INSIGHTS:")
    print(f"   Overall Wisdom Score: {results['overall_wisdom_score']}/1.0")
    print(f"   State: {results['philosophical_diagnosis']['state']}")
    print(f"   Philosophical Inspiration: {results['philosophical_diagnosis']['philosopher']}")
    
    if results['overall_wisdom_score'] < 0.5:
        print(f"\n💡 PHILOSOPHICAL PRESCRIPTION:")
        print(f"   {results['philosophical_diagnosis']['prescription']}")
    
    return results

# ==============================================================================
# COMMAND-LINE INTERFACE: PHILOSOPHY MEETS PRACTICALITY
# ==============================================================================

if __name__ == "__main__":
    """
    The Resonetics Command-Line Experience.
    
    When invoked directly, this becomes an interactive philosophical
    dialogue about code quality, survival, and wisdom.
    """
    import sys
    
    print(f"\n{'='*70}")
    print(f"🏛️  RESONETICS PHILOSOPHICAL CODE AUDITOR")
    print(f"   Version 6.4 - Survival Edition")
    print(f"{'='*70}")
    
    # Parse command-line arguments with philosophical grace
    if len(sys.argv) > 1:
        # Analyze provided file
        filepath = sys.argv[1]
        
        if not os.path.exists(filepath):
            print(f"\n❌ File not found: {filepath}")
            print("   (Philosophical note: The path to knowledge must exist)")
            sys.exit(1)
        
        results = execute_resonetics_analysis(filepath=filepath)
    
    else:
        # Self-analysis (auto-poiesis)
        print("\n📖 No file specified. Engaging in self-analysis...")
        print("   (Ancient wisdom: 'Know thyself' - Temple of Apollo at Delphi)")
        
        results = execute_resonetics_analysis()
    
    # Output results
    print(f"\n{'='*70}")
    print("📜 ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    # Option to save to file
    save_option = input("\n💾 Save detailed results to JSON file? (y/n): ")
    if save_option.lower() == 'y':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resonetics_analysis_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {filename}")
        print("   (Philosophical note: Recorded wisdom benefits future generations)")
    
    print(f"\n🎯 RESONETICS JOURNEY COMPLETE")
    print("   May your code embody wisdom, resilience, and harmony.")
