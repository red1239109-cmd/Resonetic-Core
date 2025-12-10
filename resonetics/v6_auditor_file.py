# ==============================================================================
# File: resonetics_v6_auditor.py
# Project: Resonetics (The Tool)
# Version: 6.3.1 (Survival Edition - Global)
# Author: red1239109-cmd
# Copyright (c) 2025 red1239109-cmd
#
# License: AGPL-3.0
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ==============================================================================

import ast
import math
import statistics as stats
import sys
import io
import os
from typing import Dict, List, Optional

# [Micro-Defense 1] Windows UTF-8 Encoding Fix
# Forces UTF-8 encoding on Windows consoles to prevent Emoji crashes.
# Concept: "Survival Instinct" - Adapting to hostile environments.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# [Micro-Defense 2] Robust Dependency Handling
# Falls back to standard logic if 'numpy' is missing.
# Concept: "Autopoiesis" - System maintains operation despite missing parts.
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class StructuralCodeAnalyzer:
    """
    [Resonetics Auditor v6.3.1]
    Context-Aware, Robust, and Cross-Platform Code Analysis Tool.
    Analyzes the 'Structural Resonance' and 'Survival Instinct' of Python code.
    """
    
    # Meaning-Structure Constants
    DEFAULT_IDEAL_LENGTH = 15
    DEFAULT_STD_DEV = 5
    MAX_METHOD_THRESHOLD = 20
    COMPLEXITY_PENALTY_THRESHOLD = 15

    def __init__(self, source_code: str):
        self.tree = ast.parse(source_code)
        self.functions = [n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)]
        self.classes   = [n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]

        # 1) Context-Aware Ideal Length Calculation
        # Instead of a fixed rule, it calculates the 'Median' of the current project.
        lengths = [self._func_len(f) for f in self.functions]
        
        if len(lengths) >= 4 and HAS_NUMPY:
            # Use IQR to filter outliers and find the true resonance of the code
            q1, q3 = np.percentile(lengths, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            filtered = [L for L in lengths if lower <= L <= upper]
            
            if filtered:
                self.ideal = stats.median(filtered)
                self.std   = stats.stdev(filtered) if len(filtered) > 1 else self.DEFAULT_STD_DEV
            else:
                self._set_defaults(lengths)
        else:
            self._set_defaults(lengths)

    def _set_defaults(self, lengths):
        """Fallback statistics when Numpy is missing or data is scarce."""
        if lengths:
            self.ideal = stats.median(lengths)
            self.std   = stats.stdev(lengths) if len(lengths) > 1 else self.DEFAULT_STD_DEV
        else:
            self.ideal = self.DEFAULT_IDEAL_LENGTH
            self.std   = self.DEFAULT_STD_DEV

    # -------------------- Helpers (Survival Patch Applied) --------------------
    def _func_len(self, f: ast.FunctionDef) -> int:
        """Safely calculate function length, handling REPL/Interactive modes."""
        end = f.end_lineno if f.end_lineno is not None else f.lineno
        return end - f.lineno + 1

    def _has_docstring(self, node: ast.FunctionDef | ast.ClassDef) -> bool:
        """
        [Micro-Defense 3] AST Version Compatibility
        Supports both legacy (Python < 3.8) and modern AST structures.
        """
        if sys.version_info < (3, 8):
            # Python < 3.8: Docstrings are ast.Str
            return bool(node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Str))
        else:
            # Python >= 3.8: Use standard library helper
            return ast.get_docstring(node) is not None

    def _mccabe(self, node: ast.AST) -> int:
        """
        Approximate Cyclomatic Complexity.
        Measures the 'cognitive weight' of the code.
        """
        c = 1
        for n in ast.walk(node):
            if isinstance(n, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                c += 1
            elif isinstance(n, ast.BoolOp):
                c += len(n.values) - 1
        return c

    # -------------------- Auto-Resilience Analysis --------------------
    def _analyze_resilience(self) -> Dict:
        """Checks if the code has 'Survival Instincts' (Error Handling)."""
        tries = len([n for n in ast.walk(self.tree) if isinstance(n, ast.Try)])
        funcs = len(self.functions)
        # Mercy Rule: Give small credit (0.2) even if no try-blocks exist (Potential to live)
        score = min((tries + (0 if tries else 0.2)) / max(funcs, 1), 1.0)
        return {"try_blocks": tries, "score": round(score, 2)}

    # -------------------- Main Analysis Logic --------------------
    def _analyze_functions(self) -> Dict:
        details, scores = [], []
        for f in self.functions:
            lines = self._func_len(f)
            # Resonance: Bell curve scoring based on context (Gaussian)
            dev = abs(lines - self.ideal)
            score = math.exp(-(dev**2) / (2 * self.std**2))
            
            scores.append(score)
            details.append({
                "name": f.name,
                "lines": lines,
                "complexity": self._mccabe(f),
                "docstring": self._has_docstring(f),
                "score": round(score, 2)
            })
        
        avg_score = round(stats.mean(scores), 2) if scores else 0
        return {"total": len(details), "details": details, "avg_score": avg_score}

    def _analyze_classes(self) -> Dict:
        details = []
        for cls in self.classes:
            methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
            attrs   = [n for n in cls.body if isinstance(n, ast.AnnAssign)]
            
            # Penalize God Classes (Too many methods/attributes)
            method_penalty = max(0, 1 - len(methods) / self.MAX_METHOD_THRESHOLD)
            attr_penalty   = max(0, 1 - len(attrs) / 10)
            
            score = method_penalty * 0.5 + attr_penalty * 0.3 + (0.2 if self._has_docstring(cls) else 0)
            details.append({
                "name": cls.name,
                "methods": len(methods),
                "attributes": len(attrs),
                "score": round(score, 2)
            })
        
        avg = round(stats.mean([d["score"] for d in details]), 2) if details else 0
        return {"total": len(details), "details": details, "avg_score": avg}

    # -------------------- Public API --------------------
    def analyze(self) -> Dict:
        """
        Executes the full structural audit.
        Returns a balanced report of Structure, Resilience, and Complexity.
        """
        func_r  = self._analyze_functions()
        class_r = self._analyze_classes()
        resil_r = self._analyze_resilience()

        # Calculate Complexity Penalty (Maintainability)
        avg_complexity = 0
        if func_r["total"] > 0:
            avg_complexity = sum(m["complexity"] for m in func_r["details"]) / func_r["total"]
        
        complexity_factor = 1 - min(avg_complexity / self.COMPLEXITY_PENALTY_THRESHOLD, 1)

        # Final Weighted Score (The Golden Ratio of Code Quality)
        final = (
            func_r["avg_score"] * 0.4 +
            class_r["avg_score"] * 0.2 +
            resil_r["score"]    * 0.2 +
            complexity_factor   * 0.2
        )
        return {
            "overall_score": round(final, 2),
            "functions": func_r,
            "classes": class_r,
            "resilience": resil_r,
            "complexity_penalty": round(1 - complexity_factor, 2)
        }

# ==============================================================================
# CLI (Self-Aware Mode)
# ==============================================================================
if __name__ == "__main__":
    import json, sys
    
    # If no file is provided, analyze THIS file (Autopoiesis)
    if len(sys.argv) != 2:
        print("ℹ️  No file provided. Running Self-Analysis (Autopoiesis Mode)...")
        with open(__file__, encoding="utf-8") as f:
            src = f.read()
    else:
        try:
            with open(sys.argv[1], encoding="utf-8") as f:
                src = f.read()
        except FileNotFoundError:
            print(f"❌ Error: File '{sys.argv[1]}' not found.")
            sys.exit(1)

    print(json.dumps(StructuralCodeAnalyzer(src).analyze(), indent=2, ensure_ascii=False))
