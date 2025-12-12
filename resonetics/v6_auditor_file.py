# ==============================================================================
# File: resonetics_auditor_v6_4_integrated.py
# Project: Resonetics (The Tool)
# Version: 6.4 (Integrated Fixes)
# Author: red1239109-cmd
# Copyright (c) 2025 red1239109-cmd
#
# License: AGPL-3.0
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================

import ast
import math
import statistics as stats
import sys
import io
import os
from typing import Dict, List, Optional, Union

# [FIX 1: Windows UTF-8 with better handling]
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# [FIX 2: Better numpy handling]
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Create simple replacement functions
    class SimpleStats:
        @staticmethod
        def percentile(data, percentiles):
            # Simple implementation without numpy
            sorted_data = sorted(data)
            results = []
            for p in percentiles:
                idx = (len(sorted_data) - 1) * p / 100
                results.append(sorted_data[int(idx)])
            return results
    
    np = SimpleStats()

class StructuralCodeAnalyzer_v6_4:
    DEFAULT_IDEAL_LENGTH = 15
    DEFAULT_STD_DEV = 5
    MAX_METHOD_THRESHOLD = 20
    COMPLEXITY_PENALTY_THRESHOLD = 15

    def __init__(self, source_code: str):
        self.tree = ast.parse(source_code)
        self.functions = [n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)]
        self.classes = [n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]
        
        # [FIX 3: Fixed statistics calculation]
        self._calculate_ideal_length()

    def _calculate_ideal_length(self):
        """Fixed version with proper error handling"""
        lengths = [self._func_len(f) for f in self.functions]
        
        if not lengths:
            self.ideal = self.DEFAULT_IDEAL_LENGTH
            self.std = self.DEFAULT_STD_DEV
            return
        
        # Always use median
        self.ideal = stats.median(lengths)
        
        # [FIX 4: Proper stdev error handling]
        if len(lengths) > 1:
            try:
                self.std = stats.stdev(lengths)
            except stats.StatisticsError:
                self.std = self.DEFAULT_STD_DEV
        else:
            self.std = self.DEFAULT_STD_DEV
        
        # Optional: IQR filtering if numpy available and enough data
        if HAS_NUMPY and len(lengths) >= 4:
            try:
                q1, q3 = np.percentile(lengths, [25, 75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
                filtered = [L for L in lengths if lower <= L <= upper]
                
                if filtered:
                    self.ideal = stats.median(filtered)
                    if len(filtered) > 1:
                        try:
                            self.std = stats.stdev(filtered)
                        except stats.StatisticsError:
                            pass  # Keep existing std
            except:
                pass  # Fall back to simple median

    # [FIX 5: Proper type hints for older Python]
    def _has_docstring(self, node: Union[ast.FunctionDef, ast.ClassDef]) -> bool:
        if sys.version_info < (3, 8):
            return bool(node.body and isinstance(node.body[0], ast.Expr) and
                       isinstance(node.body[0].value, ast.Str))
        else:
            return ast.get_docstring(node) is not None

    # [FIX 6: More efficient McCabe calculation]
    def _mccabe(self, node: ast.AST) -> int:
        """Efficient cyclomatic complexity calculation"""
        complexity = 1
        stack = [node]
        
        while stack:
            current = stack.pop()
            
            if isinstance(current, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(current, ast.BoolOp):
                complexity += len(current.values) - 1
            
            # Add children to stack
            for child in ast.iter_child_nodes(current):
                stack.append(child)
        
        return complexity

    # [FIX 7: Clearer resilience scoring]
    def _analyze_resilience(self) -> Dict:
        """Improved error handling analysis with clear logic"""
        tries = len([n for n in ast.walk(self.tree) if isinstance(n, ast.Try)])
        funcs = len(self.functions)
        
        if funcs == 0:
            return {"try_blocks": 0, "score": 0.0}
        
        # Clear logic: tries/funcs ratio, with mercy rule explained
        # Mercy Rule: Give 0.1 (not 0.2) for code with potential to handle errors
        base_score = tries / funcs
        mercy_bonus = 0.1 if tries == 0 else 0.0
        score = min(base_score + mercy_bonus, 1.0)
        
        return {"try_blocks": tries, "score": round(score, 2)}

    # [NEW: Async/await support]
    def _analyze_async_features(self) -> Dict:
        """Check for modern Python async features"""
        async_count = 0
        await_count = 0
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.AsyncFunctionDef):
                async_count += 1
            elif isinstance(node, ast.Await):
                await_count += 1
        
        return {
            "async_functions": async_count,
            "await_expressions": await_count,
            "has_async": async_count > 0
        }

    # Rest of the methods remain similar but use fixed versions above
    def _func_len(self, f: ast.FunctionDef) -> int:
        end = f.end_lineno if f.end_lineno is not None else f.lineno
        return end - f.lineno + 1

    def _analyze_functions(self) -> Dict:
        # Uses fixed _mccabe() and proper statistics
        details, scores = [], []
        for f in self.functions:
            lines = self._func_len(f)
            dev = abs(lines - self.ideal)
            score = math.exp(-(dev**2) / (2 * self.std**2))
            
            scores.append(score)
            details.append({
                "name": f.name,
                "lines": lines,
                "complexity": self._mccabe(f),  # Fixed version
                "docstring": self._has_docstring(f),  # Fixed version
                "score": round(score, 2)
            })
        
        avg_score = round(stats.mean(scores), 2) if scores else 0
        return {"total": len(details), "details": details, "avg_score": avg_score}

    def _analyze_classes(self) -> Dict:
        # Same as before but uses fixed _has_docstring
        details = []
        for cls in self.classes:
            methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
            attrs = [n for n in cls.body if isinstance(n, ast.AnnAssign)]
            
            method_penalty = max(0, 1 - len(methods) / self.MAX_METHOD_THRESHOLD)
            attr_penalty = max(0, 1 - len(attrs) / 10)
            
            score = (method_penalty * 0.5 + 
                    attr_penalty * 0.3 + 
                    (0.2 if self._has_docstring(cls) else 0))
            
            details.append({
                "name": cls.name,
                "methods": len(methods),
                "attributes": len(attrs),
                "score": round(score, 2)
            })
        
        avg = round(stats.mean([d["score"] for d in details]), 2) if details else 0
        return {"total": len(details), "details": details, "avg_score": avg}

    def analyze(self) -> Dict:
        """Main analysis with all fixes integrated"""
        func_r = self._analyze_functions()
        class_r = self._analyze_classes()
        resil_r = self._analyze_resilience()
        async_r = self._analyze_async_features()
        
        # Complexity calculation
        avg_complexity = 0
        if func_r["total"] > 0:
            avg_complexity = sum(m["complexity"] for m in func_r["details"]) / func_r["total"]
        
        complexity_factor = 1 - min(avg_complexity / self.COMPLEXITY_PENALTY_THRESHOLD, 1)

        # Final score with async bonus
        async_bonus = 0.05 if async_r["has_async"] else 0.0
        
        final = (
            func_r["avg_score"] * 0.35 +
            class_r["avg_score"] * 0.2 +
            resil_r["score"] * 0.2 +
            complexity_factor * 0.2 +
            async_bonus
        )
        
        return {
            "overall_score": round(final, 2),
            "functions": func_r,
            "classes": class_r,
            "resilience": resil_r,
            "async_features": async_r,
            "complexity_penalty": round(1 - complexity_factor, 2),
            "version": "6.4_integrated"
        }

if __name__ == "__main__":
    import json
    
    if len(sys.argv) != 2:
        print("🔍 Self-analysis mode")
        with open(__file__, encoding="utf-8") as f:
            src = f.read()
    else:
        with open(sys.argv[1], encoding="utf-8") as f:
            src = f.read()
    
    analyzer = StructuralCodeAnalyzer_v6_4(src)
    result = analyzer.analyze()
    print(json.dumps(result, indent=2, ensure_ascii=False))
