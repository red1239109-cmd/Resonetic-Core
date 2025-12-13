# ==============================================================================
# File: resonetics_auditor_v6_5_hardened.py
# Project: Resonetics (The Tool)
# Version: 6.5 (Reviewer-Hardened)
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
"""
Resonetics Auditor v6.5 (Reviewer-Hardened)
------------------------------------------
A structural/code-quality auditor built on Python's AST.

What's new vs v6.4:
- Safer parsing: syntax errors become structured diagnostics (no crash).
- Hardened file IO: clearer errors & UTF-8 fallback strategy.
- Complexity: counts more control-flow constructs (match/case, comprehensions, with, assert).
- Resilience: adds per-function try/except density; keeps the global score for continuity.
- Numpy optional: robust percentile/IQR without numpy; avoids bare except.
- Deterministic output schema for reviewers (versioned).
"""

from __future__ import annotations

import ast
import io
import json
import math
import os
import statistics as stats
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ------------------------------
# Windows UTF-8 (best effort)
# ------------------------------
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # py3.7+
    except Exception:
        # Fallback for older envs
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        except Exception:
            pass

# ------------------------------
# Optional numpy
# ------------------------------
try:
    import numpy as _np  # type: ignore
    HAS_NUMPY = True
except Exception:
    _np = None
    HAS_NUMPY = False


# ==============================
# Utils
# ==============================

def _read_text_file(path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (content, error_message). Exactly one will be None.
    Attempts UTF-8 first, then UTF-8 with BOM, then system default.
    """
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    if not os.path.isfile(path):
        return None, f"Not a file: {path}"

    # Ordered attempts
    encodings = ["utf-8", "utf-8-sig", None]  # None => system default
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) if enc else open(path, "r") as f:
                return f.read(), None
        except Exception as e:
            last_err = e
    return None, f"Failed to read file ({path}): {last_err}"


def _percentile(sorted_data: Sequence[float], p: float) -> float:
    """
    Simple percentile (linear interpolation) on already-sorted data.
    p is in [0, 100].
    """
    if not sorted_data:
        raise ValueError("percentile() requires non-empty data")
    if p <= 0:
        return float(sorted_data[0])
    if p >= 100:
        return float(sorted_data[-1])

    n = len(sorted_data)
    idx = (n - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_data[lo])
    frac = idx - lo
    return float(sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac)


@dataclass
class ParseDiagnostics:
    ok: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None
    text: Optional[str] = None


# ==============================
# Analyzer
# ==============================

class StructuralCodeAnalyzer_v6_5:
    DEFAULT_IDEAL_LENGTH = 15
    DEFAULT_STD_DEV = 5
    MAX_METHOD_THRESHOLD = 20
    COMPLEXITY_PENALTY_THRESHOLD = 15

    def __init__(self, source_code: str, filename: str = "<memory>"):
        self.source_code = source_code
        self.filename = filename

        self.diagnostics = self._safe_parse(source_code, filename)
        self.tree: Optional[ast.AST] = self.diagnostics.ok and ast.parse(source_code, filename=filename) or None

        self.functions: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = []
        self.classes: List[ast.ClassDef] = []

        if self.tree is not None:
            self.functions = [
                n for n in ast.walk(self.tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            self.classes = [n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]

        self.ideal = self.DEFAULT_IDEAL_LENGTH
        self.std = self.DEFAULT_STD_DEV
        self._calculate_ideal_length()

    # ---------- parsing ----------
    def _safe_parse(self, src: str, filename: str) -> ParseDiagnostics:
        try:
            ast.parse(src, filename=filename)
            return ParseDiagnostics(ok=True)
        except SyntaxError as e:
            return ParseDiagnostics(
                ok=False,
                error=str(e),
                error_type="SyntaxError",
                line=getattr(e, "lineno", None),
                col=getattr(e, "offset", None),
                text=getattr(e, "text", None),
            )
        except Exception as e:
            return ParseDiagnostics(ok=False, error=str(e), error_type=type(e).__name__)

    # ---------- docstrings ----------
    def _has_docstring(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> bool:
        # ast.get_docstring exists across modern versions, but keep conservative fallback
        try:
            return ast.get_docstring(node) is not None
        except Exception:
            try:
                return bool(node.body and isinstance(node.body[0], ast.Expr))
            except Exception:
                return False

    # ---------- function length ----------
    def _func_len(self, f: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        end = getattr(f, "end_lineno", None)
        if end is None:
            end = getattr(f, "lineno", 0)
        start = getattr(f, "lineno", 0)
        return max(1, int(end) - int(start) + 1)

    # ---------- statistics ----------
    def _calculate_ideal_length(self) -> None:
        lengths = [self._func_len(f) for f in self.functions]

        if not lengths:
            self.ideal = self.DEFAULT_IDEAL_LENGTH
            self.std = self.DEFAULT_STD_DEV
            return

        self.ideal = stats.median(lengths)

        if len(lengths) > 1:
            try:
                self.std = stats.stdev(lengths)
            except stats.StatisticsError:
                self.std = self.DEFAULT_STD_DEV
        else:
            self.std = self.DEFAULT_STD_DEV

        # Optional IQR filtering if enough data
        if len(lengths) >= 4:
            try:
                sorted_lengths = sorted(float(x) for x in lengths)
                q1 = _percentile(sorted_lengths, 25)
                q3 = _percentile(sorted_lengths, 75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                filtered = [L for L in lengths if lower <= L <= upper]
                if filtered:
                    self.ideal = stats.median(filtered)
                    if len(filtered) > 1:
                        try:
                            self.std = stats.stdev(filtered)
                        except stats.StatisticsError:
                            pass
            except Exception:
                # Keep median-based defaults
                pass

        # Avoid divide-by-zero in scoring
        self.std = max(float(self.std), 1e-6)

    # ---------- complexity ----------
    def _mccabe(self, node: ast.AST) -> int:
        """
        Cyclomatic complexity approximation:
        - +1 per branching structure
        - + (n-1) for boolean chains
        - includes match/case, comprehensions, with, assert
        """
        complexity = 1
        stack = [node]

        while stack:
            cur = stack.pop()

            if isinstance(cur, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(cur, ast.BoolOp):
                complexity += max(0, len(getattr(cur, "values", [])) - 1)
            elif isinstance(cur, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(cur, ast.Assert):
                complexity += 1
            elif isinstance(cur, ast.Try):
                # try itself + each handler + else + finally paths
                complexity += 1
                complexity += len(getattr(cur, "handlers", []))
                if getattr(cur, "orelse", None):
                    complexity += 1
                if getattr(cur, "finalbody", None):
                    complexity += 1
            elif isinstance(cur, ast.Match):
                # match has multiple cases
                complexity += max(1, len(getattr(cur, "cases", [])))
            elif isinstance(cur, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # each 'for' / 'if' clause increases paths
                gens = getattr(cur, "generators", [])
                for g in gens:
                    complexity += 1
                    complexity += len(getattr(g, "ifs", []))

            for child in ast.iter_child_nodes(cur):
                stack.append(child)

        return complexity

    # ---------- resilience ----------
    def _count_try_in_node(self, node: ast.AST) -> int:
        return sum(1 for n in ast.walk(node) if isinstance(n, ast.Try))

    def _analyze_resilience(self) -> Dict[str, Any]:
        """
        Global resilience score: try-block density per function.
        Mercy rule kept, but documented and reduced.
        """
        if self.tree is None:
            return {"try_blocks": 0, "score": 0.0, "per_function": []}

        funcs = len(self.functions)
        if funcs == 0:
            return {"try_blocks": 0, "score": 0.0, "per_function": []}

        tries_total = self._count_try_in_node(self.tree)
        base_score = tries_total / funcs
        mercy_bonus = 0.1 if tries_total == 0 else 0.0  # minimal credit for "small scripts"
        score = min(base_score + mercy_bonus, 1.0)

        per_fn = []
        for f in self.functions:
            t = self._count_try_in_node(f)
            per_fn.append({
                "name": getattr(f, "name", "<lambda>"),
                "try_blocks": t,
                "try_density": round(t / max(self._func_len(f), 1), 3),
            })

        return {"try_blocks": tries_total, "score": round(score, 2), "per_function": per_fn}

    # ---------- async ----------
    def _analyze_async_features(self) -> Dict[str, Any]:
        if self.tree is None:
            return {"async_functions": 0, "await_expressions": 0, "has_async": False}
        async_count = 0
        await_count = 0
        for node in ast.walk(self.tree):
            if isinstance(node, ast.AsyncFunctionDef):
                async_count += 1
            elif isinstance(node, ast.Await):
                await_count += 1
        return {"async_functions": async_count, "await_expressions": await_count, "has_async": async_count > 0}

    # ---------- functions ----------
    def _analyze_functions(self) -> Dict[str, Any]:
        details: List[Dict[str, Any]] = []
        scores: List[float] = []

        for f in self.functions:
            lines = self._func_len(f)
            dev = abs(lines - float(self.ideal))
            score = math.exp(-(dev ** 2) / (2 * (float(self.std) ** 2)))

            comp = self._mccabe(f)
            details.append({
                "name": f.name,
                "kind": "async" if isinstance(f, ast.AsyncFunctionDef) else "sync",
                "lines": lines,
                "complexity": comp,
                "docstring": self._has_docstring(f),
                "score": round(score, 2),
            })
            scores.append(score)

        avg_score = round(stats.mean(scores), 2) if scores else 0.0
        return {"total": len(details), "details": details, "avg_score": avg_score}

    # ---------- classes ----------
    def _analyze_classes(self) -> Dict[str, Any]:
        details: List[Dict[str, Any]] = []
        for cls in self.classes:
            methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            # Include AnnAssign + Assign for attributes
            attrs = [n for n in cls.body if isinstance(n, (ast.AnnAssign, ast.Assign))]

            method_penalty = max(0.0, 1.0 - len(methods) / self.MAX_METHOD_THRESHOLD)
            attr_penalty = max(0.0, 1.0 - len(attrs) / 10.0)

            score = (
                method_penalty * 0.5
                + attr_penalty * 0.3
                + (0.2 if self._has_docstring(cls) else 0.0)
            )

            details.append({
                "name": cls.name,
                "methods": len(methods),
                "attributes": len(attrs),
                "docstring": self._has_docstring(cls),
                "score": round(score, 2),
            })

        avg = round(stats.mean([d["score"] for d in details]), 2) if details else 0.0
        return {"total": len(details), "details": details, "avg_score": avg}

    # ---------- main ----------
    def analyze(self) -> Dict[str, Any]:
        # If parse failed, return diagnostics-only schema (reviewer friendly)
        if not self.diagnostics.ok:
            return {
                "version": "6.5_hardened",
                "parse": {
                    "ok": False,
                    "error_type": self.diagnostics.error_type,
                    "error": self.diagnostics.error,
                    "line": self.diagnostics.line,
                    "col": self.diagnostics.col,
                    "text": self.diagnostics.text,
                },
                "overall_score": 0.0,
            }

        func_r = self._analyze_functions()
        class_r = self._analyze_classes()
        resil_r = self._analyze_resilience()
        async_r = self._analyze_async_features()

        # Average function complexity
        avg_complexity = 0.0
        if func_r["total"] > 0:
            avg_complexity = sum(m["complexity"] for m in func_r["details"]) / func_r["total"]

        complexity_factor = 1.0 - min(avg_complexity / self.COMPLEXITY_PENALTY_THRESHOLD, 1.0)
        async_bonus = 0.05 if async_r["has_async"] else 0.0

        final = (
            func_r["avg_score"] * 0.35
            + class_r["avg_score"] * 0.2
            + resil_r["score"] * 0.2
            + complexity_factor * 0.2
            + async_bonus
        )

        return {
            "version": "6.5_hardened",
            "parse": {"ok": True},
            "overall_score": round(final, 2),
            "functions": func_r,
            "classes": class_r,
            "resilience": resil_r,
            "async_features": async_r,
            "complexity_penalty": round(1.0 - complexity_factor, 2),
            "config": {
                "ideal_len": float(self.ideal),
                "std_len": float(self.std),
                "complexity_penalty_threshold": self.COMPLEXITY_PENALTY_THRESHOLD,
                "max_method_threshold": self.MAX_METHOD_THRESHOLD,
            },
        }


def _usage() -> str:
    return (
        "Usage:\n"
        "  python resonetics_auditor_v6_5_hardened.py [path/to/file.py]\n\n"
        "If no path is provided, the auditor analyzes itself.\n"
    )


def main(argv: List[str]) -> int:
    if len(argv) > 2:
        print(_usage())
        return 2

    if len(argv) == 2:
        p = argv[1]
        src, err = _read_text_file(p)
        if err:
            print(json.dumps({"version": "6.5_hardened", "parse": {"ok": False, "error_type": "IOError", "error": err}}, indent=2, ensure_ascii=False))
            return 1
        filename = p
    else:
        # Self-analysis
        try:
            with open(__file__, "r", encoding="utf-8") as f:
                src = f.read()
            err = None
        except Exception:
            src, err = _read_text_file(__file__)
        filename = __file__

        if err:
            print(json.dumps({"version": "6.5_hardened", "parse": {"ok": False, "error_type": "IOError", "error": err}}, indent=2, ensure_ascii=False))
            return 1

    analyzer = StructuralCodeAnalyzer_v6_5(src, filename=filename)
    result = analyzer.analyze()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
