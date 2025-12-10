# Resonetics Auditor v6.3.1 (Survival Edition)

![Version](https://img.shields.io/badge/version-6.3.1-blue) ![License](https://img.shields.io/badge/license-AGPL--3.0-red) ![Python](https://img.shields.io/badge/python-3.x-green)

> **"Structural Resonance & Survival Instinct"**
>
> A context-aware, robust, and cross-platform code analysis tool. Beyond simple linting, it statistically evaluates the "Survival Instinct" (Error Handling) and "Structural Resonance" (Ideal Length) of Python code.

## 📋 Overview

This tool uses the `ast` module to parse Python source code and calculates a quality score based on four key metrics defined in the system:

1.  **Structural Resonance:** Evaluates how harmonious function lengths are compared to the project's median, using a Gaussian distribution (Bell Curve).
2.  **Survival Instinct:** Measures the code's ability to survive in hostile environments by analyzing error handling (`try-except` blocks).
3.  **Complexity:** Calculates the 'cognitive weight' using McCabe Cyclomatic Complexity.
4.  **Autopoiesis:** The system demonstrates self-sustainment by falling back to standard libraries if dependencies (Numpy) are missing and analyzing itself if no input is provided.

## ✨ Key Features

* **Micro-Defense 1 (Windows Compatibility):** Forces UTF-8 encoding on Windows consoles to prevent crashes when rendering emojis.
* **Micro-Defense 2 (Robust Dependency):** Automatically falls back to standard statistics logic if `numpy` is not installed.
* **Micro-Defense 3 (AST Compatibility):** Detects Python versions (< 3.8 vs >= 3.8) to handle Docstring parsing logic dynamically.
* **Statistical Precision (IQR Filtering):** Uses Interquartile Range (IQR) to filter out outliers when calculating the ideal function length.
* **God Class Detection:** Applies penalties to classes that exceed the threshold for methods or attributes.

## 🛠 Installation & Requirements

This tool operates as a single independent script.

### Prerequisites
* Python 3.x

### Optional
* `numpy` (Install for more precise statistical analysis using IQR)
    ```bash
    pip install numpy
    ```
    *※ The tool runs perfectly without Numpy.*

## 🚀 Usage

Run the script via your terminal or command prompt.

### 1. Analyze a Specific File
Pass the target Python file path as an argument.
```bash
python resonetics_v6_auditor.py target_file.py

## ⚖️ Limitations & Philosophy (Honest Disclosure)

> "A tool without opinion is just a hammer. Resonetics Auditor has a soul."

**⚠️ Technical Trade-offs**
* **Adaptive Dependency:** Uses `numpy` for advanced statistics (IQR) if available, but falls back to standard deviation for survival.
* **Heuristic Weights:** The scoring ratio (Structure 40% / Resilience 20% / Maintainability 20%) is empirical, based on the **Resonetics Philosophy**.
* **Lightweight Complexity:** Uses approximate Cyclomatic Complexity to keep the tool fast and portable (Single-file architecture).

**🎨 Core Concepts**
* **Survival Instinct:** Measures how well the code handles chaos (`try-except` blocks).
* **Structural Resonance:** Analyzes the harmony of function lengths relative to the project's median.
* **Autopoiesis:** The tool itself is designed to run in any environment (Windows/Linux, Legacy Python) without breaking.
