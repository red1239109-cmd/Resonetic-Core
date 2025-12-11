# Resonetics v3.3.2: The Entropy Gardener

![Version](https://img.shields.io/badge/version-3.3-blue) ![Python](https://img.shields.io/badge/python-3.x-green) ![License](https://img.shields.io/badge/license-AGPL--3.0-red)

> **"They learned to live. They learned to reproduce. They learned to fight entropy."**

**Resonetics** is a physics-informed artificial life simulation where autonomous agents evolve to survive in a hostile, high-entropy environment. Built with **PyTorch** and **NumPy**, this project demonstrates emergent behavior, Lamarckian evolution, and thermodynamic struggles in a vectorized 2D grid world.

## 🌟 Key Features

* **Vectorized Physics Engine:** Optimized logic for handling entropy diffusion and cleaning actions using NumPy masks.
* **Lamarckian Evolution:** Agents transmit their learned weights directly to their offspring via immediate genetic copying.
* **Phase-Based Architecture:** A robust simulation loop (Plan → Conflict → Commit) that prevents race conditions.
* **Neural Agents:** Each agent possesses a brain composed of a **CNN** (Visual Cortex) and an **LSTM** (Temporal Memory).
* **Real-time Telemetry:** Supports CSV logging and real-time visualization via Matplotlib.

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/red1239109-cmd/Resonetic-Cores.git
    cd Resonetics
    ```

2.  Create a virtual environment (Optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

Run the main simulation script:

```bash
python resonetics_v3_3_final.py

Visual Guide
Background (Magma Colormap):

🖤 Black/Dark: Low Entropy (Clean/Order)

🔥 Red/Yellow: High Entropy (Chaos/Danger)

Agents (Dots):

🔴 Red: Low Energy (Starving)

🟢 Green: High Energy (Healthy)

📊 Data Logging
The simulation automatically saves run statistics to resonetics_data.csv in the following format: Gen, Population, Avg_Energy, Total_Entropy
