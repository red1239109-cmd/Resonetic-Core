# EtherVAE v3.1: Resonetics Benchmark

![Version](https://img.shields.io/badge/version-3.1-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red) ![License](https://img.shields.io/badge/license-AGPL--3.0-green)

> **"What if the latent space wasn't a vacuum, but a medium filled with ether waves?"**

**EtherVAE** is a scientific experiment that reimagines the Variational Autoencoder (VAE) by introducing **structured interference patterns (Ether Waves)** into the latent space. It implements the core philosophy of **Resonetics**: managing entropy through adaptive resonance.

## 🔬 Experiment Modes

This benchmark compares three distinct generative modes:

1.  **Standard Mode:** The classical VAE approach using pure Gaussian noise. Safe but often blurry or generic.
2.  **Ether Mode:** Introduces a constant sinusoidal wave (`sin(z)`) into the latent vector. Adds high diversity but creates instability.
3.  **Resonetics Mode (The Hypothesis):** An entropy-aware mechanism.
    * **High Entropy (Chaos):** Suppresses waves to maintain stability.
    * **Low Entropy (Order):** Amplifies waves to encourage creativity.

## 🧪 Key Features

* **Decoupled Training/Inference:** Maintains mathematical rigor by training as a standard VAE and applying Ether effects only during inference.
* **Entropy Control Logic:** Automatically detects the "temperature" (std dev) of the latent space and adjusts wave amplitude.
* **Smoothness Metric:** Quantitatively measures the stability of the latent manifold under perturbations.

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Resonetic-Cores.git](https://github.com/YOUR_USERNAME/Resonetic-Cores.git)
    cd Resonetic-Cores
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

Run the scientific benchmark script:

```bash
python ethervae_benchmark_v3_1.py

Expected Output
The script will generate a visualization comparing the three modes. Look for:

Ether Mode: High visual variance, potentially "wavy" artifacts.

Resonetics Mode: Sharp images with controlled diversity (Low Smoothness Score = Better Stability).
