# Copyright (c) 2025 red1239109-cmd
# Licensed under AGPL-3.0. See LICENSE file for details.

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# [Project Resonetics] 04. Adaptive Resonance Anomaly
# "Keep creativity alive, catch lies only."
# =========================================================

def adaptive_anomaly_detection(resonance_scores, window_size=5):
    """
    Adaptive Thresholding based on Local Statistics (Bollinger Bands Logic).
    It distinguishes between 'Creative Variance' and 'Structural Collapse'.
    """
    thresholds = []
    anomalies = []
    
    for i in range(len(resonance_scores)):
        # Define local context window
        start = max(0, i - window_size)
        end = min(len(resonance_scores), i + window_size + 1)
        context = resonance_scores[start:end]
        
        # Calculate local statistics
        local_mean = np.mean(context)
        local_std = np.std(context)
        
        # Dynamic Threshold: Mean - 2*Std
        # If context is unstable (creative), threshold drops (more tolerant).
        # If context is stable (logical), threshold rises (strict).
        threshold = local_mean - 2.0 * local_std
        threshold = max(threshold, 0.3)  # Minimum safety net
        
        thresholds.append(threshold)
        
        # Detection
        if resonance_scores[i] < threshold:
            anomalies.append(i)
    
    return np.array(thresholds), anomalies

# =========================================================
# [Simulation] Creative Metaphor vs. Hallucination
# =========================================================
if __name__ == "__main__":
    print("🧠 [Anomaly Check] analyzing resonance flow...")
    
    # Generate Synthetic Data
    np.random.seed(42)
    time = np.arange(100)
    scores = np.ones(100) * 0.9 # Baseline high resonance

    # Case A: Creative Metaphor (High Variance, but Continuity exists)
    # e.g., "My heart is a swaying reed."
    scores[30:40] = np.random.normal(0.7, 0.1, 10)

    # Case B: Hallucination (Sudden Collapse)
    # e.g., "King Sejong threw a MacBook."
    scores[70:75] = np.random.normal(0.2, 0.05, 5)

    # Run Detection
    thresh, anomaly_idx = adaptive_anomaly_detection(scores)

    # =========================================================
    # [Visualization] The Evidence
    # =========================================================
    plt.figure(figsize=(12, 6))
    
    # Plot Resonance Score
    plt.plot(time, scores, label="Resonance Score (AI State)", color="purple", linewidth=2)
    
    # Plot Adaptive Threshold
    plt.plot(time, thresh, "--", label="Adaptive Threshold (Tolerance)", color="orange", linewidth=2)
    
    # Highlight Anomalies
    if len(anomaly_idx) > 0:
        plt.scatter(anomaly_idx, scores[anomaly_idx], color="red", s=100, zorder=5, label="Detected Lie")

    # Annotate Regions
    plt.axvspan(30, 40, alpha=0.2, color="green", label="Creative Metaphor (Allowed)")
    plt.axvspan(70, 75, alpha=0.3, color="red", label="Hallucination (Caught!)")

    plt.title("Adaptive Resonance Anomaly Detection\nCreativity Alive, Lies Dead.", fontsize=16)
    plt.xlabel("Conversation Time")
    plt.ylabel("Resonance Level (0~1)")
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("adaptive_anomaly_result.png")
    print("✅ Result saved as 'adaptive_anomaly_result.png'")
    # plt.show() # Uncomment if running locally
