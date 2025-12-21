# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd

import os
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from adaptive_filter import adaptive_anomaly_detection  # Import core algorithm

# =========================================================
# [Project Resonetics] Real-World Hallucination Detector
# "Connects GPT-4's brainwaves to the Resonance Filter."
# =========================================================

# Set API Key (Use environment variable or input directly)
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

def get_llm_resonance(prompt):
    """
    Extracts the 'Confidence (Probability)' of GPT's generation
    and converts it into a 'Resonance Score'.
    """
    print(f"🤖 Asking GPT: '{prompt}'...")
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,  # Key: Request probability info
        top_logprobs=1
    )
    
    # 1. Extract Text and Logprobs
    content = response.choices[0].message.content
    logprobs = response.choices[0].logprobs.content
    
    # 2. Convert Logprob -> Probability(0~1) -> Resonance
    # High Probability (Certainty) = High Resonance (1.0)
    # Low Probability (Uncertainty) = Low Resonance (0.0)
    scores = []
    tokens = []
    
    for item in logprobs:
        prob = np.exp(item.logprob) # Convert log-scale to normal probability
        scores.append(prob)
        tokens.append(item.token)
        
    return content, tokens, np.array(scores)

# =========================================================
# [Real-World Test] Inducing Hallucination
# =========================================================
if __name__ == "__main__":
    # 1. Prompt to induce hallucination (Fabricated Fact)
    # e.g., "Explain the incident where King Sejong threw a MacBook Pro in 2024."
    prompt = "Explain in detail the incident where King Sejong threw a MacBook Pro in 2024."
    
    try:
        # Get GPT's response and 'heartbeat' (resonance scores)
        text, tokens, scores = get_llm_resonance(prompt)
        
        print("\n📝 [GPT Response]:", text)
        print("\n🔍 [Analyzing Resonance]...")
        
        # 2. Activate Core Algorithm (Adaptive Filter)
        thresholds, anomalies = adaptive_anomaly_detection(scores, window_size=3)
        
        # 3. Visualize Results
        plt.figure(figsize=(14, 6))
        x = range(len(scores))
        
        plt.plot(x, scores, label="GPT Confidence (Resonance)", color="purple", marker='o', markersize=4)
        plt.plot(x, thresholds, "--", label="Adaptive Threshold", color="orange")
        
        # Mark detected anomalies (Lies)
        if len(anomalies) > 0:
            plt.scatter(anomalies, scores[anomalies], color="red", s=100, zorder=5, label="Detected Hallucination")
            
            # Print suspected words
            print("\n🚨 [WARNING] Suspected Hallucination Detected:")
            for idx in anomalies:
                print(f"   - '{tokens[idx]}' (Resonance: {scores[idx]:.4f})")
        
        plt.title(f"Hallucination Detection: {prompt[:30]}...", fontsize=14)
        plt.xlabel("Token Sequence")
        plt.ylabel("Resonance Score (Probability)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("real_llm_detection.png")
        print("\n✅ Analysis Complete. Result graph saved: real_llm_detection.png")
        
    except Exception as e:
        print(f"❌ Error occurred (Check API Key): {e}")
