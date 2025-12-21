# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =========================================================
# [Project Resonetics] 03. SincLock (Neuro-Symbolic Logic)
# "Logic is not a step, but a wave. Make it differentiable."
# =========================================================

# Goal: Physical barrier allowing only non-zero multiples of 3 (3, 6, 9)
POSSIBLE_TARGETS = torch.tensor([3.0, 6.0, 9.0])

class SincLockNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input (Noise) -> Output (Real value between 0-10)
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # 0.0 ~ 1.0
        )
    
    def forward(self, x):
        # Scale to 0.0 ~ 9.9 range
        return self.net(x).squeeze() * 9.9

# =========================================================
# [Training and Verification Loop]
# =========================================================
def train_and_visualize():
    print("🧠 [SincLock] Training Neural Logic Gate...")
    
    model = SincLockNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    
    history = []

    for epoch in range(2001):
        z = torch.randn(64, 10) # Random Noise Input
        pred = model(z)
        
        # 1. Sine Wave Loss (Soft Guidance)
        # sin^2(2*pi*x/3): Becomes 0 at multiples of 3
        logic_loss = torch.sin(2 * np.pi * pred / 3).pow(2).mean()
        
        # sin^2(pi*x): Becomes 0 at integers (force 3.0 instead of 3.1)
        int_loss = torch.sin(np.pi * pred).pow(2).mean()
        
        # 2. Zero Blocking (Zero is absolutely forbidden)
        # Penalize heavily if less than 1.5 -> Eliminates 0 and 1
        zero_block = torch.relu(1.5 - pred).pow(2).mean() * 10
        
        # 3. Snap Loss (Final Hardening)
        # In the latter half (after 1000 epochs), force snap to the nearest valid target
        if epoch > 1000:
            # Find the closest 3, 6, 9 from current value
            # (Calculate distance between pred and POSSIBLE)
            dist = torch.cdist(pred.unsqueeze(1), POSSIBLE_TARGETS.unsqueeze(0).to(pred.device))
            closest_val = POSSIBLE_TARGETS[torch.argmin(dist, dim=1)]
            snap_loss = (pred - closest_val).pow(2).mean() * 5
        else:
            snap_loss = 0.0
            
        total_loss = logic_loss + int_loss + zero_block + snap_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {total_loss.item():.6f} | Sample: {pred[0].item():.4f}")

    # =========================================================
    # [Result Visualization: Time of Proof]
    # =========================================================
    print("✅ Training Complete! Visualizing Distribution...")
    
    # Input 1000 random data points to check what AI outputs
    with torch.no_grad():
        test_z = torch.randn(1000, 10)
        results = model(test_z).numpy()
    
    plt.figure(figsize=(10, 6))
    
    # Plot Histogram
    counts, bins, patches = plt.hist(results, bins=50, range=(0, 10), color='purple', alpha=0.7, rwidth=0.85)
    
    # Mark Target Lines
    plt.axvline(x=3, color='green', linestyle='--', linewidth=2, label='Target: 3')
    plt.axvline(x=6, color='green', linestyle='--', linewidth=2, label='Target: 6')
    plt.axvline(x=9, color='green', linestyle='--', linewidth=2, label='Target: 9')
    plt.axvline(x=0, color='red', linestyle='-', linewidth=2, label='Forbidden: 0')
    
    plt.title("SincLock Verification: AI Converges to Multiples of 3", fontsize=14)
    plt.xlabel("Generated Output Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig("sinclocks_result.png")
    print("📊 Result saved as 'sinclocks_result.png'")
    plt.show()

if __name__ == "__main__":
    train_and_visualize()
