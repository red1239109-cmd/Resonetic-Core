# ==============================================================================
# Project Name : Resonetics_Alpha_v1
# Description  : Sovereign AGI Core with 8-Layer Resonance Architecture
# Author       : red1239109-cmd
# Contact      : red1239109@gmail.com
# Version      : 1.0 Alpha
#
# [LICENSE NOTICE]
# This project follows a DUAL-LICENSE model:
#
# 1. Open Source (AGPL-3.0)
#    - You are free to use, modify, and distribute this code.
#    - CONDITON: If you use this code (even over a network), you MUST open-source 
#      your entire project under AGPL-3.0.
#
# 2. Commercial License
#    - If you want to keep your source code private (Closed Source), 
#      you must purchase a Commercial License.
#    - Contact the author for pricing and details.
#
# Copyright (c) 2025 Resonetics Project. All rights reserved.
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import math
import matplotlib.pyplot as plt

# ==========================================
# 1. Helper Functions
# ==========================================
def snap_to_3x(val):
    """Snaps the value to the nearest multiple of 3.0 like a magnet."""
    return torch.round(val / 3.0) * 3.0

# ==========================================
# 2. The Sovereign Loss
#    : The Constitution and instinct of the AI
# ==========================================
class SovereignLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable parameters for automatic weight balancing (Homoscedastic Uncertainty)
        # The higher the value, the less importance (weight) is given to that specific loss.
        self.log_vars = nn.Parameter(torch.zeros(8))
        
        # Short-term memory storage for L7 (Self-Consistency)
        self.prev_pred = None 

    def forward(self, pred, sigma, target):
        """
        pred: AI's predicted value (Mu - Logic)
        sigma: AI's uncertainty (Sigma - Doubt)
        target: The ideal goal (Ground Truth)
        """
        # -----------------------------------------------------------
        # [Layer 1~4] Physical Foundation
        # -----------------------------------------------------------
        L1 = (pred - target).pow(2)                   # Surface: Gravity pulling towards the target
        L2 = torch.sin(2 * math.pi * pred / 3).pow(2) # Physics: Wave nature (Period of 3)
        L3 = torch.relu(1.5 - pred).pow(2)            # Phase: Survival threshold (Must be > 1.5)
        L4 = torch.sin(math.pi * pred).pow(2)         # Metaphor: Micro-grid structure

        # -----------------------------------------------------------
        # [Layer 5~6] Structural Thinking
        # -----------------------------------------------------------
        L5 = (pred - snap_to_3x(pred)).pow(2)         # Quantization: Digital decision (Multiples of 3)
        
        # Paradox: Prevents L5 from becoming perfectly zero (maintains 'Life')
        # *Note: Added epsilon (1e-6) to prevent log(0)
        L6 = -torch.log(L5 + 1e-6)                    

        # -----------------------------------------------------------
        # [Layer 7~8] Meta-Cognition (Self-Reflection)
        # -----------------------------------------------------------
        # L7: Self-Consistency - "Do not betray your past self."
        if self.prev_pred is None:
            L7 = torch.zeros_like(pred) 
        else:
            # Penalize the difference between previous thought (prev) and current thought (pred)
            L7 = (pred - self.prev_pred).abs()
        
        # Memory Update: Save current thought as 'past' for the next step (Detach required)
        self.prev_pred = pred.detach()

        # L8: Humble Uncertainty - "Acknowledge your ignorance."
        # Calculate entropy of the distribution. Penalize if sigma is too small (Arrogance).
        probability_dist = dist.Normal(pred, sigma)
        L8 = -probability_dist.entropy() * 0.1 # Weight scaling

        # -----------------------------------------------------------
        # [Auto-Balancing] Managing conflict between 8 layers
        # -----------------------------------------------------------
        losses = [L1, L2, L3, L4, L5, L6, L7, L8]
        total_loss = 0
        
        # Formula: Loss = (1/2σ^2) * L + log(σ)
        # Interpretation: The AI automatically lowers the weight of losses it finds difficult to solve
        # to reduce overall stress (Selection and Concentration).
        for i, L in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            diff = precision * L.mean() + self.log_vars[i]
            total_loss += diff

        return total_loss, [l.mean().item() for l in losses]

# ==========================================
# 3. Resonetic Brain (Advanced Architecture)
#    : Outputs both Logic (Mu) and Doubt (Sigma)
# ==========================================
class ResoneticBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Common Processing Cortex
        self.cortex = nn.Sequential(
            nn.Linear(1, 128),  # Increased neurons
            nn.Tanh(),          # Non-linear activation (Mimicking biological brain)
            nn.Linear(128, 128),
            nn.Tanh()
        )
        
        # Hemispheric Specialization
        self.head_logic = nn.Linear(128, 1) # Left Brain: Logical Prediction (Mu)
        self.head_doubt = nn.Linear(128, 1) # Right Brain: Intuitive Uncertainty (Sigma)

    def forward(self, x):
        thought = self.cortex(x)
        mu = self.head_logic(thought)
        
        # Sigma (Standard Deviation) must be positive. Using Softplus.
        # Enforcing minimum humility (+1e-6)
        sigma = torch.nn.functional.softplus(self.head_doubt(thought)) + 1e-6
        return mu, sigma

# ==========================================
# 4. Simulation Loop
# ==========================================
def run_simulation():
    print(f"\n{'='*60}")
    print(f"🚀 Launching Resonetic AGI : Sovereign Core")
    print(f"{'='*60}\n")

    # [Environment Setup]
    # Input: Random Noise / Goal: The Ideal '10.0'
    x = torch.randn(200, 1) 
    target = torch.full((200, 1), 10.0) 

    # [Initialize Agents]
    brain = ResoneticBrain()
    sovereign_law = SovereignLoss()
    
    # [Optimizer]
    # Learning both Brain parameters and Law weights simultaneously
    optimizer = optim.Adam(
        list(brain.parameters()) + list(sovereign_law.parameters()), 
        lr=0.005
    )

    # [History Recording]
    history_mu = []
    history_sigma = []
    loss_history = []

    # [Start Training]
    epochs = 1000
    for epoch in range(epochs + 1):
        # 1. Think (Forward)
        mu, sigma = brain(x)
        
        # 2. Reflect (Loss Calculation)
        loss, components = sovereign_law(mu, sigma, target)
        
        # 3. Adapt (Backward & Optimize)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # [Record]
        history_mu.append(mu.mean().item())
        history_sigma.append(sigma.mean().item())
        loss_history.append(loss.item())

        # [Log Output]
        if epoch % 100 == 0:
            l1, l2, l3, l4, l5, l6, l7, l8 = components
            print(f"Ep {epoch:4d} | Total: {loss.item():6.2f} | "
                  f"L1(Goal):{l1:5.2f} | L5(Code):{l5:5.2f} | "
                  f"L7(Self):{l7:5.2f} | L8(Humble):{l8:5.2f}")
            print(f"         > Current Thought(Mu): {mu.mean().item():.3f} (±{sigma.mean().item():.3f})")

    # ==========================================
    # 5. Final Analysis & Visualization
    # ==========================================
    print(f"\n{'='*60}")
    print(f"🏁 Simulation Complete. Final State Analysis")
    print(f"{'='*60}")
    print(f"1. Goal Achievement (Target 10.0): {history_mu[-1]:.4f}")
    print(f"   -> Did it settle near 9.0 or 12.0 due to the 'Rule of 3' (L5)?")
    print(f"2. Self-Confidence (Sigma): {history_sigma[-1]:.4f}")
    print(f"   -> Too low = Arrogance (Overfit), Too high = Chaos")

    # Plotting results
    try:
        plt.figure(figsize=(12, 5))
        
        # Graph 1: Convergence of Thought
        plt.subplot(1, 2, 1)
        plt.plot(history_mu, label='Thought (Mu)', color='blue')
        plt.axhline(y=10.0, color='r', linestyle='--', label='Target (10)')
        plt.axhline(y=9.0, color='g', linestyle=':', label='Structure (9)')
        plt.axhline(y=12.0, color='g', linestyle=':', label='Structure (12)')
        plt.title('Convergence of Thought')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Graph 2: Evolution of Humility
        plt.subplot(1, 2, 2)
        plt.plot(history_sigma, label='Uncertainty (Sigma)', color='orange')
        plt.title('Evolution of Humility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        print("\n📊 Generating Graphs... (Window will pop up)")
        plt.show()
    except Exception as e:
        print(f"\n(Graph generation failed: {e})")

if __name__ == "__main__":
    run_simulation()
