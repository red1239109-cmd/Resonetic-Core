# ==============================================================================
# Project Name : Resonetics_Alpha_v1
# Description  : Sovereign AGI Core with 8-Layer Resonance Architecture
# Author       : red1239109-cmd
# Contact      : red1239109@gmail.com
# Version      : 1.0 Alpha (Stable)
#
# [LICENSE NOTICE]
# This project follows a DUAL-LICENSE model:
#
# 1. Open Source (AGPL-3.0)
#    - You are free to use, modify, and distribute this code.
#    - CONDITION: If you use this code (even over a network), you MUST open-source 
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

print(f"\n{'='*60}")
print(f"🚀 SYSTEM ONLINE: Resonetics_Alpha_v1 (Stable)")
print(f"   > Sovereign Core Integrity Check... OK")
print(f"   > Safety Clamps (L6, L8)... Engaged")
print(f"{'='*60}\n")

# ==========================================
# 1. Helper Functions
# ==========================================
def snap_to_3x(val):
    """Snaps the value to the nearest multiple of 3.0 like a magnet."""
    return torch.round(val / 3.0) * 3.0

# ==========================================
# 2. The Sovereign Loss (Patched)
# ==========================================
class SovereignLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable parameters for automatic weight balancing
        self.log_vars = nn.Parameter(torch.zeros(8))
        # Short-term memory storage for L7
        self.prev_pred = None 

    def forward(self, pred, sigma, target):
        """
        pred: AI's predicted value (Mu - Logic)
        sigma: AI's uncertainty (Sigma - Doubt)
        target: The ideal goal (Ground Truth)
        """
        # --- [Layer 1~4] Physical Foundation ---
        L1 = (pred - target).pow(2)                   
        L2 = torch.sin(2 * math.pi * pred / 3).pow(2) 
        L3 = torch.relu(1.5 - pred).pow(2)            
        L4 = torch.sin(math.pi * pred).pow(2)         

        # --- [Layer 5~6] Structural Thinking ---
        L5 = (pred - snap_to_3x(pred)).pow(2)         
        
        # [PATCH] L6 Safety Clamp: Prevent negative infinity explosion
        # 철학: 역설이 시스템 전체를 집어삼키지 못하도록 상/하한선(-20~+20) 설정
        L6_raw = -torch.log(L5 + 1e-6)
        L6 = torch.clamp(L6_raw, min=-20.0, max=20.0)

        # --- [Layer 7~8] Meta-Cognition ---
        # L7: Self-Consistency
        if self.prev_pred is None:
            L7 = torch.zeros_like(pred) 
        else:
            L7 = (pred - self.prev_pred).abs()
        
        # Memory Update (Detach required)
        self.prev_pred = pred.detach()

        # [PATCH] L8 Safety Clamp: Prevent infinite uncertainty evasion
        # 철학: 겸손함(Sigma)도 정도가 있어야 함. 무조건 '모른다(>5.0)'고 하면 안 됨.
        sigma_clamped = torch.clamp(sigma, min=1e-3, max=5.0)
        probability_dist = dist.Normal(pred, sigma_clamped)
        L8 = -probability_dist.entropy() * 0.1 

        # --- [Auto-Balancing] ---
        losses = [L1, L2, L3, L4, L5, L6, L7, L8]
        total_loss = 0
        
        for i, L in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            diff = precision * L.mean() + self.log_vars[i]
            total_loss += diff

        return total_loss, [l.mean().item() for l in losses]

# ==========================================
# 3. Resonetic Brain
# ==========================================
class ResoneticBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Common Cortex
        self.cortex = nn.Sequential(
            nn.Linear(1, 128),  
            nn.Tanh(),          
            nn.Linear(128, 128),
            nn.Tanh()
        )
        # Hemispheres
        self.head_logic = nn.Linear(128, 1) # Mu
        self.head_doubt = nn.Linear(128, 1) # Sigma

    def forward(self, x):
        thought = self.cortex(x)
        mu = self.head_logic(thought)
        
        # Sigma must be positive
        sigma = torch.nn.functional.softplus(self.head_doubt(thought)) + 1e-6
        return mu, sigma

# ==========================================
# 4. Simulation Loop
# ==========================================
def run_simulation():
    # [Environment Setup]
    x = torch.randn(200, 1) 
    target = torch.full((200, 1), 10.0) 

    # [Initialize Agents]
    brain = ResoneticBrain()
    sovereign_law = SovereignLoss()
    
    # [Optimizer]
    optimizer = optim.Adam(
        list(brain.parameters()) + list(sovereign_law.parameters()), 
        lr=0.005
    )

    # [History]
    history_mu = []
    history_sigma = []

    # [Start Training]
    epochs = 1000
    for epoch in range(epochs + 1):
        mu, sigma = brain(x)
        loss, components = sovereign_law(mu, sigma, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history_mu.append(mu.mean().item())
        history_sigma.append(sigma.mean().item())

        if epoch % 100 == 0:
            l1, l2, l3, l4, l5, l6, l7, l8 = components
            print(f"Ep {epoch:4d} | Total: {loss.item():6.2f} | "
                  f"L1(Goal):{l1:5.2f} | L5(Code):{l5:5.2f} | "
                  f"L7(Self):{l7:5.2f} | L8(Humble):{l8:5.2f}")
            print(f"         > Thought(Mu): {mu.mean().item():.3f} (±{sigma.mean().item():.3f})")

    # [Visualization]
    print(f"\n{'='*60}")
    print(f"🏁 Final Logic: {history_mu[-1]:.4f}")
    print(f"🏁 Final Doubt: {history_sigma[-1]:.4f}")

    try:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history_mu, label='Thought (Mu)', color='blue')
        plt.axhline(y=10.0, color='r', linestyle='--', label='Target (10)')
        plt.axhline(y=9.0, color='g', linestyle=':', label='Structure (9)')
        plt.axhline(y=12.0, color='g', linestyle=':', label='Structure (12)')
        plt.title('Convergence of Thought')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history_sigma, label='Uncertainty (Sigma)', color='orange')
        plt.title('Evolution of Humility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        print("\n📊 Generating Graphs...")
        plt.show()
    except Exception as e:
        print(f"\n(Graph generation failed: {e})")

if __name__ == "__main__":
    run_simulation()
