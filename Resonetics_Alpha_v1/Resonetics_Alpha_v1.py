# ==============================================================================
# File: resonetics_alpha_grandmaster.py
# Project: Resonetics Alpha (Sovereign Core) - Grandmaster Edition
# Version: 4.0 (Final Stable)
# Author: red1239109-cmd
# Copyright (c) 2025 red1239109-cmd
#
# License: AGPL-3.0
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# Description:
#   The definitive implementation of the Sovereign AGI Core.
#   It integrates 'Sovereign Loss' (Philosophy), 'Resonetic Brain' (Structure),
#   and 'Mean Teacher' (Self-Consistency) into a unified evolutionary system.
#
# Key Features:
#   1. Sovereign Loss: 8-layer multi-objective optimization with auto-balancing.
#   2. EMA Teacher: Temporal self-consistency check using Exponential Moving Average.
#   3. Sigma Safety Clamps: Prevents overconfidence (0.1) and evasion (5.0).
#   4. Tension Energy: Uses Tanh-based energy to model Reality-Structure conflict.
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import math
import copy
import matplotlib.pyplot as plt
import numpy as np

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Sovereign Loss (The Law)
# ==========================================
class SovereignLoss(nn.Module):
    """
    Implements the 8-Layer Philosophy of Resonetics.
    Balances Logic, Structure, and Meta-Cognition dynamically.
    """
    def __init__(self):
        super().__init__()
        # Learnable weights for 8 loss components (Log-Space for stability)
        # Based on Multi-Task Learning (Kendall et al., CVPR 2018)
        self.log_vars = nn.Parameter(torch.zeros(8))

    def forward(self, pred, sigma, target, teacher_pred=None):
        """
        Calculates the total loss based on physical, structural, and cognitive constraints.
        
        Args:
            pred (Tensor): The logical prediction (Mu).
            sigma (Tensor): The uncertainty/doubt (Sigma).
            target (Tensor): The ground truth (Reality).
            teacher_pred (Tensor): Prediction from the EMA Teacher (Past Self).
        """
        
        # --- [Layer 1~2: Physics & Reality] ---
        # L1: Reality Gap (MSE) - The distance to ground truth.
        L1 = (pred - target).pow(2)
        
        # L2: Resonance Frequency - Encourages wave-like alignment.
        L2 = torch.sin(2 * math.pi * pred / 3.0).pow(2)
        
        # --- [Layer 5~6: Structural Alignment] ---
        # L5: Rule of Three (Structure) - Attraction to multiples of 3.
        snap_target = torch.round(pred / 3.0) * 3.0
        dist_struct = torch.abs(pred - snap_target)
        L5 = dist_struct.pow(2)
        
        # L6: Tension Energy (Paradox) - The stress between Reality and Structure.
        # Uses Tanh to prevent infinite divergence, modeling elastic tension.
        L6 = torch.tanh(torch.abs(target - snap_target)) * 10.0

        # --- [Layer 7~8: Meta-Cognition] ---
        # L7: Self-Consistency - Alignment with the stable self (Teacher).
        if teacher_pred is not None:
            L7 = (pred - teacher_pred).pow(2) 
        else:
            L7 = torch.zeros_like(pred)

        # L8: Humility (Uncertainty) - Gaussian Negative Log Likelihood.
        # Safety Clamps applied to prevent 'Overconfidence' (min=0.1) 
        # and 'Evasion' (max=5.0).
        sigma_safe = torch.clamp(sigma, min=0.1, max=5.0)
        var = sigma_safe.pow(2)
        L8 = 0.5 * torch.log(var) + (pred - target).pow(2) / (2 * var)

        # Placeholders for future expansion (L3, L4)
        L3 = torch.zeros_like(L1)
        L4 = torch.zeros_like(L1)

        # --- [Auto-Balancing Mechanism] ---
        losses = [L1, L2, L3, L4, L5, L6, L7, L8]
        total_loss = 0
        
        for i, L in enumerate(losses):
            # Clamp log_vars to prevent numerical explosion (-5 to 5 range)
            safe_log_var = torch.clamp(self.log_vars[i], min=-5.0, max=5.0)
            precision = torch.exp(-safe_log_var)
            
            # Loss = Precision * TaskLoss + RegularizationTerm
            total_loss += precision * L.mean() + safe_log_var

        return total_loss, [l.mean().item() for l in losses]

# ==========================================
# 2. Resonetic Brain (The Mind)
# ==========================================
class ResoneticBrain(nn.Module):
    """
    The neural substrate that generates Logic (Mu) and Doubt (Sigma).
    """
    def __init__(self):
        super().__init__()
        self.cortex = nn.Sequential(
            nn.Linear(1, 64), 
            nn.Tanh(),          
            nn.Linear(64, 64), 
            nn.Tanh()
        )
        self.head_logic = nn.Linear(64, 1) # Output: Mu
        self.head_doubt = nn.Linear(64, 1) # Output: Sigma

    def forward(self, x):
        h = self.cortex(x)
        mu = self.head_logic(h)
        # Softplus ensures positive uncertainty before clamping in Loss
        sigma = torch.nn.functional.softplus(self.head_doubt(h)) + 1e-4
        return mu, sigma

# ==========================================
# 3. Grandmaster Simulation Loop
# ==========================================
def run_grandmaster_simulation():
    print(f"{'='*60}")
    print("🚀 Resonetics Alpha: Grandmaster Edition (v4.0)")
    print("   > System: Initialized on " + str(DEVICE).upper())
    print("   > Safety Protocols: Sigma Clamp (0.1~5.0), Gradient Clipping")
    print("   > Meta-Learning: EMA Teacher Enabled")
    print(f"{'='*60}")

    # [Scenario Setup]
    # The Conflict: Reality (10.0) vs Structure (9.0/12.0)
    # The AI must resolve this tension.
    x = torch.randn(200, 1).to(DEVICE)
    target = torch.full((200, 1), 10.0).to(DEVICE)

    # [Agents]
    student = ResoneticBrain().to(DEVICE)
    teacher = copy.deepcopy(student).to(DEVICE) # EMA Teacher
    
    # Freeze teacher (updated only via EMA)
    for p in teacher.parameters(): 
        p.requires_grad = False
    
    loss_fn = SovereignLoss().to(DEVICE)
    
    # Unified Optimizer (Brain + Philosophy)
    optimizer = optim.Adam(
        list(student.parameters()) + list(loss_fn.parameters()), 
        lr=0.01
    )

    # History Tracking
    history_mu = []
    history_sigma = []
    
    EPOCHS = 1500

    print("\n[Training Start] Evolving Strategy...")
    for epoch in range(EPOCHS + 1):
        # 1. Forward Pass
        mu, sigma = student(x)
        with torch.no_grad():
            teacher_mu, _ = teacher(x)
            
        # 2. Compute Loss (Sovereign Law)
        loss, comps = loss_fn(mu, sigma, target, teacher_pred=teacher_mu)
        
        # 3. Backward & Optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping for Stability
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0) 
        optimizer.step()

        # 4. EMA Scheduling (Maturing Wisdom)
        # Decay increases from 0.9 (Flexible) to 0.99 (Stable) over time
        current_decay = min(0.99, 0.9 + (epoch / EPOCHS) * 0.09)
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data.mul_(current_decay).add_(s_param.data, alpha=(1 - current_decay))

        # 5. Record Stats
        history_mu.append(mu.mean().item())
        history_sigma.append(sigma.mean().item())

        # 6. Real-time Psychometrics (Monitoring)
        if epoch % 300 == 0:
            # Convert log_vars to interpretable weights
            weights = torch.exp(-loss_fn.log_vars).detach().cpu().numpy()
            
            print(f"\n[Ep {epoch:4d}] Logic: {mu.mean().item():.3f} (±{sigma.mean().item():.3f})")
            print(f"   ⚖️  Value Priorities (Top 3):")
            
            val_names = ["Real(L1)", "Wave(L2)", "L3", "L4", "Struct(L5)", "Tension(L6)", "Consist(L7)", "Humble(L8)"]
            top_idx = np.argsort(weights)[-3:][::-1]
            for idx in top_idx:
                print(f"      - {val_names[idx]}: {weights[idx]:.4f}")

    # [Final Analysis]
    final_mu = history_mu[-1]
    final_sigma = history_sigma[-1]
    
    print(f"\n{'='*60}")
    print(f"🏁 Final Resolution")
    print(f"   > Reality Target : 10.0")
    print(f"   > AI Decision    : {final_mu:.4f}")
    print(f"   > Uncertainty    : {final_sigma:.4f}")
    
    dist_9 = abs(final_mu - 9.0)
    dist_10 = abs(final_mu - 10.0)
    
    if dist_10 < dist_9:
        print("   ✅ Verdict: Pragmatism. (Aligned with Reality)")
    else:
        print("   ⚠️ Verdict: Idealism. (Aligned with Structure)")

    # [Visualization]
    try:
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Convergence
        plt.subplot(1, 2, 1)
        plt.plot(history_mu, label='AI Thought', color='blue', linewidth=1.5)
        plt.axhline(y=10.0, color='r', linestyle='--', label='Reality (10)')
        plt.axhline(y=9.0, color='g', linestyle=':', label='Structure (9)')
        
        # Uncertainty Bounds
        upper = [m + s for m, s in zip(history_mu, history_sigma)]
        lower = [m - s for m, s in zip(history_mu, history_sigma)]
        plt.fill_between(range(len(history_mu)), lower, upper, color='blue', alpha=0.1, label='Uncertainty')
        
        plt.title('The Conflict of Reason')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Value Hierarchy
        plt.subplot(1, 2, 2)
        final_weights = torch.exp(-loss_fn.log_vars).detach().cpu().numpy()
        bars = plt.barh(val_names, final_weights, color='#6c5ce7', alpha=0.7)
        plt.title('Final Value Hierarchy')
        plt.xlabel('Importance Weight')
        
        plt.tight_layout()
        plt.savefig("resonetics_grandmaster_result.png")
        print(f"\n📊 Visualization saved: 'resonetics_grandmaster_result.png'")
        # plt.show() # Uncomment to display window
    except Exception as e:
        print(f"\n(Graph generation skipped: {e})")

if __name__ == "__main__":
    run_grandmaster_simulation()
