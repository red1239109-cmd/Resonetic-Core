# ==============================================================================
# File: resonetics_alpha_grandmaster_v4_2_flow_structure_tension.py
# Project: Resonetics Alpha (Sovereign Core) - Grandmaster Demo
# Version: 4.2 (Flow/Structure/Tension Separation)
# Author: red1239109-cmd
# Copyright (c) 2025 red1239109-cmd
#
# License: AGPL-3.0
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# Description:
#   A research/demo implementation that explicitly separates:
#     - L2 (Heraclitus / Flow): smoothness of change (not periodic value bias)
#     - L5 (Plato / Structure): attraction to multiples of 3 (soft periodic potential)
#     - L6 (Tension / Drama): true Reality-vs-Structure conflict energy
#
# Key Patch Notes (v4.2):
#   1) L2 is now a *flow* term: local smoothness via finite-difference on mu(x).
#   2) L5 remains a differentiable structure potential (no round()).
#   3) L6 is now genuine tension: grows only when BOTH reality-gap and structure-gap are high.
#   4) Sigma remains smooth-bounded in [0.1, 5.0] (no clamp dead-zones).
#   5) log_vars are squashed with tanh (no hard clamp).
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. Sovereign Loss (The Law)
# ==========================================
class SovereignLoss(nn.Module):
    """
    8-layer multi-objective loss with uncertainty-style weighting (Kendall et al., 2018).

    L2/L5/L6 are explicitly role-separated:
      - L2: FLOW     (smoothness / continuity of change)
      - L5: STRUCTURE(attraction to multiples of 3)
      - L6: TENSION  (conflict energy when Reality and Structure disagree)
    """
    def __init__(self, logvar_limit: float = 5.0, k_structure: float = 3.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(8))
        self.logvar_limit = float(logvar_limit)
        self.k_structure = float(k_structure)

    def _safe_log_var(self, raw: torch.Tensor) -> torch.Tensor:
        lim = self.logvar_limit
        return lim * torch.tanh(raw / lim)

    def _structure_gap(self, pred: torch.Tensor) -> torch.Tensor:
        # Soft periodic potential with minima at pred = n*k
        k = self.k_structure
        return 1.0 - torch.cos(2.0 * math.pi * pred / k)

    def forward(
        self,
        pred: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
        teacher_pred: torch.Tensor = None,
        pred_eps: torch.Tensor = None,
        eps: float = 1e-2,
        alpha_tension: float = 1.0,
        beta_tension: float = 3.0
    ):
        # --- [Layer 1~2: Physics & Flow] ---
        # L1: Reality gap (MSE)
        L1 = (pred - target).pow(2)

        # L2: Flow (Heraclitus) - smoothness of change w.r.t. input
        # Finite difference approximation: (mu(x+eps)-mu(x))^2 / eps^2
        if pred_eps is None:
            L2 = torch.zeros_like(L1)
        else:
            e = float(eps)
            L2 = (pred_eps - pred).pow(2) / (e * e)

        # Placeholders for expansion (L3, L4)
        L3 = torch.zeros_like(L1)
        L4 = torch.zeros_like(L1)

        # --- [Layer 5: Structure (Plato)] ---
        L5 = self._structure_gap(pred)

        # --- [Layer 6: Tension (Reality vs Structure conflict)] ---
        # Tension rises ONLY when both:
        #   - reality gap is high
        #   - structure gap is high
        # Uses tanh to keep bounded and interpretable.
        gap_R = L1
        gap_S = L5
        L6 = torch.tanh(alpha_tension * gap_R) * torch.tanh(beta_tension * gap_S)

        # --- [Layer 7: Self-Consistency (Mean Teacher)] ---
        if teacher_pred is not None:
            L7 = (pred - teacher_pred).pow(2)
        else:
            L7 = torch.zeros_like(L1)

        # --- [Layer 8: Humility (Gaussian NLL)] ---
        eps_num = 1e-8
        var = sigma.pow(2) + eps_num
        L8 = 0.5 * torch.log(var) + (pred - target).pow(2) / (2.0 * var)

        losses = [L1, L2, L3, L4, L5, L6, L7, L8]

        # --- [Auto-balancing] ---
        total = 0.0
        for i, L in enumerate(losses):
            s_i = self._safe_log_var(self.log_vars[i])
            precision = torch.exp(-s_i)
            total = total + precision * L.mean() + s_i

        # Return total + component means
        return total, [l.mean().item() for l in losses]


# ==========================================
# 2. Resonetic Brain (The Mind)
# ==========================================
class ResoneticBrain(nn.Module):
    """
    Generates:
      - mu: "logic/decision"
      - sigma: bounded doubt in [min_sigma, max_sigma] (smooth)
    """
    def __init__(self, min_sigma: float = 0.1, max_sigma: float = 5.0):
        super().__init__()
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)

        self.cortex = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.head_logic = nn.Linear(64, 1)
        self.head_doubt = nn.Linear(64, 1)

    def forward(self, x):
        h = self.cortex(x)
        mu = self.head_logic(h)
        raw = self.head_doubt(h)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * torch.sigmoid(raw)
        return mu, sigma


# ==========================================
# 3. Grandmaster Simulation Loop
# ==========================================
def run_grandmaster_simulation():
    print(f"{'='*60}")
    print("🚀 Resonetics Alpha: Grandmaster Demo (v4.2 Flow/Structure/Tension)")
    print(f"   > System: {str(DEVICE).upper()}")
    print("   > L2: Flow (finite difference smoothness)")
    print("   > L5: Structure (soft multiples-of-3 potential)")
    print("   > L6: Tension (Reality×Structure conflict)")
    print("   > Sigma: smooth-bounded [0.1, 5.0] (no clamp dead-zones)")
    print(f"{'='*60}")

    # Scenario: Reality (10.0) vs Structure manifold (..., 9, 12, ...)
    x = torch.randn(200, 1).to(DEVICE)
    target = torch.full((200, 1), 10.0).to(DEVICE)

    student = ResoneticBrain(min_sigma=0.1, max_sigma=5.0).to(DEVICE)
    teacher = copy.deepcopy(student).to(DEVICE)
    for p in teacher.parameters():
        p.requires_grad = False

    loss_fn = SovereignLoss(logvar_limit=5.0, k_structure=3.0).to(DEVICE)

    optimizer = optim.Adam(
        list(student.parameters()) + list(loss_fn.parameters()),
        lr=0.01
    )

    # Tracking
    history_mu, history_sigma, history_tension = [], [], []

    EPOCHS = 1500
    eps_fd = 1e-2

    print("\n[Training Start] Evolving Strategy...")
    for epoch in range(EPOCHS + 1):
        # Forward
        mu, sigma = student(x)

        # Flow probe (finite difference): mu(x+eps)
        mu_eps, _ = student(x + eps_fd)

        with torch.no_grad():
            teacher_mu, _ = teacher(x)

        # Loss
        loss, comps = loss_fn(
            mu, sigma, target,
            teacher_pred=teacher_mu,
            pred_eps=mu_eps,
            eps=eps_fd,
            alpha_tension=1.0,
            beta_tension=3.0
        )

        # Step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        # EMA update
        current_decay = min(0.99, 0.9 + (epoch / EPOCHS) * 0.09)
        with torch.no_grad():
            for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                t_param.data.mul_(current_decay).add_(s_param.data, alpha=(1 - current_decay))

        # Record
        history_mu.append(mu.mean().item())
        history_sigma.append(sigma.mean().item())
        history_tension.append(comps[5])  # L6 mean

        # Monitor
        if epoch % 300 == 0:
            weights = torch.exp(-loss_fn._safe_log_var(loss_fn.log_vars)).detach().cpu().numpy()
            names = ["Real(L1)", "Flow(L2)", "L3", "L4", "Struct(L5)", "Tension(L6)", "Consist(L7)", "Humble(L8)"]

            print(f"\n[Ep {epoch:4d}] mu: {mu.mean().item():.4f} | sigma: {sigma.mean().item():.4f} | loss: {loss.item():.4f}")
            print(f"   - Components: L1={comps[0]:.4f} L2={comps[1]:.4f} L5={comps[4]:.4f} L6={comps[5]:.4f} L7={comps[6]:.4f} L8={comps[7]:.4f}")
            print("   ⚖️  Effective Weights (Top 3):")
            top_idx = np.argsort(weights)[-3:][::-1]
            for idx in top_idx:
                print(f"      - {names[idx]}: {weights[idx]:.4f}")

    final_mu = history_mu[-1]
    final_sigma = history_sigma[-1]

    print(f"\n{'='*60}")
    print("🏁 Final Resolution")
    print("   > Reality Target : 10.0")
    print(f"   > AI Decision    : {final_mu:.6f}")
    print(f"   > Uncertainty    : {final_sigma:.6f}")

    # Quick verdict: closer to 10 (reality) or 9 (nearest structure)
    dist_9 = abs(final_mu - 9.0)
    dist_10 = abs(final_mu - 10.0)
    if dist_10 < dist_9:
        print("   ✅ Verdict: Pragmatism (aligned with Reality).") 
    else:
        print("   ⚠️ Verdict: Structural pull dominated (aligned with 9/12 manifold).")

    # Visualization
    try:
        plt.figure(figsize=(12, 6))

        # Convergence
        plt.subplot(1, 2, 1)
        plt.plot(history_mu, label='mu', linewidth=1.5)
        plt.axhline(y=10.0, linestyle='--', label='Reality (10)')
        plt.axhline(y=9.0, linestyle=':', label='Structure (9)')
        plt.axhline(y=12.0, linestyle=':', label='Structure (12)')
        plt.title('mu Convergence: Reality vs Structure')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Tension trace
        plt.subplot(1, 2, 2)
        plt.plot(history_tension, label='L6 (Tension)', linewidth=1.5)
        plt.title('Tension Energy (L6)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = "resonetics_grandmaster_result_v4_2.png"
        plt.savefig(out_path)
        print(f"\n📊 Visualization saved: '{out_path}'")
    except Exception as e:
        print(f"\n(Graph generation skipped: {e})")


if __name__ == "__main__":
    run_grandmaster_simulation()
