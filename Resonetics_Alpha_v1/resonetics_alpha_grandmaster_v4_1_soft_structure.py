# ==============================================================================
# File: resonetics_alpha_grandmaster_v4_1_soft_structure.py
# Project: Resonetics Alpha (Sovereign Core) - Grandmaster Demo
# Version: 4.1 (Soft-Structure Patch)
# Author: red1239109-cmd
# Copyright (c) 2025 red1239109-cmd
#
# License: AGPL-3.0
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# Description:
#   A research/demo implementation of a multi-objective "Sovereign Loss" system:
#     - Physics/Reality terms (L1, L2)
#     - Differentiable structure terms (L5, L6)  [PATCHED: removed hard round()]
#     - Mean Teacher self-consistency (L7)
#     - Uncertainty-aware NLL (L8)              [PATCHED: sigma range is smooth]
#
# Key Patch Notes (v4.1):
#   1) L5/L6 are now differentiable everywhere (no torch.round in the loss path).
#   2) Sigma is produced in a bounded interval via a smooth sigmoid mapping,
#      avoiding clamp-induced dead gradients.
#   3) log_vars are squashed smoothly (tanh) to avoid exp overflow while preserving
#      gradients (no hard clamp).
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
    8-Layer Resonetics-style multi-objective loss with uncertainty-style weighting
    (Kendall et al., 2018), implemented robustly for demo use.

    Notes:
    - L3, L4 are placeholders here (kept for interface symmetry).
    - L5/L6 are "soft structure" potentials: minima at multiples of 3 without
      non-differentiable rounding.
    """
    def __init__(self, logvar_limit: float = 5.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(8))
        self.logvar_limit = float(logvar_limit)

    def _safe_log_var(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Smoothly squash log-variance into [-limit, limit] using tanh.
        This prevents exp overflow while preserving gradients.
        """
        lim = self.logvar_limit
        return lim * torch.tanh(raw / lim)

    @staticmethod
    def _soft_structure_energy(pred: torch.Tensor, k: float = 3.0) -> torch.Tensor:
        """
        Differentiable 'attraction to multiples of k'.

        Energy:
            E_struct = 1 - cos(2π * pred / k)
        Minima occur at pred = n*k.

        This is a smooth periodic potential (a "cosine well").
        """
        return 1.0 - torch.cos(2.0 * math.pi * pred / k)

    @staticmethod
    def _phase_alignment(pred: torch.Tensor, target: torch.Tensor, k: float = 3.0) -> torch.Tensor:
        """
        Wave-like alignment between pred and target.
        Minima occur when (pred - target) is multiple of k.
        """
        return torch.sin(2.0 * math.pi * (pred - target) / k).pow(2)

    def forward(self, pred: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, teacher_pred=None):
        # --- [Layer 1~2: Physics & Reality] ---
        L1 = (pred - target).pow(2)
        L2 = self._phase_alignment(pred, target, k=3.0)

        # --- [Layer 5~6: Structural Alignment] ---
        L5 = self._soft_structure_energy(pred, k=3.0)

        gap = torch.abs(pred - target)
        L6 = torch.tanh(gap) * (1.0 + 5.0 * L5)

        # --- [Layer 7: Meta-Cognition - Self-Consistency] ---
        if teacher_pred is not None:
            L7 = (pred - teacher_pred).pow(2)
        else:
            L7 = torch.zeros_like(L1)

        # --- [Layer 8: Humility - Gaussian NLL] ---
        eps = 1e-8
        var = sigma.pow(2) + eps
        L8 = 0.5 * torch.log(var) + (pred - target).pow(2) / (2.0 * var)

        # Placeholders (L3, L4)
        L3 = torch.zeros_like(L1)
        L4 = torch.zeros_like(L1)

        losses = [L1, L2, L3, L4, L5, L6, L7, L8]

        total_loss = 0.0
        for i, L in enumerate(losses):
            s_i = self._safe_log_var(self.log_vars[i])
            precision = torch.exp(-s_i)
            total_loss = total_loss + precision * L.mean() + s_i

        return total_loss, [l.mean().item() for l in losses]


# ==========================================
# 2. Resonetic Brain (The Mind)
# ==========================================
class ResoneticBrain(nn.Module):
    """
    Generates Logic (mu) and bounded Doubt (sigma).

    v4.1 patch:
      sigma is produced via a smooth mapping into [min_sigma, max_sigma],
      avoiding clamp-induced dead gradients.
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
    print("🚀 Resonetics Alpha: Grandmaster Demo (v4.1 Soft-Structure Patch)")
    print(f"   > System: Initialized on {str(DEVICE).upper()}")
    print("   > Sigma: Smooth-bounded in [0.1, 5.0] (no clamp dead-zones)")
    print("   > Meta-Learning: EMA Teacher Enabled")
    print("   > Structure: Differentiable periodic potential (no round())")
    print(f"{'='*60}")

    x = torch.randn(200, 1).to(DEVICE)
    target = torch.full((200, 1), 10.0).to(DEVICE)

    student = ResoneticBrain(min_sigma=0.1, max_sigma=5.0).to(DEVICE)
    teacher = copy.deepcopy(student).to(DEVICE)
    for p in teacher.parameters():
        p.requires_grad = False

    loss_fn = SovereignLoss(logvar_limit=5.0).to(DEVICE)

    optimizer = optim.Adam(
        list(student.parameters()) + list(loss_fn.parameters()),
        lr=0.01
    )

    history_mu = []
    history_sigma = []
    EPOCHS = 1500

    print("\n[Training Start] Evolving Strategy...")
    for epoch in range(EPOCHS + 1):
        mu, sigma = student(x)
        with torch.no_grad():
            teacher_mu, _ = teacher(x)

        loss, _ = loss_fn(mu, sigma, target, teacher_pred=teacher_mu)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        current_decay = min(0.99, 0.9 + (epoch / EPOCHS) * 0.09)
        with torch.no_grad():
            for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                t_param.data.mul_(current_decay).add_(s_param.data, alpha=(1 - current_decay))

        history_mu.append(mu.mean().item())
        history_sigma.append(sigma.mean().item())

        if epoch % 300 == 0:
            weights = torch.exp(-loss_fn._safe_log_var(loss_fn.log_vars)).detach().cpu().numpy()
            val_names = ["Real(L1)", "Wave(L2)", "L3", "L4", "Struct(L5)", "Tension(L6)", "Consist(L7)", "Humble(L8)"]
            print(f"\n[Ep {epoch:4d}] mu: {mu.mean().item():.4f} | sigma: {sigma.mean().item():.4f} | loss: {loss.item():.4f}")
            print("   ⚖️  Effective Weights (Top 3):")
            top_idx = np.argsort(weights)[-3:][::-1]
            for idx in top_idx:
                print(f"      - {val_names[idx]}: {weights[idx]:.4f}")

    final_mu = history_mu[-1]
    final_sigma = history_sigma[-1]

    print(f"\n{'='*60}")
    print("🏁 Final Resolution")
    print("   > Reality Target : 10.0")
    print(f"   > AI Decision    : {final_mu:.6f}")
    print(f"   > Uncertainty    : {final_sigma:.6f}")

    dist_9 = abs(final_mu - 9.0)
    dist_10 = abs(final_mu - 10.0)
    if dist_10 < dist_9:
        print("   ✅ Verdict: Pragmatism (aligned with Reality).") 
    else:
        print("   ⚠️ Verdict: Structural pull dominated (aligned with 9/12 manifold).")

    try:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history_mu, label='mu (AI decision)', linewidth=1.5)
        plt.axhline(y=10.0, linestyle='--', label='Reality (10)')
        plt.axhline(y=9.0, linestyle=':', label='Structure (9)')
        plt.axhline(y=12.0, linestyle=':', label='Structure (12)')

        upper = [m + s for m, s in zip(history_mu, history_sigma)]
        lower = [m - s for m, s in zip(history_mu, history_sigma)]
        plt.fill_between(range(len(history_mu)), lower, upper, alpha=0.15, label='± sigma')

        plt.title('Constrained Dynamics: Reality vs Structure')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        val_names = ["Real(L1)", "Wave(L2)", "L3", "L4", "Struct(L5)", "Tension(L6)", "Consist(L7)", "Humble(L8)"]
        final_weights = torch.exp(-loss_fn._safe_log_var(loss_fn.log_vars)).detach().cpu().numpy()
        plt.barh(val_names, final_weights, alpha=0.7)
        plt.title('Final Effective Weighting')
        plt.xlabel('Weight')

        plt.tight_layout()
        out_path = "resonetics_grandmaster_result_v4_1.png"
        plt.savefig(out_path)
        print(f"\n📊 Visualization saved: '{out_path}'")
    except Exception as e:
        print(f"\n(Graph generation skipped: {e})")


if __name__ == "__main__":
    run_grandmaster_simulation()
