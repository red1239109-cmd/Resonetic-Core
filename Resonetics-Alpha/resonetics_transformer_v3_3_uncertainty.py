# ==============================================================================
# File: resonetics_transformer_v3_3_uncertainty.py
# Project: Resonetic Transformer (Unit 02) - Philosophy-Grounded Core (Uncertainty-Weighted)
# Version: 3.3 (Consolidated: Phase/Hidden History + Aux Losses + Uncertainty Weighting)
# Author: red1239109-cmd (original), patched/consolidated in-session
# License: AGPL-3.0
#
# Copyright (C) 2025
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------------
# Description:
#   A research-grade Transformer variant with:
#     - Resonance-biased attention (phase difference penalty)
#     - R-Grammar encoder (4D semantic axes: S/R/T/G)
#     - Per-layer BoundaryLayer (local contradiction/shock detector)
#     - Stabilization via residual damping (avoids multiplicative collapse)
#
#   Philosophy-grounded auxiliary losses:
#     - Plato:       sharpen + diversify R-Grammar coordinates
#     - Heraclitus:  regulate phase flow (structured change, not noise)
#     - Socrates:    calibrate boundary scores against measured layer-to-layer instability
#
#   Uncertainty-weighted combination of aux losses:
#     - Learnable log-variances automatically balance the aux losses
#       (no hand-tuned weights like 0.1 required).
#
# Notes:
#   - This is a research prototype. Full attention is O(L^2) in memory/time.
#   - For long sequences, consider FlashAttention / sparse attention / sliding window.
# ==============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# 1) Components
# ==============================

class RGrammarEncoder(nn.Module):
    """Projects hidden states into a 4D semantic coordinate system (S/R/T/G)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 4)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,L,D) -> (B,L,4) in [0,1]
        return torch.sigmoid(self.proj(x))


class ResonanceAttention(nn.Module):
    """
    Attention with a resonance bias:
      - Standard dot-product attention
      - Penalize attention scores by phase distance: scores -= resonance_scale * (Δphase)^2

    Optionally returns per-token phase values for auxiliary losses.
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.d_h = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.phase = nn.Linear(d_model, n_heads)

        # Learnable scaling factors
        self.resonance_scale = nn.Parameter(torch.tensor(0.1))
        self.phase_scale = nn.Parameter(torch.tensor(1.0))

        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.constant_(self.qkv.bias, 0.0)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_phase: bool = False,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
          x: (B,L,D)
          mask: optional padding mask. Accepts:
                - (B,L) or (B,1,L) or (B,L,L) etc.
          return_phase: if True, also return phase_val (B,H,L)
          return_attn: if True, also return attention weights (B,H,L,L)

        Returns:
          out: (B,L,D)
          attn_weights (optional): (B,H,L,L)
          phase_val (optional): (B,H,L)
        """
        B, L, D = x.shape

        # QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_h)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3,B,H,L,d_h)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Base attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_h)  # (B,H,L,L)

        # Phase values (B,H,L)
        phase_val = torch.tanh(self.phase_scale * self.phase(x))
        phase_val = phase_val.transpose(1, 2)  # (B,H,L)

        # Resonance bias: (Δphase)^2
        pi = phase_val.unsqueeze(-1)  # (B,H,L,1)
        pj = phase_val.unsqueeze(-2)  # (B,H,1,L)
        phase_diff = (pi - pj) ** 2   # (B,H,L,L)
        scores = scores - self.resonance_scale * phase_diff

        # Masking
        if mask is not None:
            # Normalize mask shapes to broadcast over (B,H,L,L)
            if mask.dim() == 2:       # (B,L)
                mask = mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,L)
            elif mask.dim() == 3:     # (B,L,L) or (B,1,L)
                mask = mask.unsqueeze(1)              # (B,1,L,L) or (B,1,1,L)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        if self.training:
            attn_weights = F.dropout(attn_weights, p=0.1)

        out = torch.matmul(attn_weights, v)           # (B,H,L,d_h)
        out = out.transpose(1, 2).reshape(B, L, D)    # (B,L,D)
        out = self.out(out)

        aw = attn_weights if return_attn else None
        pv = phase_val if return_phase else None
        return out, aw, pv


class BoundaryLayer(nn.Module):
    """
    A local guard that outputs a 'safety score' in (0,1).
    Intended meaning:
      - score ~ 1 => safe/consistent region
      - score ~ 0 => high contradiction/shock; stabilization should engage
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(1, d_model // 2)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,L,D) -> (B,L,1)
        return self.mlp(x)


# ==============================
# 2) Main Architecture
# ==============================

class ResoneticTransformer(nn.Module):
    """
    Transformer variant with:
      - R-Grammar modulation
      - ResonanceAttention
      - Per-layer Boundary check
      - Residual damping (stabilization) based on boundary score

    Returns rich histories for analysis and auxiliary losses.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        max_len: int = 1024,
        damping_alpha: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.damping_alpha = damping_alpha

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model))

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "rg": RGrammarEncoder(d_model),
                "attn": ResonanceAttention(d_model, n_heads),
                "ff": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(0.1),
                ),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
                "boundary": BoundaryLayer(d_model),
            })
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                if "embed" in name or "head" in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        nn.init.normal_(self.pos_enc, mean=0.0, std=0.02)

    def forward(
        self,
        ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          ids: (B,L)
          mask: optional padding mask (B,L) with 1 for valid tokens
          return_attn: if True, returns attention weights history (costly)

        Returns dict:
          logits: (B,L,V)
          hidden_states: (B,L,D) final
          grammar_states: (B,L,4) mean across layers
          boundary_scores: (B,L,1) mean across layers
          grammar_history: (T,B,L,4)
          shock_history: (T,B,L,1)
          hidden_history: (T,B,L,D)
          phase_history: (T,B,H,L)
          attn_history (optional): (T,B,H,L,L)
        """
        B, L = ids.shape
        x = self.embed(ids) * math.sqrt(self.d_model)
        x = x + self.pos_enc[:, :L, :]

        grammar_states = []
        shock_scores = []
        hidden_history = []
        phase_history = []
        attn_history = [] if return_attn else None

        for blk in self.blocks:
            # 1) R-Grammar
            rg = blk["rg"](x)  # (B,L,4)
            grammar_states.append(rg)

            # Modulation: mild gain from semantic coherence
            grammar_mod = 1.0 + 0.1 * rg.mean(dim=-1, keepdim=True)  # (B,L,1)
            x_mod = x * grammar_mod

            # 2) Attention (pre-norm)
            attn_in = blk["norm1"](x_mod)
            attn_out, attn_w, phase_val = blk["attn"](
                attn_in, mask, return_phase=True, return_attn=return_attn
            )
            if return_attn and attn_w is not None:
                attn_history.append(attn_w)
            if phase_val is None:
                raise RuntimeError("phase_val expected but got None (return_phase=True).")
            phase_history.append(phase_val)

            x = x_mod + attn_out

            # 3) Feed-forward (pre-norm)
            ff_in = blk["norm2"](x)
            ff_out = blk["ff"](ff_in)
            x = x + ff_out

            # 4) Local boundary check
            shock = blk["boundary"](x)  # (B,L,1), high => safe
            shock_scores.append(shock)

            # Record per-layer hidden for Socrates calibration
            hidden_history.append(x)

            # 5) Stabilization: residual damping
            alpha = self.damping_alpha
            x = x + alpha * (shock - 1.0) * x

        x = self.final_norm(x)
        logits = self.head(x)

        grammar_stack = torch.stack(grammar_states, dim=0)  # (T,B,L,4)
        shock_stack = torch.stack(shock_scores, dim=0)      # (T,B,L,1)
        hidden_stack = torch.stack(hidden_history, dim=0)   # (T,B,L,D)
        phase_stack = torch.stack(phase_history, dim=0)     # (T,B,H,L)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "hidden_states": x,
            "grammar_states": grammar_stack.mean(dim=0),
            "boundary_scores": shock_stack.mean(dim=0),
            "grammar_history": grammar_stack,
            "shock_history": shock_stack,
            "hidden_history": hidden_stack,
            "phase_history": phase_stack,
        }
        if return_attn and attn_history is not None and len(attn_history) > 0:
            out["attn_history"] = torch.stack(attn_history, dim=0)  # (T,B,H,L,L)

        return out


# ==============================
# 3) Philosophy-Grounded Auxiliary Losses
# ==============================

def compute_aux_losses(
    output: Dict[str, torch.Tensor],
    tau: float = 1.0,
    s: float = 0.3,
    tv_target: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """
    Computes auxiliary losses that ground internal modules into measurable meaning.

    output must contain:
      - grammar_history: (T,B,L,4)
      - shock_history:   (T,B,L,1)   # higher => safer
      - hidden_history:  (T,B,L,D)
      - phase_history:   (T,B,H,L)

    Returns scalar tensors:
      - L_plato
      - L_heraclitus
      - L_socrates
    """
    rg = output["grammar_history"]
    shock = output["shock_history"]
    hidden = output["hidden_history"]
    phase = output["phase_history"]

    # ---- Socrates: calibrate boundary vs measured instability
    delta = (hidden[1:] - hidden[:-1]).norm(dim=-1, keepdim=True)  # (T-1,B,L,1)
    target_shock = torch.sigmoid((delta - tau) / s)
    target_safe = 1.0 - target_shock

    shock_pred = shock[1:]  # align with delta
    L_socrates = F.binary_cross_entropy(shock_pred, target_safe.detach())

    # ---- Heraclitus: regulate phase flow (structured change)
    tv = (phase[..., 1:] - phase[..., :-1]).abs().mean()
    L_heraclitus = (tv - tv_target) ** 2

    # ---- Plato: sharpen + diversify grammar coordinates
    L_sharp = (rg * (1.0 - rg)).mean()
    m = rg.mean(dim=(0, 1, 2))  # (4,)
    L_div = ((m - 0.5) ** 2).sum()
    L_plato = L_sharp + 0.1 * L_div

    return {
        "L_plato": L_plato,
        "L_heraclitus": L_heraclitus,
        "L_socrates": L_socrates,
    }


# ==============================
# 4) Uncertainty-Weighted Auxiliary Loss
# ==============================

class AuxUncertainty(nn.Module):
    """
    Learnable uncertainty (log variance) for each auxiliary loss.
    This removes the need for hand-tuned weights.

    Weighted form:
      0.5 * (exp(-log_var) * loss + log_var)
    """

    def __init__(self):
        super().__init__()
        self.log_var_plato = nn.Parameter(torch.zeros(1))
        self.log_var_heraclitus = nn.Parameter(torch.zeros(1))
        self.log_var_socrates = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _weighted(loss: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.exp(-log_var) * loss + log_var)

    def forward(self, aux_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self._weighted(aux_losses["L_plato"], self.log_var_plato)
            + self._weighted(aux_losses["L_heraclitus"], self.log_var_heraclitus)
            + self._weighted(aux_losses["L_socrates"], self.log_var_socrates)
        )

    @torch.no_grad()
    def sigmas(self) -> Dict[str, float]:
        # sigma^2 = exp(log_var)
        return {
            "sigma2_plato": float(torch.exp(self.log_var_plato).cpu().item()),
            "sigma2_heraclitus": float(torch.exp(self.log_var_heraclitus).cpu().item()),
            "sigma2_socrates": float(torch.exp(self.log_var_socrates).cpu().item()),
        }


# ==============================
# 5) Minimal Verification & Demo Step
# ==============================

@dataclass
class DemoConfig:
    vocab_size: int = 1000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    max_len: int = 128
    seq_len: int = 16
    batch_size: int = 2
    lr: float = 3e-4
    device: str = "cpu"


def verify_model() -> None:
    print("🧪 Resonetic Transformer (V3.3) - Verification")
    print("=" * 72)

    cfg = DemoConfig()
    device = torch.device(cfg.device)

    model = ResoneticTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_len=cfg.max_len,
    ).to(device)

    aux_u = AuxUncertainty().to(device)

    model.train()

    ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
    mask = torch.ones(cfg.batch_size, cfg.seq_len, device=device)
    mask[:, -2:] = 0

    print("🔄 Forward pass...")
    out = model(ids, mask)

    print("✅ Shape checks:")
    print(f"   - logits:          {tuple(out['logits'].shape)}  (B,L,V)")
    print(f"   - grammar_history: {tuple(out['grammar_history'].shape)}  (T,B,L,4)")
    print(f"   - shock_history:   {tuple(out['shock_history'].shape)}  (T,B,L,1)")
    print(f"   - hidden_history:  {tuple(out['hidden_history'].shape)}  (T,B,L,D)")
    print(f"   - phase_history:   {tuple(out['phase_history'].shape)}  (T,B,H,L)")

    # Parameter independence check (boundaries are separate modules)
    b0 = model.blocks[0]["boundary"].mlp[0].weight
    b1 = model.blocks[1]["boundary"].mlp[0].weight
    print("✅ Independent boundary params:",
          "OK" if id(b0) != id(b1) else "FAIL")

    aux_losses = compute_aux_losses(out)
    loss_aux = aux_u(aux_losses)

    print("✅ Aux losses computed:",
          {k: float(v.detach().cpu()) for k, v in aux_losses.items()})
    print("✅ Uncertainty sigmas:", aux_u.sigmas())
    print("✅ Uncertainty-weighted aux:", float(loss_aux.detach().cpu()))

    # Backward pass check
    print("🔄 Backward pass...")
    loss = out["logits"].sum() + loss_aux
    loss.backward()

    has_grad = model.blocks[0]["boundary"].mlp[0].weight.grad is not None
    print("✅ Gradient flow (boundary):", "Success" if has_grad else "Failed")
    has_grad_u = aux_u.log_var_plato.grad is not None
    print("✅ Gradient flow (uncertainty):", "Success" if has_grad_u else "Failed")

    print("\n🚀 Verification complete. System ready.")


def demo_training_step() -> None:
    """
    A tiny, self-contained demo training step (NOT a full training script).
    - Creates random token data and a next-token target shift.
    - Computes main CE loss + uncertainty-weighted aux losses.
    """
    cfg = DemoConfig()
    device = torch.device(cfg.device)

    model = ResoneticTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_len=cfg.max_len,
    ).to(device)

    aux_u = AuxUncertainty().to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aux_u.parameters()),
        lr=cfg.lr
    )

    model.train()

    ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
    mask = torch.ones(cfg.batch_size, cfg.seq_len, device=device)

    # Next-token prediction target (shift left); last token ignored
    targets = ids.clone()
    targets[:, :-1] = ids[:, 1:]
    targets[:, -1] = -100  # ignore

    out = model(ids, mask)

    loss_main = F.cross_entropy(
        out["logits"].reshape(-1, out["logits"].size(-1)),
        targets.reshape(-1),
        ignore_index=-100
    )

    aux_losses = compute_aux_losses(out)
    loss_aux = aux_u(aux_losses)
    loss = loss_main + loss_aux

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print("🧩 Demo training step:")
    print(f"   loss_main={float(loss_main.detach().cpu()):.6f}  loss_aux={float(loss_aux.detach().cpu()):.6f}  total={float(loss.detach().cpu()):.6f}")
    print(f"   sigmas={aux_u.sigmas()}")


if __name__ == "__main__":
    verify_model()
    demo_training_step()
