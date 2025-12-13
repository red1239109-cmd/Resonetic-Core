# ==============================================================================
# File: ethervae_benchmark_v5_baselines.py
# Version: 5.0 (Baselines + Tradeoffs)
# Description:
#   Adds proper baselines (VAE vs β-VAE) and makes the evaluation more interpretable:
#     - Reconstruction vs KL tradeoff (ELBO components)
#     - Smoothness (local latent perturbation stability)
#     - Entropy-response curves for Ether / Resonetics inference modes
#
# Author: Resonetics Lab
# License: AGPL-3.0
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
# ==============================================================================

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


# ------------------------------
# Model
# ------------------------------

class EtherVAE(nn.Module):
    """
    A small MLP VAE for MNIST flattened images.

    "ether" and "resonetics" modes are *inference-time* latent perturbations
    applied after sampling z from q(z|x). They do not change training unless
    you explicitly incorporate them into the training objective.
    """
    def __init__(self, in_dim: int = 784, ether_dim: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.ether_dim = ether_dim

        self.enc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, ether_dim * 2)
        )

        self.dec = nn.Sequential(
            nn.Linear(ether_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, in_dim),
            nn.Sigmoid()
        )

        # Resonetics-mode knobs (exposed for reproducible experiments)
        self.wave_amp = 0.1
        self.entropy_mid = 0.5
        self.entropy_temp = 8.0

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_with_ether(self, mu: torch.Tensor, logvar: torch.Tensor, mode: str = "standard") -> torch.Tensor:
        z = self.reparameterize(mu, logvar)

        if mode == "standard":
            return z

        wave = torch.sin(z * 2 * math.pi) * self.wave_amp

        if mode == "ether":
            return z + wave

        if mode == "resonetics":
            std = torch.exp(0.5 * logvar)
            global_entropy = torch.mean(std, dim=1, keepdim=True)  # scalar per sample
            entropy_pattern = std / (std.mean(dim=1, keepdim=True) + 1e-8)

            # low entropy -> stronger wave; high entropy -> weaker wave
            entropy_factor = torch.sigmoid((self.entropy_mid - global_entropy) * self.entropy_temp)
            wave_modulated = wave * entropy_factor * entropy_pattern
            return z + wave_modulated

        raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar


# ------------------------------
# Metrics
# ------------------------------

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # KL(q(z|x) || N(0,1)) per-sample, sum over dims
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def recon_bce(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # per-sample, sum over pixels
    return F.binary_cross_entropy(recon, x, reduction="none").sum(dim=1)

@torch.no_grad()
def latent_smoothness(model: EtherVAE, mu: torch.Tensor, logvar: torch.Tensor,
                      mode: str = "standard", n_probes: int = 15, radius: float = 0.1) -> Tuple[float, float]:
    """
    Local smoothness: perturb z in random directions and measure recon drift.
    Lower is smoother (more stable).
    """
    z_base = model.generate_with_ether(mu, logvar, mode=mode)
    recon_base = model.dec(z_base)

    mses: List[float] = []
    for _ in range(n_probes):
        direction = torch.randn_like(z_base)
        direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)
        z_pert = z_base + direction * radius
        recon_pert = model.dec(z_pert)
        mses.append(float(F.mse_loss(recon_base, recon_pert).item()))
    return float(np.mean(mses)), float(np.std(mses))

@torch.no_grad()
def entropy_response_curve(model: EtherVAE, entropy_range: Tuple[float, float] = (-3.0, 3.0), n_levels: int = 25) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    For a fixed (mu=0) and varying logvar, measure how far inference modes deviate from standard.
    Returns dict: mode -> (entropy_levels, mse_vs_standard)
    """
    device = next(model.parameters()).device
    ent = torch.linspace(entropy_range[0], entropy_range[1], n_levels, device=device)

    mu = torch.zeros(1, model.ether_dim, device=device)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    diffs = { "ether": [], "resonetics": [] }
    for e in ent:
        logvar = torch.full((1, model.ether_dim), float(e.item()), device=device)
        base = model.dec(model.generate_with_ether(mu, logvar, mode="standard"))
        for mode in diffs.keys():
            recon = model.dec(model.generate_with_ether(mu, logvar, mode=mode))
            diffs[mode].append(float(F.mse_loss(recon, base).item()))

    ent_np = ent.detach().cpu().numpy()
    for mode, vals in diffs.items():
        out[mode] = (ent_np, np.array(vals, dtype=np.float32))
    return out


# ------------------------------
# Training / Evaluation
# ------------------------------

@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    beta: float = 4.0  # for β-VAE baseline
    seed: int = 42
    fast: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "bench_out_v5"

def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_loaders(batch_size: int, fast: bool = False):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))
    ])
    dataset = torchvision.datasets.MNIST(".", train=True, download=True, transform=transform)

    if fast:
        # small subset for quick sanity (reviewers can flip fast=False)
        subset_idx = torch.randperm(len(dataset))[:10000]
        dataset = torch.utils.data.Subset(dataset, subset_idx)

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_vae(model: EtherVAE, train_loader, val_loader, device, beta: float = 1.0,
              epochs: int = 12, lr: float = 1e-3, weight_decay: float = 1e-5, tag: str = "vae",
              out_dir: str = ".") -> str:
    """
    Train with objective: recon + beta * KL.
    Returns path to best checkpoint.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_path = os.path.join(out_dir, f"best_{tag}.pth")

    for ep in range(1, epochs + 1):
        model.train()
        tr_total = 0.0
        n_tr = 0

        for x, _ in train_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)

            r = recon_bce(recon, x).mean()
            k = kl_divergence(mu, logvar).mean()
            loss = r + beta * k

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_total += float(loss.item()) * x.size(0)
            n_tr += x.size(0)

        model.eval()
        va_total = 0.0
        n_va = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                r = recon_bce(recon, x).mean()
                k = kl_divergence(mu, logvar).mean()
                loss = r + beta * k
                va_total += float(loss.item()) * x.size(0)
                n_va += x.size(0)

        tr = tr_total / max(n_tr, 1)
        va = va_total / max(n_va, 1)
        print(f"[{tag}] Ep {ep:02d}/{epochs}  train={tr:.3f}  val={va:.3f}  (beta={beta})")

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), best_path)

    return best_path

@torch.no_grad()
def evaluate_tradeoff(model: EtherVAE, loader, device, modes: Tuple[str, ...] = ("standard", "ether", "resonetics")) -> Dict[str, Dict[str, float]]:
    """
    Evaluate:
      - recon BCE (per-sample sum, averaged)
      - KL (per-sample sum over dims, averaged)
    Notes:
      - KL depends only on encoder outputs (mu/logvar), independent of modes.
      - recon depends on how z is sampled/perturbed (modes).
    """
    model.eval()
    stats: Dict[str, Dict[str, float]] = {m: {"recon_bce": 0.0, "kl": 0.0, "n": 0.0} for m in modes}

    for x, _ in loader:
        x = x.to(device)
        mu, logvar = model.encode(x)
        k = kl_divergence(mu, logvar)  # (B,)

        for m in modes:
            z = model.generate_with_ether(mu, logvar, mode=m)
            recon = model.dec(z)
            r = recon_bce(recon, x)  # (B,)
            stats[m]["recon_bce"] += float(r.sum().item())
            stats[m]["kl"] += float(k.sum().item())
            stats[m]["n"] += float(x.size(0))

    out: Dict[str, Dict[str, float]] = {}
    for m, d in stats.items():
        n = max(d["n"], 1.0)
        out[m] = {
            "recon_bce": d["recon_bce"] / n,
            "kl": d["kl"] / n,
            "elbo_proxy": (d["recon_bce"] / n) + (d["kl"] / n),
        }
    return out

@torch.no_grad()
def sample_grid(model: EtherVAE, device, mode: str, n: int = 8, out_path: str = "samples.png"):
    model.eval()
    mu = torch.randn(n, model.ether_dim, device=device)
    logvar = torch.zeros(n, model.ether_dim, device=device)
    z = model.generate_with_ether(mu, logvar, mode=mode)
    recon = model.dec(z).view(n, 28, 28).cpu()

    cols = n
    plt.figure(figsize=(cols * 1.2, 1.6))
    for i in range(n):
        ax = plt.subplot(1, cols, i + 1)
        ax.imshow(recon[i], cmap="gray")
        ax.axis("off")
    plt.suptitle(f"Prior Samples (mode={mode})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--beta", type=float, default=4.0, help="β for β-VAE baseline")
    p.add_argument("--fast", action="store_true", help="Use a smaller dataset subset for speed")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--out_dir", type=str, default="bench_out_v5")
    args = p.parse_args()

    cfg = TrainConfig(epochs=args.epochs, beta=args.beta, fast=args.fast, device=args.device, out_dir=args.out_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)

    set_seeds(cfg.seed)
    device = torch.device(cfg.device)
    print(f"🔬 EtherVAE Benchmark v5 | device={device} | fast={cfg.fast} | epochs={cfg.epochs}")

    train_loader, val_loader = get_loaders(cfg.batch_size, cfg.fast)

    # --- Baseline 1: Standard VAE (beta=1)
    vae = EtherVAE(ether_dim=32)
    ckpt_vae = train_vae(
        vae, train_loader, val_loader, device,
        beta=1.0, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        tag="vae_beta1", out_dir=cfg.out_dir
    )
    vae.load_state_dict(torch.load(ckpt_vae, map_location=device))
    vae.to(device)

    # --- Baseline 2: β-VAE (beta=cfg.beta)
    bvae = EtherVAE(ether_dim=32)
    ckpt_bvae = train_vae(
        bvae, train_loader, val_loader, device,
        beta=cfg.beta, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        tag=f"vae_beta{cfg.beta:g}", out_dir=cfg.out_dir
    )
    bvae.load_state_dict(torch.load(ckpt_bvae, map_location=device))
    bvae.to(device)

    # --- Tradeoff evaluation
    modes = ("standard", "ether", "resonetics")
    trade_vae = evaluate_tradeoff(vae, val_loader, device, modes=modes)
    trade_bvae = evaluate_tradeoff(bvae, val_loader, device, modes=("standard",))  # objective baseline

    print("\n📌 Tradeoff (Validation set averages)")
    print("Standard VAE (beta=1):")
    for m in modes:
        d = trade_vae[m]
        print(f"  {m:10s} recon={d['recon_bce']:.3f}  kl={d['kl']:.3f}  elbo≈{d['elbo_proxy']:.3f}")
    print(f"β-VAE (beta={cfg.beta:g}) [standard mode]:")
    d = trade_bvae["standard"]
    print(f"  {'standard':10s} recon={d['recon_bce']:.3f}  kl={d['kl']:.3f}  elbo≈{d['elbo_proxy']:.3f}")

    # --- Plot recon vs KL
    plt.figure(figsize=(7, 4))
    for m in modes:
        d = trade_vae[m]
        plt.scatter(d["kl"], d["recon_bce"])
        plt.text(d["kl"] * 1.01, d["recon_bce"] * 1.01, f"VAE-{m}", fontsize=9)

    db = trade_bvae["standard"]
    plt.scatter(db["kl"], db["recon_bce"])
    plt.text(db["kl"] * 1.01, db["recon_bce"] * 1.01, f"βVAE({cfg.beta:g})", fontsize=9)

    plt.xlabel("KL (nats, summed over latent dims; averaged per sample)")
    plt.ylabel("Reconstruction BCE (sum over pixels; averaged per sample)")
    plt.title("Recon–KL Tradeoff (Validation)")
    plt.grid(True, alpha=0.25)
    out_trade = os.path.join(cfg.out_dir, "recon_kl_tradeoff.png")
    plt.tight_layout()
    plt.savefig(out_trade, dpi=160)
    plt.close()
    print(f"\n🖼️ Saved: {out_trade}")

    # --- Entropy response curves (from the trained standard VAE)
    curves = entropy_response_curve(vae, entropy_range=(-3, 3), n_levels=25)
    plt.figure(figsize=(7, 4))
    for mode, (x, y) in curves.items():
        plt.plot(x, y, label=mode)
    plt.xlabel("logvar (entropy level)")
    plt.ylabel("MSE vs standard mode")
    plt.title("Entropy Response Curves (Inference-time perturbations)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    out_curve = os.path.join(cfg.out_dir, "entropy_response.png")
    plt.tight_layout()
    plt.savefig(out_curve, dpi=160)
    plt.close()
    print(f"🖼️ Saved: {out_curve}")

    # --- Smoothness quick report
    print("\n🌀 Local Smoothness (lower is smoother)")
    test_configs = [
        ("Low Entropy", torch.zeros(1, 32, device=device), torch.full((1, 32), -2.0, device=device)),
        ("Mid Entropy", torch.zeros(1, 32, device=device), torch.zeros(1, 32, device=device)),
        ("High Entropy", torch.zeros(1, 32, device=device), torch.full((1, 32), 1.0, device=device)),
    ]
    for name, mu, logvar in test_configs:
        print(f"  [{name}]")
        for m in modes:
            sm, sd = latent_smoothness(vae, mu, logvar, mode=m, n_probes=15, radius=0.1)
            print(f"    {m:10s} mean={sm:.6f}  std={sd:.6f}")

    # --- Sample grids
    sample_grid(vae, device, mode="standard", out_path=os.path.join(cfg.out_dir, "samples_standard.png"))
    sample_grid(vae, device, mode="ether", out_path=os.path.join(cfg.out_dir, "samples_ether.png"))
    sample_grid(vae, device, mode="resonetics", out_path=os.path.join(cfg.out_dir, "samples_resonetics.png"))
    print(f"🖼️ Saved sample grids to: {cfg.out_dir}/samples_*.png")

    # Save a short reviewer summary
    summary_path = os.path.join(cfg.out_dir, "SUMMARY.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("EtherVAE Benchmark v5 Summary\n")
        f.write("================================\n")
        f.write(f"device: {device}\nfast: {cfg.fast}\nepochs: {cfg.epochs}\n\n")
        f.write("Standard VAE (beta=1) tradeoff:\n")
        for m in modes:
            d = trade_vae[m]
            f.write(f"  {m:10s} recon={d['recon_bce']:.6f}  kl={d['kl']:.6f}  elbo≈{d['elbo_proxy']:.6f}\n")
        f.write(f"\nβ-VAE (beta={cfg.beta:g}) tradeoff:\n")
        d = trade_bvae["standard"]
        f.write(f"  standard   recon={d['recon_bce']:.6f}  kl={d['kl']:.6f}  elbo≈{d['elbo_proxy']:.6f}\n")
        f.write("\nArtifacts:\n")
        f.write("  recon_kl_tradeoff.png\n  entropy_response.png\n  samples_standard.png\n  samples_ether.png\n  samples_resonetics.png\n")
    print(f"🧾 Saved: {summary_path}")

    print("\n✅ Done. Baselines + tradeoff plots are now in place.")

if __name__ == "__main__":
    main()
