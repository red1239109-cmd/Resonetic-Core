# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: ethervae_benchmark_v6_logging.py
# Version: 6.0 (Baselines + Reproducible Experiment Logs)
# Description: EtherVAE benchmark with baselines and reviewer-friendly logging.
# Author: red1239109-cmd
# ==============================================================================

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def now_run_id() -> str:
    # Reviewer-friendly timestamp (local machine time)
    return time.strftime("%Y%m%d-%H%M%S")


class JSONLLogger:
    """Tiny, dependency-free experiment logger (JSON Lines)."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "events.jsonl")

    def log(self, event: str, **payload):
        rec = {"t": time.time(), "event": event, **payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def recon_bce(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # per-sample sum over pixels
    return F.binary_cross_entropy(recon, x, reduction="none").sum(dim=-1)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # per-sample sum over latent dims
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------


class EtherVAE(nn.Module):
    def __init__(self, in_dim: int = 784, ether_dim: int = 32):
        super().__init__()
        self.ether_dim = ether_dim

        self.enc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, ether_dim * 2),
        )

        self.dec = nn.Sequential(
            nn.Linear(ether_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, in_dim),
            nn.Sigmoid(),
        )

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

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_with_ether(self, mu: torch.Tensor, logvar: torch.Tensor, mode: str = "standard") -> torch.Tensor:
        z = self.reparameterize(mu, logvar)
        if mode == "standard":
            return z

        # Base Ether wave
        wave = torch.sin(z * 2 * np.pi) * 0.1
        if mode == "ether":
            return z + wave

        if mode == "resonetics":
            std = torch.exp(0.5 * logvar)
            global_entropy = torch.mean(std, dim=1, keepdim=True)
            entropy_pattern = std / (std.mean(dim=1, keepdim=True) + 1e-8)
            entropy_factor = torch.sigmoid((0.5 - global_entropy) * 8.0)
            wave_mod = wave * entropy_factor * entropy_pattern
            return z + wave_mod

        raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------


@torch.no_grad()
def evaluate_tradeoff(model: EtherVAE, loader, device, modes: Tuple[str, ...]) -> Dict[str, Dict[str, float]]:
    model.eval()
    stats = {m: {"recon_bce": 0.0, "kl": 0.0, "n": 0.0} for m in modes}

    for x, _ in loader:
        x = x.to(device)
        mu, logvar = model.encode(x)
        k = kl_divergence(mu, logvar)
        for m in modes:
            z = model.generate_with_ether(mu, logvar, mode=m)
            recon = model.dec(z)
            r = recon_bce(recon, x)
            stats[m]["recon_bce"] += float(r.sum().item())
            stats[m]["kl"] += float(k.sum().item())
            stats[m]["n"] += float(x.size(0))

    out = {}
    for m, d in stats.items():
        n = max(d["n"], 1.0)
        out[m] = {
            "recon_bce": d["recon_bce"] / n,
            "kl": d["kl"] / n,
            "elbo_proxy": (d["recon_bce"] / n) + (d["kl"] / n),
        }
    return out


@torch.no_grad()
def entropy_response_curve(model: EtherVAE, device, entropy_range=(-3, 3), n_levels: int = 25):
    model.eval()
    levels = torch.linspace(entropy_range[0], entropy_range[1], n_levels, device=device)
    xs = levels.detach().cpu().numpy()

    out = {"ether": [], "resonetics": []}
    for lv in levels:
        mu = torch.zeros(1, model.ether_dim, device=device)
        logvar = torch.full((1, model.ether_dim), float(lv.item()), device=device)

        z0 = model.generate_with_ether(mu, logvar, mode="standard")
        base = model.dec(z0)
        for m in out.keys():
            zm = model.generate_with_ether(mu, logvar, mode=m)
            rm = model.dec(zm)
            out[m].append(float(F.mse_loss(rm, base).item()))

    return {
        "ether": (xs, np.array(out["ether"])) ,
        "resonetics": (xs, np.array(out["resonetics"]))
    }


@torch.no_grad()
def latent_smoothness(model: EtherVAE, mu: torch.Tensor, logvar: torch.Tensor, mode: str,
                      n_probes: int = 15, radius: float = 0.1) -> Tuple[float, float]:
    z_base = model.generate_with_ether(mu, logvar, mode=mode)
    recon_base = model.dec(z_base)
    mses = []

    for _ in range(n_probes):
        d = torch.randn_like(z_base)
        d = d / (d.norm(dim=1, keepdim=True) + 1e-8)
        z = z_base + d * radius
        recon = model.dec(z)
        mses.append(float(F.mse_loss(recon, recon_base).item()))

    return float(np.mean(mses)), float(np.std(mses))


@torch.no_grad()
def sample_grid(model: EtherVAE, device, mode: str, n: int, out_path: str):
    model.eval()
    mu = torch.randn(n, model.ether_dim, device=device)
    logvar = torch.zeros(n, model.ether_dim, device=device)
    z = model.generate_with_ether(mu, logvar, mode=mode)
    recon = model.dec(z).view(n, 28, 28).cpu()

    plt.figure(figsize=(n * 1.2, 1.6))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        ax.imshow(recon[i], cmap="gray")
        ax.axis("off")
    plt.suptitle(f"Prior Samples (mode={mode})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------


def get_loaders(batch_size: int, fast: bool):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
    ])
    dataset = torchvision.datasets.MNIST(".", train=True, download=True, transform=transform)

    if fast:
        subset_idx = torch.randperm(len(dataset))[:10000]
        dataset = torch.utils.data.Subset(dataset, subset_idx)

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------


def train_vae(
    model: EtherVAE,
    train_loader,
    val_loader,
    device,
    beta: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    tag: str,
    out_dir: str,
    logger: JSONLLogger,
) -> str:
    """Train objective: recon + beta * KL. Returns best ckpt path."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_path = os.path.join(out_dir, f"best_{tag}.pth")

    for ep in range(1, epochs + 1):
        model.train()
        tr_total, n_tr = 0.0, 0
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
        va_total, n_va = 0.0, 0
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

        logger.log("epoch", tag=tag, epoch=ep, epochs=epochs, beta=beta, train_loss=tr, val_loss=va)

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), best_path)
            logger.log("checkpoint", tag=tag, epoch=ep, path=os.path.basename(best_path), best_val=best_val)

    return best_path


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-5
    beta: float = 4.0
    seed: int = 42
    fast: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    root_out: str = "runs"
    run_name: str = ""  # optional custom label


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--beta", type=float, default=4.0, help="β for β-VAE baseline")
    p.add_argument("--fast", action="store_true", help="Use a smaller MNIST subset")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--root_out", type=str, default="runs")
    p.add_argument("--run_name", type=str, default="", help="Optional run label (appended)")
    args = p.parse_args()

    cfg = TrainConfig(epochs=args.epochs, beta=args.beta, fast=args.fast, device=args.device,
                      root_out=args.root_out, run_name=args.run_name)

    set_seeds(cfg.seed)
    device = torch.device(cfg.device)

    run_id = now_run_id()
    run_dir = os.path.join(cfg.root_out, run_id + (f"_{cfg.run_name}" if cfg.run_name else ""))
    os.makedirs(run_dir, exist_ok=True)

    logger = JSONLLogger(run_dir)
    logger.log("start", config=asdict(cfg), device=str(device))

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    print(f"🔬 EtherVAE Benchmark v6 | device={device} | fast={cfg.fast} | epochs={cfg.epochs}")
    print(f"🗂️  Run dir: {run_dir}")

    train_loader, val_loader = get_loaders(cfg.batch_size, cfg.fast)

    # Baseline 1: Standard VAE (beta=1)
    vae = EtherVAE(ether_dim=32)
    ckpt_vae = train_vae(
        vae, train_loader, val_loader, device,
        beta=1.0, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        tag="vae_beta1", out_dir=run_dir, logger=logger
    )
    vae.load_state_dict(torch.load(ckpt_vae, map_location=device))
    vae.to(device)

    # Baseline 2: β-VAE (beta=cfg.beta)
    bvae = EtherVAE(ether_dim=32)
    ckpt_bvae = train_vae(
        bvae, train_loader, val_loader, device,
        beta=cfg.beta, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        tag=f"vae_beta{cfg.beta:g}", out_dir=run_dir, logger=logger
    )
    bvae.load_state_dict(torch.load(ckpt_bvae, map_location=device))
    bvae.to(device)

    # Tradeoff evaluation
    modes = ("standard", "ether", "resonetics")
    trade_vae = evaluate_tradeoff(vae, val_loader, device, modes=modes)
    trade_bvae = evaluate_tradeoff(bvae, val_loader, device, modes=("standard",))

    logger.log("tradeoff", model="vae_beta1", metrics=trade_vae)
    logger.log("tradeoff", model=f"vae_beta{cfg.beta:g}", metrics=trade_bvae)

    # Save metrics.json
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"vae_beta1": trade_vae, f"vae_beta{cfg.beta:g}": trade_bvae}, f, indent=2, ensure_ascii=False)

    print("\n📌 Tradeoff (Validation set averages)")
    print("Standard VAE (beta=1):")
    for m in modes:
        d = trade_vae[m]
        print(f"  {m:10s} recon={d['recon_bce']:.3f}  kl={d['kl']:.3f}  elbo≈{d['elbo_proxy']:.3f}")
    print(f"β-VAE (beta={cfg.beta:g}) [standard mode]:")
    d = trade_bvae["standard"]
    print(f"  {'standard':10s} recon={d['recon_bce']:.3f}  kl={d['kl']:.3f}  elbo≈{d['elbo_proxy']:.3f}")

    # Recon vs KL plot
    plt.figure(figsize=(7, 4))
    for m in modes:
        d = trade_vae[m]
        plt.scatter(d["kl"], d["recon_bce"])
        plt.text(d["kl"] * 1.01, d["recon_bce"] * 1.01, f"VAE-{m}", fontsize=9)

    db = trade_bvae["standard"]
    plt.scatter(db["kl"], db["recon_bce"])
    plt.text(db["kl"] * 1.01, db["recon_bce"] * 1.01, f"βVAE({cfg.beta:g})", fontsize=9)

    plt.xlabel("KL (nats; summed over latent dims; averaged per sample)")
    plt.ylabel("Reconstruction BCE (sum over pixels; averaged per sample)")
    plt.title("Recon–KL Tradeoff (Validation)")
    plt.grid(True, alpha=0.25)
    out_trade = os.path.join(run_dir, "recon_kl_tradeoff.png")
    plt.tight_layout()
    plt.savefig(out_trade, dpi=160)
    plt.close()
    logger.log("artifact", kind="plot", path=os.path.basename(out_trade))

    # Entropy response curves
    curves = entropy_response_curve(vae, device, entropy_range=(-3, 3), n_levels=25)
    plt.figure(figsize=(7, 4))
    for mode, (x, y) in curves.items():
        plt.plot(x, y, label=mode)
    plt.xlabel("logvar (entropy level)")
    plt.ylabel("MSE vs standard mode")
    plt.title("Entropy Response Curves (Inference-time perturbations)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    out_curve = os.path.join(run_dir, "entropy_response.png")
    plt.tight_layout()
    plt.savefig(out_curve, dpi=160)
    plt.close()
    logger.log("artifact", kind="plot", path=os.path.basename(out_curve))

    # Smoothness quick report
    smooth_report = []
    print("\n🌀 Local Smoothness (lower is smoother)")
    test_configs = [
        ("Low Entropy", torch.zeros(1, 32, device=device), torch.full((1, 32), -2.0, device=device)),
        ("Mid Entropy", torch.zeros(1, 32, device=device), torch.zeros(1, 32, device=device)),
        ("High Entropy", torch.zeros(1, 32, device=device), torch.full((1, 32), 1.0, device=device)),
    ]
    for name, mu, logvar in test_configs:
        row = {"config": name}
        print(f"  [{name}]")
        for m in modes:
            sm, sd = latent_smoothness(vae, mu, logvar, mode=m, n_probes=15, radius=0.1)
            row[m] = {"mean": sm, "std": sd}
            print(f"    {m:10s} mean={sm:.6f}  std={sd:.6f}")
        smooth_report.append(row)
    logger.log("smoothness", report=smooth_report)

    # Sample grids
    sample_grid(vae, device, mode="standard", n=8, out_path=os.path.join(run_dir, "samples_standard.png"))
    sample_grid(vae, device, mode="ether", n=8, out_path=os.path.join(run_dir, "samples_ether.png"))
    sample_grid(vae, device, mode="resonetics", n=8, out_path=os.path.join(run_dir, "samples_resonetics.png"))
    logger.log("artifact", kind="samples", path="samples_standard.png")
    logger.log("artifact", kind="samples", path="samples_ether.png")
    logger.log("artifact", kind="samples", path="samples_resonetics.png")

    # Reviewer summary
    summary_path = os.path.join(run_dir, "SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# EtherVAE Benchmark v6 Summary\n\n")
        f.write(f"- device: `{device}`\n")
        f.write(f"- fast: `{cfg.fast}`\n")
        f.write(f"- epochs: `{cfg.epochs}`\n")
        f.write(f"- beta (β-VAE): `{cfg.beta}`\n\n")
        f.write("## Tradeoff (Validation averages)\n\n")
        f.write("### Standard VAE (beta=1)\n\n")
        for m in modes:
            d = trade_vae[m]
            f.write(f"- **{m}**: recon={d['recon_bce']:.6f}, kl={d['kl']:.6f}, elbo≈{d['elbo_proxy']:.6f}\n")
        f.write("\n### β-VAE (standard mode)\n\n")
        d = trade_bvae["standard"]
        f.write(f"- recon={d['recon_bce']:.6f}, kl={d['kl']:.6f}, elbo≈{d['elbo_proxy']:.6f}\n\n")
        f.write("## Artifacts\n\n")
        f.write("- recon_kl_tradeoff.png\n")
        f.write("- entropy_response.png\n")
        f.write("- samples_standard.png\n")
        f.write("- samples_ether.png\n")
        f.write("- samples_resonetics.png\n")
        f.write("\n## Logs\n\n")
        f.write("- events.jsonl (one JSON object per line)\n")
        f.write("- config.json\n")
        f.write("- metrics.json\n")
    logger.log("artifact", kind="summary", path=os.path.basename(summary_path))

    logger.log("end", run_dir=run_dir)

    print(f"\n🧾 Saved: {summary_path}")
    print(f"🧷 Logs: {os.path.join(run_dir, 'events.jsonl')}")
    print("✅ Done.")


if __name__ == "__main__":
    main()
