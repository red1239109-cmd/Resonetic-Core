# ==============================================================================
# File: ethervae_benchmark_v3_1.py
# Version: 3.1 (Comprehensive Benchmark)
# Description: Scientifically rigorous evaluation of EtherVAE modes
# Author: Resonetics Lab
# License: AGPL-3.0
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# [Model Definition]
# ---------------------------------------------------------
class EtherVAE(nn.Module):
    def __init__(self, in_dim=784, ether_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, ether_dim * 2)
        )
        self.dec = nn.Sequential(
            nn.Linear(ether_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, in_dim), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """Standard VAE Sampling (Used for Training)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def generate_with_ether(self, mu, logvar, mode='standard'):
        """Inference Sampling with Ether Effects"""
        z = self.reparameterize(mu, logvar)
        
        if mode == 'standard': return z
        
        # Base Wave (The Ether)
        wave = torch.sin(z * 2 * np.pi) * 0.1
        
        if mode == 'resonetics':
            # Control Logic: 
            # Order (Low Entropy) -> Strong Wave (Creativity)
            # Chaos (High Entropy) -> Weak Wave (Stability)
            std = torch.exp(0.5 * logvar)
            avg_entropy = torch.mean(std, dim=1, keepdim=True)
            entropy_factor = torch.sigmoid((0.5 - avg_entropy) * 10.0)
            wave = wave * entropy_factor
            
        return z + wave

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

# ---------------------------------------------------------
# [Advanced Metrics]
# ---------------------------------------------------------
def calculate_latent_smoothness(model, mu, logvar, mode, n_samples=20):
    """
    Measures how smooth the latent space is by probing random directions.
    Lower value = Smoother (More robust)
    """
    smoothness_vals = []
    
    with torch.no_grad():
        z_base = model.generate_with_ether(mu, logvar, mode=mode)
        recon_base = model.dec(z_base)
        
        for _ in range(n_samples):
            # Perturb Z in random direction
            delta = torch.randn_like(z_base) * 0.05
            z_pert = z_base + delta
            
            recon_pert = model.dec(z_pert)
            mse = F.mse_loss(recon_base, recon_pert).item()
            smoothness_vals.append(mse)
            
    return np.mean(smoothness_vals)

def test_entropy_control(model, device):
    """Verifies if Resonetics reacts differently to entropy levels"""
    print("\n🧪 Testing Entropy Control Logic...")
    
    # 1. Low Entropy Case (Order)
    mu_low = torch.randn(1, 32).to(device) * 0.1
    logvar_low = torch.zeros(1, 32).to(device) - 2.0 
    
    # 2. High Entropy Case (Chaos)
    mu_high = torch.randn(1, 32).to(device) * 2.0
    logvar_high = torch.zeros(1, 32).to(device) + 1.0 
    
    with torch.no_grad():
        # Check Low Entropy
        z_r_low = model.generate_with_ether(mu_low, logvar_low, 'resonetics')
        z_e_low = model.generate_with_ether(mu_low, logvar_low, 'ether')
        diff_low = torch.mean((z_r_low - z_e_low)**2).item()
        
        # Check High Entropy
        z_r_high = model.generate_with_ether(mu_high, logvar_high, 'resonetics')
        z_e_high = model.generate_with_ether(mu_high, logvar_high, 'ether')
        diff_high = torch.mean((z_r_high - z_e_high)**2).item()
        
        print(f"   [Low Entropy]  Resonetics vs Ether Diff: {diff_low:.6f} (Small = Wave Active)")
        print(f"   [High Entropy] Resonetics vs Ether Diff: {diff_high:.6f} (Large = Wave Suppressed)")

# ---------------------------------------------------------
# [Benchmark Runner]
# ---------------------------------------------------------
def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Resonetics Benchmark v3.1 on {device}...")
    
    # 1. Train Base Model
    dataset = torchvision.datasets.MNIST('.', download=True, transform=torchvision.transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    model = EtherVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("🚀 Training Base Model (10 Epochs)...")
    for epoch in range(10):
        total_loss = 0
        for x, _ in loader:
            x = x.view(-1, 784).to(device)
            recon, mu, logvar = model(x)
            loss = F.binary_cross_entropy(recon, x, reduction='sum') + \
                   -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"   Ep {epoch+1}: Loss {total_loss/len(loader.dataset):.4f}")

    # 2. Entropy Logic Verification
    test_entropy_control(model, device)

    # 3. Visual & Metric Benchmark
    print("\n📊 Generating Benchmark Plot...")
    model.eval()
    
    fixed_mu = torch.randn(8, 32).to(device)
    fixed_logvar = torch.zeros(8, 32).to(device)
    
    modes = ['standard', 'ether', 'resonetics']
    fig, axes = plt.subplots(len(modes), 8, figsize=(12, 6))
    plt.subplots_adjust(hspace=0.6)
    
    with torch.no_grad():
        for row, mode in enumerate(modes):
            for col in range(8):
                mu_s = fixed_mu[col:col+1]
                log_s = fixed_logvar[col:col+1]
                
                z = model.generate_with_ether(mu_s, log_s, mode=mode)
                recon = model.dec(z).view(28, 28).cpu()
                smooth = calculate_latent_smoothness(model, mu_s, log_s, mode)
                
                ax = axes[row, col]
                ax.imshow(recon, cmap='gray')
                ax.axis('off')
                
                if col == 0:
                    ax.text(-0.5, 0.5, f"{mode.upper()}\nSmooth: {smooth:.4f}", 
                            transform=ax.transAxes, va='center', ha='right', fontsize=9, fontweight='bold')
                else:
                    ax.text(0.5, -0.2, f"{smooth:.4f}", transform=ax.transAxes, 
                            ha='center', fontsize=7, color='blue')

    plt.suptitle("EtherVAE Benchmark: Smoothness & Reconstruction")
    plt.show()
    print("✨ Experiment Complete.")

if __name__ == "__main__":
    run_benchmark()
