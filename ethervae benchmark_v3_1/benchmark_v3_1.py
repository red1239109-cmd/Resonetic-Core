# ==============================================================================
# File: ethervae_benchmark_v4_comprehensive.py
# Version: 4.0 (Scientific Deep Dive)
# Description: Multi-dimensional evaluation of EtherVAE
# Author: Resonetics Lab
# License: AGPL-3.0
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

print(f"\n{'='*70}")
print(f"🔬 ETHERVAE BENCHMARK v4.0 - SCIENTIFIC DEEP DIVE")
print(f"{'='*70}\n")

# ---------------------------------------------------------
# [Enhanced EtherVAE Model]
# ---------------------------------------------------------
class EtherVAE(nn.Module):
    def __init__(self, in_dim=784, ether_dim=32):
        super().__init__()
        self.ether_dim = ether_dim
        
        # Encoder with batch norm for stability
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, ether_dim * 2)
        )
        
        # Decoder with residual connections
        self.dec = nn.Sequential(
            nn.Linear(ether_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, in_dim),
            nn.Sigmoid()
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def reparameterize(self, mu, logvar):
        """Standard VAE sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def generate_with_ether(self, mu, logvar, mode='standard'):
        """Inference with Ether effects"""
        z = self.reparameterize(mu, logvar)
        
        if mode == 'standard':
            return z
        
        # Base Ether wave
        wave = torch.sin(z * 2 * np.pi) * 0.1
        
        if mode == 'ether':
            return z + wave
        
        elif mode == 'resonetics':
            # Advanced entropy-aware control
            std = torch.exp(0.5 * logvar)
            
            # 1. Global entropy level
            global_entropy = torch.mean(std, dim=1, keepdim=True)
            
            # 2. Local entropy patterns
            entropy_pattern = std / (std.mean(dim=1, keepdim=True) + 1e-8)
            
            # 3. Adaptive wave modulation
            # Low entropy → Strong creative influence
            # High entropy → Conservative stability
            entropy_factor = torch.sigmoid((0.5 - global_entropy) * 8.0)
            
            # Pattern-aware modulation
            wave_modulated = wave * entropy_factor * entropy_pattern
            
            return z + wave_modulated
        
        return z

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

# ---------------------------------------------------------
# [Advanced Scientific Metrics]
# ---------------------------------------------------------
class VAEMetrics:
    """Comprehensive VAE evaluation metrics"""
    
    @staticmethod
    def latent_smoothness(model, mu, logvar, mode='standard', n_probes=50, radius=0.1):
        """
        Measures local smoothness by probing in multiple directions
        Returns: mean MSE, std of MSEs (consistency)
        """
        with torch.no_grad():
            z_base = model.generate_with_ether(mu, logvar, mode=mode)
            recon_base = model.dec(z_base)
            
            mses = []
            for _ in range(n_probes):
                # Random direction on hypersphere
                direction = torch.randn_like(z_base)
                direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)
                
                z_pert = z_base + direction * radius
                recon_pert = model.dec(z_pert)
                
                mse = F.mse_loss(recon_base, recon_pert).item()
                mses.append(mse)
            
            return np.mean(mses), np.std(mses)
    
    @staticmethod
    def global_manifold_quality(model, n_samples=1000):
        """
        Evaluates the overall quality of the learned manifold
        - Reconstruction consistency across space
        - Latent space coverage
        """
        with torch.no_grad():
            # Sample diverse latent points
            z_samples = torch.randn(n_samples, model.ether_dim)
            recon_samples = model.dec(z_samples)
            
            # 1. Reconstruction diversity
            recon_std = recon_samples.std(dim=0).mean().item()
            
            # 2. Latent utilization (how well space is used)
            z_norms = z_samples.norm(dim=1)
            norm_stats = {
                'mean': z_norms.mean().item(),
                'std': z_norms.std().item(),
                'cv': (z_norms.std() / z_norms.mean()).item()  # Coefficient of variation
            }
            
            return {'recon_diversity': recon_std, 'latent_stats': norm_stats}
    
    @staticmethod
    def entropy_response_analysis(model, entropy_range=(-3, 3), n_levels=20):
        """
        Analyzes how each mode responds to different entropy levels
        Returns response curves for comparison
        """
        device = next(model.parameters()).device
        entropy_levels = torch.linspace(entropy_range[0], entropy_range[1], n_levels)
        
        responses = {'standard': [], 'ether': [], 'resonetics': []}
        
        for entropy in entropy_levels:
            mu = torch.zeros(1, model.ether_dim).to(device)
            logvar = torch.full((1, model.ether_dim), entropy).to(device)
            
            with torch.no_grad():
                # Get reconstructions for each mode
                recons = {}
                for mode in responses.keys():
                    z = model.generate_with_ether(mu, logvar, mode=mode)
                    recon = model.dec(z)
                    recons[mode] = recon
                
                # Calculate differences from standard baseline
                base = recons['standard']
                for mode in ['ether', 'resonetics']:
                    diff = F.mse_loss(recons[mode], base).item()
                    responses[mode].append(diff)
        
        return entropy_levels.numpy(), responses

# ---------------------------------------------------------
# [Enhanced Benchmark Runner]
# ---------------------------------------------------------
def run_comprehensive_benchmark():
    """Main benchmarking function with scientific rigor"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Running on: {device}")
    
    # 1. Model Training with validation
    print("\n" + "="*50)
    print("🚀 PHASE 1: MODEL TRAINING")
    print("="*50)
    
    # Load MNIST
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))
    ])
    
    dataset = torchvision.datasets.MNIST('.', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128)
    
    # Initialize model
    model = EtherVAE(ether_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Training loop with validation
    best_val_loss = float('inf')
    for epoch in range(15):
        # Training
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            
            # VAE loss
            recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss = F.binary_cross_entropy(recon, x, reduction='sum') + \
                       -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader.dataset)
        avg_val = val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:2d}: Train={avg_train:.4f}, Val={avg_val:.4f}", 
              end="")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'best_model.pth')
            print(" ✓")
        else:
            print()
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # 2. Scientific Analysis
    print("\n" + "="*50)
    print("🔬 PHASE 2: SCIENTIFIC ANALYSIS")
    print("="*50)
    
    metrics = VAEMetrics()
    
    # A. Entropy Response Analysis
    print("\n📈 A. Entropy Response Curves")
    entropy_levels, responses = metrics.entropy_response_analysis(model)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for mode, values in responses.items():
        if mode != 'standard':
            plt.plot(entropy_levels, values, label=mode.capitalize(), linewidth=2)
    
    plt.xlabel('Log Variance (Entropy Level)')
    plt.ylabel('MSE Difference from Standard')
    plt.title('Ether Effect by Entropy Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # B. Smoothness Analysis
    print("\n🌀 B. Local Smoothness Analysis")
    
    # Test different latent configurations
    test_configs = [
        ('Low Entropy', torch.zeros(1, 32).to(device) * 0.1, torch.ones(1, 32).to(device) * -2.0),
        ('Medium Entropy', torch.randn(1, 32).to(device) * 0.5, torch.zeros(1, 32).to(device)),
        ('High Entropy', torch.randn(1, 32).to(device) * 1.0, torch.ones(1, 32).to(device) * 1.0),
    ]
    
    results = {}
    for name, mu, logvar in test_configs:
        config_results = {}
        for mode in ['standard', 'ether', 'resonetics']:
            smooth_mean, smooth_std = metrics.latent_smoothness(model, mu, logvar, mode)
            config_results[mode] = (smooth_mean, smooth_std)
        results[name] = config_results
    
    # Plot smoothness comparison
    plt.subplot(1, 2, 2)
    modes = ['standard', 'ether', 'resonetics']
    x = np.arange(len(test_configs))
    width = 0.25
    
    for i, mode in enumerate(modes):
        values = [results[config][mode][0] for config, _, _ in test_configs]
        plt.bar(x + i*width, values, width, label=mode.capitalize())
    
    plt.xlabel('Entropy Configuration')
    plt.ylabel('Local Smoothness (Lower = Smoother)')
    plt.title('Smoothness Across Configurations')
    plt.xticks(x + width, [name for name, _, _ in test_configs])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # C. Global Manifold Analysis
    print("\n🌐 C. Global Manifold Analysis")
    manifold_stats = metrics.global_manifold_quality(model, n_samples=1000)
    
    print(f"   Reconstruction Diversity: {manifold_stats['recon_diversity']:.4f}")
    print(f"   Latent Space Utilization:")
    print(f"     - Mean Norm: {manifold_stats['latent_stats']['mean']:.3f}")
    print(f"     - Std Norm: {manifold_stats['latent_stats']['std']:.3f}")
    print(f"     - CV: {manifold_stats['latent_stats']['cv']:.3f}")
    
    # 3. Visual Comparison
    print("\n" + "="*50)
    print("🎨 PHASE 3: VISUAL COMPARISON")
    print("="*50)
    
    # Generate comparative samples
    n_samples = 5
    fixed_mu = torch.randn(n_samples, 32).to(device)
    fixed_logvar = torch.zeros(n_samples, 32).to(device)
    
    fig, axes = plt.subplots(3, n_samples + 1, figsize=(14, 8))
    modes = ['standard', 'ether', 'resonetics']
    
    with torch.no_grad():
        for row, mode in enumerate(modes):
            # Mode label
            axes[row, 0].text(0.5, 0.5, mode.upper(), fontsize=14, fontweight='bold',
                            ha='center', va='center')
            axes[row, 0].axis('off')
            
            for col in range(n_samples):
                mu_s = fixed_mu[col:col+1]
                log_s = fixed_logvar[col:col+1]
                
                # Generate and reconstruct
                z = model.generate_with_ether(mu_s, log_s, mode=mode)
                recon = model.dec(z).view(28, 28).cpu()
                
                # Calculate metrics
                smooth_mean, smooth_std = metrics.latent_smoothness(model, mu_s, log_s, mode)
                
                # Display
                ax = axes[row, col + 1]
                ax.imshow(recon, cmap='gray')
                ax.set_title(f"Smooth: {smooth_mean:.4f}\n±{smooth_std:.4f}", 
                           fontsize=9)
                ax.axis('off')
    
    plt.suptitle('EtherVAE: Mode Comparison with Local Smoothness', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 4. Statistical Significance Testing
    print("\n" + "="*50)
    print("📊 PHASE 4: STATISTICAL ANALYSIS")
    print("="*50)
    
    # Collect data for statistical testing
    n_trials = 100
    smoothness_data = {mode: [] for mode in modes}
    
    with torch.no_grad():
        for _ in range(n_trials):
            mu = torch.randn(1, 32).to(device) * 0.5
            logvar = torch.randn(1, 32).to(device) * 0.5
            
            for mode in modes:
                smooth_mean, _ = metrics.latent_smoothness(model, mu, logvar, mode)
                smoothness_data[mode].append(smooth_mean)
    
    # Perform statistical tests
    print("\nStatistical Comparison of Smoothness:")
    for i, mode1 in enumerate(modes):
        for j, mode2 in enumerate(modes):
            if i < j:
                data1 = np.array(smoothness_data[mode1])
                data2 = np.array(smoothness_data[mode2])
                
                # T-test
                t_stat, p_value = stats.ttest_rel(data1, data2)
                
                # Effect size
                mean_diff = np.mean(data1) - np.mean(data2)
                std_pooled = np.sqrt((np.std(data1)**2 + np.std(data2)**2) / 2)
                cohens_d = mean_diff / std_pooled if std_pooled > 0 else 0
                
                print(f"\n  {mode1.upper()} vs {mode2.upper()}:")
                print(f"    Mean difference: {mean_diff:.6f}")
                print(f"    p-value: {p_value:.6f} {'**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
                print(f"    Cohen's d: {cohens_d:.3f}")
    
    print(f"\n{'='*70}")
    print(f"✅ COMPREHENSIVE BENCHMARK COMPLETE")
    print(f"{'='*70}")

# ---------------------------------------------------------
# [Main Execution]
# ---------------------------------------------------------
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the comprehensive benchmark
    run_comprehensive_benchmark()
