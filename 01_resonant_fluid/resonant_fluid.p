## License

**Open Source**: AGPL-3.0 License  
→ You can use, modify, and distribute this code for free,  
   but **any derivative work or software that uses Resonetic-Core must also be open-sourced under AGPL-3.0**.

**Commercial / Closed-Source Use**  
→ Want to use Resonetic-Core in a proprietary product, service, or internal tool without disclosing your source code?  
   You need a **Commercial License**.

   Contact: red1239109@gmail.com  
   Price: negotiable (starting from $10,000 USD per year)

> “Free as in freedom for the community.  
>  Paid as in beer for companies.”

Dual-license model (AGPL-3.0 + Commercial)  
© 2025 red1239109-cmd – All rights reserved for commercial use.



# Copyright (c) 2025 red1239109-cmd
# Licensed under AGPL-3.0. See LICENSE file for details.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =========================================================
# 1. Fluid Phase Encoder: Transform Velocity to Phase Field
# =========================================================
class FluidPhaseEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64):
        super().__init__()
        self.embed = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.phase_transform = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1)
        )

    def forward(self, velocity):
        # Safety check: Ensure input shape is (B, 2, H, W)
        if velocity.ndim == 4 and velocity.shape[-1] == 2:
            velocity = velocity.permute(0, 3, 1, 2)
            
        encoded = self.embed(velocity)
        return self.phase_transform(encoded)

# =========================================================
# 2. Singularity Detector: Detect Resonant Singularities (Auto-Resizing)
# =========================================================
class SingularityResonanceDetector(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Initial base resonance field
        self.base_resonance_field = nn.Parameter(torch.randn(1, hidden_dim, 16, 16))
        
        self.detector = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.InstanceNorm2d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//2, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, phase):
        # [Core Logic] Dynamic Resizing: Auto-adjust internal field to match input size
        current_size = phase.shape[-2:]
        if self.base_resonance_field.shape[-2:] != current_size:
            res_field = F.interpolate(
                self.base_resonance_field, size=current_size, mode='bilinear', align_corners=False
            )
        else:
            res_field = self.base_resonance_field
            
        interaction = phase * res_field 
        return self.detector(interaction)

# =========================================================
# 3. Resonance Convection: Energy Flow via Velocity
# =========================================================
class ResonanceConvection(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.vel_embed = nn.Conv2d(2, hidden_dim, kernel_size=1)
        
        # Lightweight interaction mixer
        self.interaction_mixer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )
        self.diffusion = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, phase_field, velocity):
        if velocity.ndim == 4 and velocity.shape[-1] == 2:
            velocity = velocity.permute(0, 3, 1, 2)
            
        vel_emb = self.vel_embed(velocity)
        interaction = phase_field * vel_emb 
        mixed = self.interaction_mixer(interaction)
        return self.diffusion(mixed)

# =========================================================
# 4. Full Model Assembly (Resonant Navier-Stokes)
# =========================================================
class ResonantNavierStokes(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FluidPhaseEncoder()
        self.detector = SingularityResonanceDetector()
        self.resonator = ResonanceConvection()

    def forward(self, velocity):
        phase_field = self.encoder(velocity)
        resonance = self.resonator(phase_field, velocity)
        singularity_map = self.detector(phase_field)
        return singularity_map, resonance

# =========================================================
# [Self-Running Demo] Execute this file to see results instantly.
# =========================================================
if __name__ == "__main__":
    print("🌊 [Resonant Fluid] Generating Vortex Field...")
    
    # 1. Generate Data (Vortex)
    size = 64
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    U = -Y ; V = X 
    
    velocity = np.stack([U, V], axis=-1)
    input_vel = torch.FloatTensor(velocity).unsqueeze(0) # (1, 64, 64, 2)

    # 2. Run Model
    model = ResonantNavierStokes()
    with torch.no_grad():
        singularity, resonance = model(input_vel)

    # 3. Visualize Results
    print("🎨 Visualizing Resonance...")
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Physical Reality
    ax[0].quiver(X[::4, ::4], Y[::4, ::4], U[::4, ::4], V[::4, ::4], color='black')
    ax[0].set_title("1. Physical Reality (Vortex)", fontsize=14)
    ax[0].set_aspect('equal')

    # AI Singularity Detection
    im1 = ax[1].imshow(singularity.squeeze(), cmap='magma', origin='lower')
    ax[1].set_title("2. Singularity Detection (Center)", fontsize=14)
    plt.colorbar(im1, ax=ax[1])

    # Resonance Energy Field
    res_mean = torch.mean(resonance, dim=1).squeeze()
    im2 = ax[2].imshow(res_mean, cmap='cyan', origin='lower')
    ax[2].set_title("3. Resonance Flow Field", fontsize=14)
    plt.colorbar(im2, ax=ax[2])

    plt.suptitle("Resonant Navier-Stokes: Unsupervised Discovery", fontsize=16)
    plt.tight_layout()
    plt.savefig("resonant_fluid_result.png")
    print("✅ Result saved as 'resonant_fluid_result.png'")
