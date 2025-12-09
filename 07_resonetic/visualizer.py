# Copyright (c) 2025 red1239109-cmd
# Licensed under AGPL-3.0. See LICENSE file for details.

"""
Resonetic Riemann Visualizer (Unit 07)
--------------------------------------
The mathematical core that visualizes the "Phase Vortex" and "Resonance Density"
of the Riemann Zeta function. This provides the ground truth for the 
Resonetics philosophy.

[Key Features]
1. High-Precision Zeta Calculation: Uses mpmath for 50-digit precision.
2. Phase Vortex Visualization: Shows the topological winding around zeros.
3. Resonance Density Map: Visualizes the 'Energy Ridge' on the critical line.
"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import os

class RiemannVisualizer:
    def __init__(self, precision=50):
        # Set precision (dps = decimal places)
        mp.mp.dps = precision
        print(f"🌌 Riemann Visualizer (Unit 07) Initialized (Precision: {precision} dps)")
        
        # Known non-trivial zeros (Imaginary part) for reference
        self.known_zeros_t = [14.134725, 21.022039, 25.010857, 30.424876, 32.935061]

    def zeta_grid(self, sigma_vals, t_vals):
        """
        Compute Riemann Zeta on a complex grid.
        Note: mpmath is precise but slow. Progress is printed.
        """
        S, T = np.meshgrid(sigma_vals, t_vals)
        zeta_vals = np.empty(S.shape, dtype=complex)
        
        total = S.shape[0]
        print(f"▶ Computing Zeta Grid ({S.shape[0]}x{S.shape[1]})... This may take a moment.")
        
        for i in range(S.shape[0]):
            # Progress log every 10%
            if i % (total // 10) == 0:
                print(f"   Processing row {i}/{total}...")
                
            for j in range(S.shape[1]):
                # Create high-precision complex number s = sigma + i*t
                s = mp.mpf(S[i, j]) + 1j * mp.mpf(T[i, j])
                z = mp.zeta(s)
                zeta_vals[i, j] = complex(z.real, z.imag)
                
        return S, T, zeta_vals

    def plot_phase_vortex(self, save_path="riemann_phase_vortex.png"):
        """
        Visualizes the Argument (Phase) of Zeta(s).
        Identifying the 'Vortex' patterns around zeros.
        """
        print("\n🌀 Generating Phase Vortex Plot...")
        
        # Grid settings (Adjusted for visual clarity)
        sigma_vals = np.linspace(0.0, 1.0, 200)
        t_vals = np.linspace(0.0, 40.0, 400)
        
        S, T, zeta_vals = self.zeta_grid(sigma_vals, t_vals)
        
        # Phase calculation & Unwrapping
        phase = np.angle(zeta_vals)
        # Unwrap phase along the t-axis to show continuous vortex flow
        phase_unwrapped = np.unwrap(phase, axis=0) 
        
        fig, ax = plt.subplots(figsize=(10, 8))
        # cmap='twilight_shifted' is perfect for cyclic phase data
        im = ax.pcolormesh(S, T, phase_unwrapped, cmap="twilight_shifted", shading="auto")
        
        cbar = fig.colorbar(im, ax=ax, label="arg ζ(s) (Unwrapped Phase)")
        
        ax.set_title("Resonetic Phase Vortex: The Topology of Truth\n(Argument of ζ(s) on Critical Strip)", fontsize=14)
        ax.set_xlabel("Real Part ($\sigma$)")
        ax.set_ylabel("Imaginary Part ($t$)")
        
        # Mark Critical Line and Zeros
        ax.axvline(x=0.5, color="white", linestyle=":", alpha=0.5, label="Critical Line")
        for tz in self.known_zeros_t:
            ax.axhline(y=tz, color="yellow", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.scatter([0.5], [tz], color="red", s=40, zorder=5, edgecolor='black')
            
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✅ Saved: {save_path}")
        plt.close(fig)

    def plot_resonance_density(self, save_path="riemann_resonance_density.png"):
        """
        Visualizes the custom 'Resonance Density' function rho(s).
        rho(s) = Gaussian_Penalty / |zeta(s)|^2
        This shows zeros as infinite density peaks (singularities).
        """
        print("\nEx Generating Resonance Density Plot...")
        
        # Finer grid near critical line
        sigma_vals = np.linspace(0.0, 1.0, 200)
        t_vals = np.linspace(0.0, 40.0, 400)
        
        S, T, zeta_vals = self.zeta_grid(sigma_vals, t_vals)
        
        # Calculate Magnitude Squared
        abs_sq = np.abs(zeta_vals) ** 2
        eps = 1e-6 # Avoid division by zero
        
        # Gaussian Penalty to focus on the critical line (Resonetics Logic)
        k = 40.0 
        gaussian_penalty = np.exp(-k * (S - 0.5) ** 2)
        
        # Density Function rho(s)
        rho = gaussian_penalty / (abs_sq + eps)
        
        # Clip high values for better visualization contrast
        rho_clipped = np.clip(rho, 0, np.percentile(rho, 99))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.pcolormesh(S, T, rho_clipped, cmap="inferno", shading="auto")
        
        cbar = fig.colorbar(im, ax=ax, label="Resonance Density $\\rho(s)$")
        
        ax.set_title(r"Resonance Density Map: The Energy of Zeros", fontsize=14)
        ax.set_xlabel("Real Part ($\sigma$)")
        ax.set_ylabel("Imaginary Part ($t$)")
        
        # Mark Critical Line and Zeros
        ax.axvline(x=0.5, color="cyan", linestyle="--", alpha=0.7, label="Critical Line $\sigma=1/2$")
        for tz in self.known_zeros_t:
            # White dots for density peaks
            ax.scatter([0.5], [tz], color="white", s=30, edgecolor="black", zorder=5)
            
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✅ Saved: {save_path}")
        plt.close(fig)

# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    print("🌌 Initializing Unit 07: Riemann Mathematical Core")
    visualizer = RiemannVisualizer(precision=30) # Reduced precision slightly for speed
    
    visualizer.plot_phase_vortex()
    visualizer.plot_resonance_density()
    
    print("\n🏁 Resonance Visualization Complete.")
