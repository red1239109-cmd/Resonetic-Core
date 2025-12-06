import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# [Project Resonetics] 02. Burgers' Shock of Emotion
# "Emotion is a wave. And the moment the wave breaks, shock (Trauma) remains."
# ---------------------------------------------------------

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Physical constant (Viscosity)
# Smaller nu (closer to 0) leads to sharper shock waves. (Emotion cuts sharply)
nu = 0.01 / np.pi

# 1. Physics-Informed Neural Network (PINN) Model
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (t, x) -> Output: u (Velocity/Emotion Intensity)
        # Modified layers to be deeper and wider (to learn complex shock waves)
        self.net = nn.Sequential(
            nn.Linear(2, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 1)
        )

    def forward(self, tx):
        return self.net(tx)

# 2. Physics Law (Burgers' Equation Residual)
# u_t + u*u_x - nu*u_xx = 0
def pde_residual(model, tx):
    tx.requires_grad_(True)
    u = model(tx)
    
    # Automatic Differentiation
    # create_graph=True is mandatory (required for second-order differentiation)
    grads = torch.autograd.grad(u, tx, torch.ones_like(u), create_graph=True)[0]
    u_t = grads[:, 0:1]
    u_x = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
    
    # Return Residual
    return u_t + u * u_x - nu * u_xx

# ---------------------------------------------------------
# [Generate Training Data]
# ---------------------------------------------------------
def get_training_data(n_physics=5000, n_bc=500): # Increased data points
    # 1. PDE Collocation Points (For learning physics laws)
    tx_physics = torch.rand(n_physics, 2)
    tx_physics[:, 1] = tx_physics[:, 1] * 2 - 1 # x range [-1, 1]
    
    # 2. Initial Condition (t=0): -sin(pi*x)
    x_ic = torch.linspace(-1, 1, n_bc).view(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    tx_ic = torch.cat([t_ic, x_ic], dim=1)
    u_ic = -torch.sin(np.pi * x_ic)
    
    # 3. Boundary Condition (x=-1, x=1): u=0
    t_bc = torch.rand(n_bc, 1)
    x_bc_left = torch.ones_like(t_bc) * -1
    x_bc_right = torch.ones_like(t_bc) * 1
    
    tx_bc = torch.cat([
        torch.cat([t_bc, x_bc_left], dim=1),
        torch.cat([t_bc, x_bc_right], dim=1)
    ], dim=0)
    u_bc = torch.zeros(len(tx_bc), 1)
    
    return tx_physics, tx_ic, u_ic, tx_bc, u_bc

# ---------------------------------------------------------
# [Execution and Visualization]
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🌊 [Shock Simulation] Starting Burgers' Equation training...")
    print("   (This creates a 'shock wave' at t=0.5)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    tx_physics, tx_ic, u_ic, tx_bc, u_bc = get_training_data()
    
    # Move to Device
    tx_physics = tx_physics.to(device)
    tx_ic, u_ic = tx_ic.to(device), u_ic.to(device)
    tx_bc, u_bc = tx_bc.to(device), u_bc.to(device)
    
    # Training Loop (Epochs significantly increased)
    epochs = 15000 
    
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        
        loss_physics = pde_residual(model, tx_physics).pow(2).mean()
        loss_ic = (model(tx_ic) - u_ic).pow(2).mean()
        loss_bc = (model(tx_bc) - u_bc).pow(2).mean()
        
        # [IMPORTANT] Increase weights for boundary conditions (IC, BC). (Fixing initial state)
        total_loss = loss_physics + (loss_ic * 10) + (loss_bc * 10)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{epochs} | Total Loss: {total_loss.item():.6f} (Phy: {loss_physics.item():.6f})")

    print("✅ Training complete! Visualizing results...")

    # Visualization
    model.cpu()
    t = np.linspace(0, 1, 100)
    x = np.linspace(-1, 1, 256)
    T, X = np.meshgrid(t, x)
    TX = torch.tensor(np.stack([T.flatten(), X.flatten()], axis=1), dtype=torch.float32)
    
    with torch.no_grad():
        U = model(TX).numpy().reshape(T.shape)

    # Plot 1: Heatmap
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121)
    h = ax1.pcolormesh(T, X, U, cmap='seismic', shading='auto')
    fig.colorbar(h, ax=ax1, label='Emotion Intensity (u)')
    ax1.set_title("Space-Time Heatmap of Trauma")
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Space (x)')
    ax1.axvline(x=0.5, color='yellow', linestyle='--', alpha=0.7, label='Shock Formation')
    ax1.legend()

    # Plot 2: Snapshots (Wave Breaking Process)
    ax2 = fig.add_subplot(122)
    times = [0.0, 0.25, 0.5, 0.75]
    colors = ['blue', 'green', 'orange', 'red']
    
    for t_val, color in zip(times, colors):
        # Extract x values corresponding to specific time t
        t_idx = int(t_val * 99) # index mapping
        u_slice = U[:, t_idx]
        ax2.plot(x, u_slice, label=f't={t_val}', color=color, linewidth=2)

    ax2.set_title("The Moment the Wave Breaks")
    ax2.set_xlabel('Space (x)')
    ax2.set_ylabel('Intensity (u)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('burgers_shock_result.png')
    print("📊 Result image saved: burgers_shock_result.png")
