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

# Physical constant (Viscosity: smaller values lead to sharper shock waves)
nu = 0.01 / np.pi

# 1. Physics-Informed Neural Network (PINN) Model
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (t, x) -> Output: u (Velocity/Emotion Intensity)
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, tx):
        return self.net(tx)

# 2. Physics Law (Burgers' Equation Residual)
# u_t + u*u_x - nu*u_xx = 0
def pde_residual(model, tx):
    tx.requires_grad_(True)
    u = model(tx)
    
    # Automatic Differentiation
    grads = torch.autograd.grad(u, tx, torch.ones_like(u), create_graph=True)[0]
    u_t = grads[:, 0:1]
    u_x = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
    
    # Return Residual: Closer to 0 means better adherence to physical laws
    return u_t + u * u_x - nu * u_xx

# ---------------------------------------------------------
# [Generate Training Data: Initial & Boundary Conditions]
# ---------------------------------------------------------
def get_training_data(n_physics=2000, n_bc=100):
    # 1. PDE Collocation Points (For learning physics laws)
    tx_physics = torch.rand(n_physics, 2) # t=[0,1], x=[0,1]
    tx_physics[:, 1] = tx_physics[:, 1] * 2 - 1 # Expand x to [-1, 1]
    
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
    
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    tx_physics, tx_ic, u_ic, tx_bc, u_bc = get_training_data()
    
    # Training Loop
    epochs = 3000
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        
        # Calculate Loss
        loss_physics = pde_residual(model, tx_physics).pow(2).mean()
        loss_ic = (model(tx_ic) - u_ic).pow(2).mean()
        loss_bc = (model(tx_bc) - u_bc).pow(2).mean()
        
        total_loss = loss_physics + loss_ic + loss_bc
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {total_loss.item():.6f}")

    print("✅ Training complete! Visualizing results...")

    # Visualization (Spatiotemporal Graph)
    t = np.linspace(0, 1, 100)
    x = np.linspace(-1, 1, 256)
    T, X = np.meshgrid(t, x)
    TX = torch.tensor(np.stack([T.flatten(), X.flatten()], axis=1), dtype=torch.float32)
    
    with torch.no_grad():
        U = model(TX).numpy().reshape(T.shape)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T, X, U, cmap='jet', shading='auto')
    plt.colorbar(label='Emotion Intensity (u)')
    plt.title("Burgers' Shock of Emotion: When Feelings Collapse", fontsize=14)
    plt.xlabel('Time (t)')
    plt.ylabel('Space (x)')
    
    # Mark Shock Formation
    plt.axvline(x=0.5, color='white', linestyle='--', alpha=0.5, label='Shock Formation (Trauma)')
    plt.legend()
    
    plt.savefig('burgers_shock_result.png')
    print("📊 Result image saved: burgers_shock_result.png")
