# ==============================================================================
# File: resonetics_v3_3_final.py
# Project: Resonetics (The Entropy Gardener)
# Version: 3.3 (Final Production Release)
# Description: A physics-informed, bio-mimetic AI simulation.
#              Agents fight against entropy using Lamarckian evolution.
#
# Key Features:
#   1. Vectorized Physics (NumPy optimized cleaning)
#   2. Correct Coordinate System (Row/Col vs X/Y fix)
#   3. Phase-Based Architecture (Plan -> Conflict -> Commit)
#   4. CSV Data Logging (Real-time tracking)
#   5. Suicide Prevention & Energy-Based Conflict Resolution
#
# License: AGPL-3.0
# ==============================================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
import os

# ---------------------------------------------------------
# [Configuration]
# ---------------------------------------------------------
CONFIG = {
    "grid_size": 60,
    "init_agents": 25,
    "view_size": 7,          # Local vision window (odd number)
    "feature_dim": 8,        # CNN feature maps
    "hidden_dim": 16,        # LSTM hidden state size
    "mutation_rate": 0.02,
    "move_cost": 0.5,
    "clean_reward": 2.5,     # Energy gained per unit of entropy removed
    "reproduce_cost": 40.0,
    "reproduce_threshold": 80.0,
    "min_population": 5,
    "max_population": 80,
    "log_file": "resonetics_data.csv"
}

# ---------------------------------------------------------
# [Core Intelligence: CNN + LSTM]
# ---------------------------------------------------------
class CompactBrain(nn.Module):
    """
    The brain of the agent.
    - Visual Cortex: CNN to process the 7x7 grid view.
    - Memory: LSTM to retain temporal context.
    - Policy: Linear layer to decide the next action.
    """
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(1, CONFIG["feature_dim"], 3), # 7x7 -> 5x5
            nn.ReLU(),
            nn.MaxPool2d(2), # 5x5 -> 2x2
            nn.Flatten()
        )
        # 2x2 * feature_dim
        self.fc_dim = CONFIG["feature_dim"] * 4
        self.lstm = nn.LSTM(self.fc_dim, CONFIG["hidden_dim"], batch_first=True)
        self.policy = nn.Linear(CONFIG["hidden_dim"], 6) # Up, Down, Left, Right, Clean, Repro

    def forward(self, x, hidden):
        # x shape: (1, 1, 7, 7)
        feat = self.visual(x).unsqueeze(1) # Add seq len dim: (1, 1, feature_dim)
        out, new_hidden = self.lstm(feat, hidden)
        logits = self.policy(out[:, -1, :])
        return torch.softmax(logits, dim=-1), new_hidden

class Agent:
    def __init__(self, id, row, col):
        self.id = id
        self.row = row  # Vertical position (matrix row)
        self.col = col  # Horizontal position (matrix col)
        self.brain = CompactBrain()
        self.hidden = None  # LSTM state
        self.energy = 50.0
        self.age = 0
        self.planned_action = 5 # Default: Idle

    def perceive(self, view_tensor):
        """Processes visual input and determines the next action."""
        with torch.no_grad():
            # Explicit LSTM Initialization for safety
            if self.hidden is None:
                h0 = torch.zeros(1, 1, CONFIG["hidden_dim"])
                c0 = torch.zeros(1, 1, CONFIG["hidden_dim"])
                self.hidden = (h0, c0)
            
            probs, self.hidden = self.brain(view_tensor, self.hidden)
            self.planned_action = torch.argmax(probs).item()

# ---------------------------------------------------------
# [The Engine]
# ---------------------------------------------------------
class ResoneticsEngine:
    def __init__(self):
        self.size = CONFIG["grid_size"]
        # World Grid: 0.0 (Order) to 1.0 (Chaos/Entropy)
        self.grid = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Initialize Agents
        self.agents = [
            Agent(i, np.random.randint(0, self.size), np.random.randint(0, self.size)) 
            for i in range(CONFIG["init_agents"])
        ]
        self.id_counter = CONFIG["init_agents"]
        self.gen = 0

    def get_view(self, row, col):
        """Extracts a local view with periodic boundary conditions (Torus)."""
        s = CONFIG["view_size"] // 2
        rows = np.arange(row - s, row + s + 1) % self.size
        cols = np.arange(col - s, col + s + 1) % self.size
        # Numpy fancy indexing
        view = self.grid[np.ix_(rows, cols)]
        return torch.from_numpy(view).float().unsqueeze(0).unsqueeze(0)

    def step(self):
        """Executes one simulation step using Phase-Based Logic."""
        
        # --- Phase 0: Vitality Check & Respawn ---
        self.agents = [a for a in self.agents if a.energy > 0]
        
        # Conservation of Life (Minimum Population)
        if len(self.agents) < CONFIG["min_population"]:
            missing = CONFIG["min_population"] - len(self.agents)
            for _ in range(missing):
                self.id_counter += 1
                self.agents.append(Agent(self.id_counter, 
                                         np.random.randint(0, self.size), 
                                         np.random.randint(0, self.size)))

        # --- Phase 1: Planning (Read-Only) ---
        moves = defaultdict(list) # Target (row, col) -> [Agents]
        cleaners = []
        breeders = []

        for agent in self.agents:
            # Suicide Prevention: Force Idle if energy is too low to move
            if agent.energy <= CONFIG["move_cost"] + 0.1:
                agent.planned_action = 5 
            else:
                agent.perceive(self.get_view(agent.row, agent.col))

            act = agent.planned_action
            target = (agent.row, agent.col)

            # Action Mapping: 0:Up, 1:Down, 2:Left, 3:Right
            if act < 4: 
                d_row, d_col = [(-1,0), (1,0), (0,-1), (0,1)][act]
                target = ((agent.row + d_row) % self.size, (agent.col + d_col) % self.size)
                moves[target].append(agent)
            elif act == 4: # Clean
                moves[target].append(agent)
                cleaners.append(agent)
            elif act == 5: # Reproduce
                moves[target].append(agent)
                if agent.energy > CONFIG["reproduce_threshold"]:
                    breeders.append(agent)

        # --- Phase 2: Conflict Resolution (Move) ---
        new_positions = {}
        for pos, candidates in moves.items():
            # Survival of the Fittest: Highest energy agent takes the spot
            winner = max(candidates, key=lambda a: a.energy)
            new_positions[winner.id] = pos
            
            # Losers get a collision penalty
            for loser in candidates:
                if loser != winner:
                    loser.energy -= 0.1 

        for agent in self.agents:
            if agent.id in new_positions:
                nr, nc = new_positions[agent.id]
                # Apply move cost only if position actually changed
                if (nr, nc) != (agent.row, agent.col):
                    agent.energy -= CONFIG["move_cost"]
                agent.row, agent.col = nr, nc
            
            agent.energy -= 0.1 # Metabolic Tax (Time cost)
            agent.age += 1

        # --- Phase 3: Physics (Vectorized Update) ---
        # 1. Natural Entropy Increase (Thermodynamics)
        noise = np.random.rand(self.size, self.size) * 0.005
        self.grid = np.clip(self.grid + noise, 0, 1)

        # 2. Agent Cleaning (Vectorized)
        if cleaners:
            clean_mask = np.zeros_like(self.grid)
            cleaner_count_map = defaultdict(int)
            
            for agent in cleaners:
                clean_mask[agent.row, agent.col] = 1
                cleaner_count_map[(agent.row, agent.col)] += 1

            # Determine how much dirt can be removed (Max 0.5 per step)
            dirt_to_clean = np.minimum(self.grid, 0.5) * clean_mask
            
            # Update Grid
            self.grid = np.clip(self.grid - dirt_to_clean, 0, 1)

            # Distribute Energy Rewards
            for agent in cleaners:
                # Share reward if multiple agents clean the same spot
                count = cleaner_count_map[(agent.row, agent.col)]
                reward = dirt_to_clean[agent.row, agent.col] * CONFIG["clean_reward"]
                agent.energy += reward / count 
        
        # --- Phase 4: Reproduction (Lamarckian Evolution) ---
        if len(self.agents) < CONFIG["max_population"]:
            # Prioritize high-energy parents
            breeders.sort(key=lambda a: a.energy, reverse=True)
            
            for parent in breeders:
                if len(self.agents) >= CONFIG["max_population"]: break
                
                if parent.energy > CONFIG["reproduce_cost"]:
                    parent.energy -= CONFIG["reproduce_cost"]
                    self.id_counter += 1
                    # Spawn child at same location
                    child = Agent(self.id_counter, parent.row, parent.col)
                    
                    # Direct Weight Copy (Fast & Memory Efficient)
                    with torch.no_grad():
                        for cp, pp in zip(child.brain.parameters(), parent.brain.parameters()):
                            cp.data.copy_(pp.data)
                            cp.add_(torch.randn_like(cp) * CONFIG["mutation_rate"])
                    
                    self.agents.append(child)

        self.gen += 1
        return self.grid, self.agents

# ---------------------------------------------------------
# [Visualization & Logging]
# ---------------------------------------------------------
def run_simulation():
    engine = ResoneticsEngine()
    
    # Initialize CSV Log
    if os.path.exists(CONFIG["log_file"]): os.remove(CONFIG["log_file"])
    with open(CONFIG["log_file"], 'w') as f:
        f.write("Gen,Population,Avg_Energy,Total_Entropy\n")

    # Setup Plot
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Render Grid (Magma: Black=Order, Red/Yellow=Chaos)
    im = ax.imshow(engine.grid, cmap='magma', vmin=0, vmax=1, animated=True)
    
    # Render Agents
    # Note: scatter(x, y) expects (col, row)
    scatter = ax.scatter([], [], s=40, edgecolors='white', animated=True)
    
    # Status Text
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', 
                   fontsize=12, fontweight='bold',
                   bbox=dict(facecolor='black', alpha=0.5))
    
    def update(frame):
        grid, agents = engine.step()
        im.set_data(grid)
        
        if agents:
            # Correct Coordinate Mapping: [col, row] for Scatter
            offsets = np.array([[a.col, a.row] for a in agents])
            
            # Color based on energy (Red=Low, Green=High)
            energies = np.array([a.energy for a in agents])
            normalized_e = np.clip(energies / 100, 0, 1)
            colors = plt.cm.RdYlGn(normalized_e)
            
            scatter.set_offsets(offsets)
            scatter.set_facecolors(colors)
        else:
            scatter.set_offsets(np.zeros((0, 2)))

        # Stats Calculation
        avg_e = np.mean([a.energy for a in agents]) if agents else 0
        total_entropy = np.sum(grid)
        
        # CSV Logging
        with open(CONFIG["log_file"], 'a') as f:
            f.write(f"{engine.gen},{len(agents)},{avg_e:.2f},{total_entropy:.2f}\n")

        # HUD Update
        stats = (f"GENERATION: {engine.gen:04d}\n"
                 f"POPULATION: {len(agents):02d}\n"
                 f"AVG ENERGY: {avg_e:6.1f}\n"
                 f"ENTROPY   : {total_entropy:7.1f}")
        text.set_text(stats)
        
        return im, scatter, text

    print(f"🌱 [Resonetics v3.3] Simulation Started...")
    print(f"   - Logging data to: {CONFIG['log_file']}")
    print(f"   - Watch as agents learn to organize against chaos.")
    
    ani = animation.FuncAnimation(fig, update, frames=None, interval=50, blit=True)
    plt.title("Resonetics v3.3: The Entropy Gardener (Final Release)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_simulation()
