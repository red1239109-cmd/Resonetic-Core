# ==============================================================================
# File: resonetics_v3_3_2_final.py
# Project: Resonetics (The Entropy Gardener)
# Version: 3.3.2 (Robustness Patch)
# Description: A physics-informed, bio-mimetic AI simulation.
#              Refined with safety nets based on code review.
#
# Changes applied based on review:
#   - Added Try-Except blocks for simulation stability (Critical)
#   - Maintained dictionary config for script simplicity
#   - Kept O(N) Spatial Hashing (defaultdict) as it is optimal for N<1000
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
import sys
import traceback

# ---------------------------------------------------------
# [Configuration]
# ---------------------------------------------------------
CONFIG = {
    "grid_size": 60,
    "init_agents": 25,
    "view_size": 7,
    "feature_dim": 8,
    "hidden_dim": 16,
    "mutation_rate": 0.02,
    "move_cost": 0.5,
    "emergency_move_cost": 0.1,
    "clean_reward": 3.0,
    "reproduce_cost": 40.0,
    "reproduce_threshold": 80.0,
    "min_population": 15,
    "max_population": 80,
    "log_file": "resonetics_data.csv",
    "log_buffer_size": 50
}

# ---------------------------------------------------------
# [Core Intelligence]
# ---------------------------------------------------------
class CompactBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(1, CONFIG["feature_dim"], 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc_dim = CONFIG["feature_dim"] * 4
        self.lstm = nn.LSTM(self.fc_dim, CONFIG["hidden_dim"], batch_first=True)
        self.policy = nn.Linear(CONFIG["hidden_dim"], 6)

    def forward(self, x, hidden):
        feat = self.visual(x).unsqueeze(1)
        out, new_hidden = self.lstm(feat, hidden)
        logits = self.policy(out[:, -1, :])
        return torch.softmax(logits, dim=-1), new_hidden

class Agent:
    def __init__(self, id, row, col):
        self.id = id
        self.row = row
        self.col = col
        self.brain = CompactBrain()
        self.hidden = None
        self.energy = 50.0
        self.age = 0
        self.planned_action = 5
        self.is_emergency_move = False

    def perceive(self, view_tensor):
        with torch.no_grad():
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
        self.grid = np.zeros((self.size, self.size), dtype=np.float32)
        self.agents = [
            Agent(i, np.random.randint(0, self.size), np.random.randint(0, self.size)) 
            for i in range(CONFIG["init_agents"])
        ]
        self.id_counter = CONFIG["init_agents"]
        self.gen = 0
        self.log_buffer = []
        self.emergency_moves = 0
        self.isolation_saves = 0

    def get_view(self, row, col):
        s = CONFIG["view_size"] // 2
        rows = np.arange(row - s, row + s + 1) % self.size
        cols = np.arange(col - s, col + s + 1) % self.size
        view = self.grid[np.ix_(rows, cols)]
        return torch.from_numpy(view).float().unsqueeze(0).unsqueeze(0)

    def find_dirty_neighbor(self, agent):
        best_action = None
        best_dirt = 0.0
        for action, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
            nr, nc = (agent.row + dr) % self.size, (agent.col + dc) % self.size
            dirt = self.grid[nr, nc]
            if dirt > best_dirt:
                best_dirt = dirt
                best_action = action
        return best_action, best_dirt

    def step(self):
        # [Robustness] Wrap logic in try-except to prevent simulation crash
        try:
            self.emergency_moves = 0
            
            # Phase 0: Vitality
            self.agents = [a for a in self.agents if a.energy > 0]
            if len(self.agents) < CONFIG["min_population"]:
                missing = CONFIG["min_population"] - len(self.agents)
                for _ in range(missing):
                    self.id_counter += 1
                    self.agents.append(Agent(self.id_counter, 
                                           np.random.randint(0, self.size), 
                                           np.random.randint(0, self.size)))

            # Phase 1: Planning
            moves = defaultdict(list)
            cleaners = []
            breeders = []

            for agent in self.agents:
                agent.is_emergency_move = False
                
                # Smart Survival Logic
                if agent.energy <= CONFIG["move_cost"] + 0.1:
                    current_dirt = self.grid[agent.row, agent.col]
                    if current_dirt > 0.1:
                        agent.planned_action = 4
                    elif agent.energy >= CONFIG["emergency_move_cost"] + 0.1:
                        best_dir, best_dirt = self.find_dirty_neighbor(agent)
                        if best_dir is not None and best_dirt > 0.2:
                            agent.planned_action = best_dir
                            agent.is_emergency_move = True
                            self.emergency_moves += 1
                            self.isolation_saves += 1
                        else:
                            agent.planned_action = 5
                    else:
                        agent.planned_action = 5
                else:
                    agent.perceive(self.get_view(agent.row, agent.col))

                act = agent.planned_action
                target = (agent.row, agent.col)

                if act < 4: 
                    d_row, d_col = [(-1,0), (1,0), (0,-1), (0,1)][act]
                    target = ((agent.row + d_row) % self.size, (agent.col + d_col) % self.size)
                    moves[target].append(agent)
                elif act == 4:
                    moves[target].append(agent)
                    cleaners.append(agent)
                elif act == 5:
                    moves[target].append(agent)
                    if agent.energy > CONFIG["reproduce_threshold"]:
                        breeders.append(agent)

            # Phase 2: Conflict & Move
            new_positions = {}
            for pos, candidates in moves.items():
                # [Optimization] O(N) Spatial Hashing (defaultdict) is already used here.
                # No need for Quadtree as N is small (<100).
                winner = max(candidates, key=lambda a: a.energy)
                new_positions[winner.id] = pos
                for loser in candidates:
                    if loser != winner: loser.energy -= 0.1

            for agent in self.agents:
                if agent.id in new_positions:
                    nr, nc = new_positions[agent.id]
                    if (nr, nc) != (agent.row, agent.col):
                        cost = CONFIG["emergency_move_cost"] if agent.is_emergency_move else CONFIG["move_cost"]
                        agent.energy -= cost
                    agent.row, agent.col = nr, nc
                agent.energy -= 0.1
                agent.age += 1

            # Phase 3: Physics
            noise = np.random.rand(self.size, self.size) * 0.005
            self.grid = np.clip(self.grid + noise, 0, 1)

            if cleaners:
                clean_mask = np.zeros_like(self.grid)
                count_map = defaultdict(int)
                for agent in cleaners:
                    clean_mask[agent.row, agent.col] = 1
                    count_map[(agent.row, agent.col)] += 1

                dirt_to_clean = np.minimum(self.grid, 0.5) * clean_mask
                self.grid = np.clip(self.grid - dirt_to_clean, 0, 1)

                for agent in cleaners:
                    count = count_map[(agent.row, agent.col)]
                    if count > 0: # Safety check
                        reward = dirt_to_clean[agent.row, agent.col] * CONFIG["clean_reward"]
                        agent.energy += reward / count
            
            # Phase 4: Reproduction
            if len(self.agents) < CONFIG["max_population"]:
                breeders.sort(key=lambda a: a.energy, reverse=True)
                for parent in breeders:
                    if len(self.agents) >= CONFIG["max_population"]: break
                    if parent.energy > CONFIG["reproduce_cost"]:
                        parent.energy -= CONFIG["reproduce_cost"]
                        self.id_counter += 1
                        child = Agent(self.id_counter, parent.row, parent.col)
                        
                        with torch.no_grad():
                            for cp, pp in zip(child.brain.parameters(), parent.brain.parameters()):
                                cp.data.copy_(pp.data)
                                fan_in = cp.size(0) if cp.ndim > 1 else 1
                                scale = CONFIG["mutation_rate"] / np.sqrt(max(fan_in, 1))
                                cp.add_(torch.randn_like(cp) * scale)
                        self.agents.append(child)

            self.gen += 1
            return self.grid, self.agents

        except Exception as e:
            print(f"🚨 Critical Error in Engine Step: {e}")
            traceback.print_exc()
            return self.grid, self.agents

    def log_to_buffer(self, avg_energy, total_entropy):
        self.log_buffer.append(
            f"{self.gen},{len(self.agents)},{avg_energy:.2f},{total_entropy:.2f},"
            f"{self.emergency_moves},{self.isolation_saves}\n"
        )
        if len(self.log_buffer) >= CONFIG["log_buffer_size"]:
            self.flush_buffer()
    
    def flush_buffer(self):
        if self.log_buffer:
            try:
                with open(CONFIG["log_file"], 'a') as f:
                    f.writelines(self.log_buffer)
                self.log_buffer.clear()
            except IOError as e:
                print(f"⚠️ Warning: Logging failed - {e}")

# ---------------------------------------------------------
# [Visualization]
# ---------------------------------------------------------
def run_simulation():
    engine = ResoneticsEngine()
    
    if os.path.exists(CONFIG["log_file"]): os.remove(CONFIG["log_file"])
    with open(CONFIG["log_file"], 'w') as f:
        f.write("Gen,Population,Avg_Energy,Total_Entropy,Emergency_Moves,Isolation_Saves\n")

    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    im = ax.imshow(engine.grid, cmap='magma', vmin=0, vmax=1, animated=True)
    scatter = ax.scatter([], [], s=40, edgecolors='white', animated=True)
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', 
                   fontsize=11, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6))
    
    def update(frame):
        grid, agents = engine.step()
        im.set_data(grid)
        
        if agents:
            offsets = np.array([[a.col, a.row] for a in agents])
            energies = np.array([a.energy for a in agents])
            normalized_e = np.clip(energies / 100, 0, 1)
            colors = plt.cm.RdYlGn(normalized_e)
            scatter.set_offsets(offsets)
            scatter.set_facecolors(colors)
        else:
            scatter.set_offsets(np.zeros((0, 2)))

        avg_e = np.mean([a.energy for a in agents]) if agents else 0
        total_entropy = np.sum(grid)
        engine.log_to_buffer(avg_e, total_entropy)

        stats = (f"GENERATION: {engine.gen:04d}\n"
                 f"POPULATION: {len(agents):02d}\n"
                 f"AVG ENERGY: {avg_e:6.1f}\n"
                 f"ENTROPY   : {total_entropy:7.1f}\n"
                 f"EMERGENCIES: {engine.emergency_moves:02d}\n"
                 f"SAVED      : {engine.isolation_saves:04d}")
        text.set_text(stats)
        return im, scatter, text

    print(f"🌱 [Resonetics v3.3.2] Simulation Started (Robustness Patch Applied)...")
    
    ani = animation.FuncAnimation(fig, update, frames=None, interval=50, blit=True)
    plt.title("Resonetics v3.3.2: The Entropy Gardener\n(Robustness Patch)")
    plt.axis('off')
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")
    finally:
        engine.flush_buffer()
        print(f"✅ Simulation ended cleanly.")
        print(f"📊 Total isolation saves: {engine.isolation_saves}")

if __name__ == "__main__":
    run_simulation()
