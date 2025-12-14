# ==============================================================================

# File: resonetics_v3_3_3_complete.py

# Project: Resonetics (The Entropy Gardener)

# Version: 3.3.3 Complete (System Flow + Action Damping)

# Author: red1239109-cmd

# Copyright (c) 2025 Resonetics Project

#

# License: AGPL-3.0

#

# This program is free software: you can redistribute it and/or modify

# it under the terms of the GNU Affero General Public License as published by

# the Free Software Foundation, either version 3 of the License, or

# (at your option) any later version.

#

# This program is distributed in the hope that it will be useful,

# but WITHOUT ANY WARRANTY; without even the implied warranty of

# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the

# GNU Affero General Public License for more details.

#

# You should have received a copy of the GNU Affero General Public License

# along with this program. If not, see <https://www.gnu.org/licenses/>.

# ==============================================================================

"""

Resonetics: Artificial Ecosystem with System Flow Monitoring & Action Damping

A multi-agent simulation where agents clean entropy (dirt) while maintaining

system stability. Version 3.3.3 Complete integrates:

1. System Flow measurement - monitoring sensitivity to perturbations

2. Action Damping - risk/flow-based behavioral suppression

3. Survival mode - emergency response to near-collapse conditions

"""

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

# [Configuration] + SYSTEM FLOW + ACTION DAMPING

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

# Risk assessment configuration

CONFIG.update({

    "risk": {

        "ema_alpha": 0.15,

        "w_entropy_grad": 0.55,

        "w_pop_vol": 0.25,

        "w_emg_ratio": 0.20,

        "entropy_grad_scale": 15.0,

        "pop_vol_scale": 8.0,

        "collapse_th": 0.72,

        "bubble_th": 0.45

    }

})

# SYSTEM FLOW configuration

CONFIG.update({

    "flow": {

        "type": "system", # system flow monitoring

        "eps": 0.02, # perturbation scale for flow calculation

        "ema_alpha": 0.10, # EMA smoothing for flow

        "w_flow": 0.20, # flow weight in risk calculation

        "hot_th": 0.80, # hotspot threshold

        "clip_grid_noise": True # keep grid in [0,1] after perturbation

    }

})

# ACTION DAMPING configuration (Integrated from diff patch)

CONFIG.update({

    "control": {

        # alpha in [min_alpha, 1.0]; higher risk => lower alpha => stronger damping

        "min_alpha": 0.20,

        "risk_to_damping": 1.00, # multiplier on last_risk

        "flow_to_damping": 0.40, # extra damping from last_flow (NEW INTEGRATION)

        # survival triggers (use last_risk, last_flow, and last_verdict)

        "survival_on_verdict": True,

        "survival_on_risk": 0.78, # if last_risk >= this => force survival

        "survival_on_flow": 0.85, # if last_flow >= this => force survival (NEW)

        # what is "risky" in this env?

        "suppress_moves": True,

        "suppress_reproduce": True,

        # when suppressing, what to do?

        "safe_action_clean_threshold": 0.10, # if dirt > this => clean(4) else wait(5)

    }

})

# ---------------------------------------------------------

# [Core Intelligence]

# ---------------------------------------------------------

class CompactBrain(nn.Module):

    """

    Neural network brain for agents with LSTM memory.

    fc_dim is automatically inferred from input size.

    """

    def __init__(self):

        super().__init__()

        self.visual = nn.Sequential(

            nn.Conv2d(1, CONFIG["feature_dim"], 3),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Flatten()

        )

       

        # Infer fc_dim from dummy forward pass

        with torch.no_grad():

            vs = int(CONFIG["view_size"])

            dummy = torch.zeros(1, 1, vs, vs)

            flat = self.visual(dummy)

            self.fc_dim = int(flat.shape[-1])

       

        self.lstm = nn.LSTM(self.fc_dim, CONFIG["hidden_dim"], batch_first=True)

        self.policy = nn.Linear(CONFIG["hidden_dim"], 6)

    def forward(self, x, hidden):

        feat = self.visual(x).unsqueeze(1)

        out, new_hidden = self.lstm(feat, hidden)

        logits = self.policy(out[:, -1, :])

        return torch.softmax(logits, dim=-1), new_hidden

class Agent:

    """Individual agent with neural brain and energy system."""

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

        """Process visual input and decide action."""

        with torch.no_grad():

            if self.hidden is None:

                h0 = torch.zeros(1, 1, CONFIG["hidden_dim"])

                c0 = torch.zeros(1, 1, CONFIG["hidden_dim"])

                self.hidden = (h0, c0)

           

            probs, self.hidden = self.brain(view_tensor, self.hidden)

            self.planned_action = torch.argmax(probs).item()

# ---------------------------------------------------------

# [The Engine] + SYSTEM FLOW + ACTION DAMPING

# ---------------------------------------------------------

class ResoneticsEngine:

    """Main simulation engine with integrated flow monitoring and action damping."""

   

    def __init__(self, seed=None):

        self.size = CONFIG["grid_size"]

        self.grid = np.zeros((self.size, self.size), dtype=np.float32)

       

        # Random number generator

        if seed is not None:

            np.random.seed(seed)

            torch.manual_seed(seed)

        self.np_random = np.random

       

        # Initialize agents

        self.agents = [

            Agent(i, np.random.randint(0, self.size), np.random.randint(0, self.size))

            for i in range(CONFIG["init_agents"])

        ]

        self.id_counter = CONFIG["init_agents"]

        self.gen = 0

        self.log_buffer = []

        self.emergency_moves = 0

        self.isolation_saves = 0

       

        # Risk tracking

        self.prev_total_entropy = 0.0

        self.prev_population = len(self.agents)

        self.risk_ema = 0.0

        self.last_risk = 0.0

        self.last_verdict = "creative"

       

        # SYSTEM FLOW tracking

        self.flow_ema = 0.0

        self.last_flow = 0.0

    def get_view(self, row, col):

        """Get circular wrapped view around agent position."""

        s = CONFIG["view_size"] // 2

        rows = np.arange(row - s, row + s + 1) % self.size

        cols = np.arange(col - s, col + s + 1) % self.size

        view = self.grid[np.ix_(rows, cols)]

        return torch.from_numpy(view).float().unsqueeze(0).unsqueeze(0)

    def find_dirty_neighbor(self, agent):

        """Find the dirtiest neighboring cell for emergency moves."""

        best_action = None

        best_dirt = 0.0

        for action, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):

            nr, nc = (agent.row + dr) % self.size, (agent.col + dc) % self.size

            dirt = self.grid[nr, nc]

            if dirt > best_dirt:

                best_dirt = dirt

                best_action = action

        return best_action, best_dirt

    # ---------------------------------------------------------

    # Utility functions

    # ---------------------------------------------------------

    def _sigmoid(self, x: float) -> float:

        """Sigmoid activation function."""

        return 1.0 / (1.0 + np.exp(-x))

    # ---------------------------------------------------------

    # SYSTEM FLOW calculation

    # ---------------------------------------------------------

    def _compute_system_flow(self) -> float:

        """

        Calculate system flow - sensitivity of metrics to small perturbations.

       

        Flow measures how much the system metrics (entropy and hotspot rate)

        change in response to small noise added to the grid.

        This serves as a stability sensor for collapse prediction.

        """

        fconf = CONFIG["flow"]

        eps = float(fconf["eps"])

        hot_th = float(fconf["hot_th"])

        # Current metrics: entropy + hotspot_rate

        grid = self.grid

        entropy = float(np.sum(grid))

        hotspot_rate = float(np.mean(grid > hot_th))

        # Inject small perturbation (does not mutate actual grid)

        noise = np.random.randn(*grid.shape).astype(np.float32)

        grid2 = grid + eps * noise

        if fconf.get("clip_grid_noise", True):

            grid2 = np.clip(grid2, 0.0, 1.0)

        # Metrics after perturbation

        entropy2 = float(np.sum(grid2))

        hotspot_rate2 = float(np.mean(grid2 > hot_th))

        # Flow: sensitivity (Lipschitz-ish estimate)

        d_entropy = (entropy2 - entropy)

        d_hotspot = (hotspot_rate2 - hotspot_rate)

        # Normalize entropy delta by grid size

        n = float(grid.size)

        d_entropy_norm = d_entropy / max(1.0, n)

        # Compute flow as squared gradient normalized by epsilon

        flow = (d_entropy_norm * d_entropy_norm + d_hotspot * d_hotspot) / max(1e-9, (eps * eps))

        flow = float(np.clip(flow, 0.0, 1.0))

        # EMA smoothing

        a = float(fconf["ema_alpha"])

        self.flow_ema = (1.0 - a) * self.flow_ema + a * flow

        self.last_flow = float(np.clip(self.flow_ema, 0.0, 1.0))

        return self.last_flow

    # ---------------------------------------------------------

    # ACTION DAMPING control helpers (Integrated from diff patch)

    # ---------------------------------------------------------

    def _damping_alpha(self) -> float:

        """

        Compute damping alpha based on risk and flow.

        Returns alpha in [min_alpha, 1.0]

        Higher risk/flow => lower alpha => stronger damping

        """

        c = CONFIG.get("control", {})

        min_alpha = float(c.get("min_alpha", 0.20))

       

        # Risk contribution

        risk = float(np.clip(self.last_risk, 0.0, 1.0))

        risk_weight = float(c.get("risk_to_damping", 1.0))

        risk_contribution = risk_weight * risk

       

        # Flow contribution (NEW INTEGRATION)

        flow = float(np.clip(self.last_flow, 0.0, 2.0)) # flow can be >1

        flow_weight = float(c.get("flow_to_damping", 0.4))

        flow_contribution = flow_weight * (flow / 2.0) # Normalize to [0,1]

       

        # Combined damping factor

        damping = risk_contribution + flow_contribution

        damping = float(np.clip(damping, 0.0, 1.0))

       

        # Convert to alpha (damping 0 => alpha 1.0, damping 1 => alpha min_alpha)

        alpha = 1.0 - (1.0 - min_alpha) * damping

        return float(np.clip(alpha, min_alpha, 1.0))

    def _force_survival_now(self) -> bool:

        """

        Determine if survival mode should be forced.

        Considers verdict, risk, and flow (NEW).

        """

        c = CONFIG.get("control", {})

       

        # Verdict-based trigger

        if bool(c.get("survival_on_verdict", True)) and self.last_verdict in ("collapse",):

            return True

       

        # Risk-based trigger

        if float(self.last_risk) >= float(c.get("survival_on_risk", 0.78)):

            return True

       

        # Flow-based trigger (NEW INTEGRATION)

        if float(self.last_flow) >= float(c.get("survival_on_flow", 0.85)):

            return True

       

        # Combined risk-flow trigger (extra safety)

        if self.last_risk > 0.6 and self.last_flow > 0.7:

            return True

           

        return False

    def _safe_action(self, agent) -> int:

        """

        Return a safe action for the agent based on local dirt level.

        4 = clean, 5 = wait (no-op)

        """

        c = CONFIG.get("control", {})

        dirt = float(self.grid[agent.row, agent.col])

        threshold = float(c.get("safe_action_clean_threshold", 0.10))

       

        if dirt > threshold:

            return 4 # clean

        return 5 # wait / no-op

    # ---------------------------------------------------------

    # Risk and verdict calculation with flow integration

    # ---------------------------------------------------------

    def _compute_risk_and_verdict(self, entropy_gradient: float,

                                  pop_volatility: int, emergency_ratio: float):

        """

        Compute system risk level and verdict with flow integration.

       

        Returns:

            tuple: (risk_score, verdict_string, flow_value)

        """

        rconf = CONFIG["risk"]

       

        # Normalize components

        eg = abs(entropy_gradient) / max(1e-6, rconf["entropy_grad_scale"])

        pv = float(pop_volatility) / max(1e-6, rconf["pop_vol_scale"])

        er = float(emergency_ratio)

       

        # Sigmoid transforms

        eg_s = self._sigmoid(3.0 * (eg - 0.5))

        pv_s = self._sigmoid(3.0 * (pv - 0.5))

        er_s = self._sigmoid(4.0 * (er - 0.25))

        # Base risk

        raw = (

            rconf["w_entropy_grad"] * eg_s +

            rconf["w_pop_vol"] * pv_s +

            rconf["w_emg_ratio"] * er_s

        )

        # INTEGRATION: Incorporate flow into risk calculation

        flow = self._compute_system_flow() # 0..1 (ema)

        w_flow = float(CONFIG["flow"]["w_flow"])

        raw = float(np.clip(raw + w_flow * flow, 0.0, 1.0))

        # EMA smoothing

        a = rconf["ema_alpha"]

        self.risk_ema = (1 - a) * self.risk_ema + a * raw

        risk = float(np.clip(self.risk_ema, 0.0, 1.0))

        # Determine verdict

        if risk >= rconf["collapse_th"]:

            verdict = "collapse"

        elif risk >= rconf["bubble_th"]:

            verdict = "bubble"

        else:

            verdict = "creative"

        self.last_risk = risk

        self.last_verdict = verdict

       

        return risk, verdict, float(flow)

    # ---------------------------------------------------------

    # Main simulation step with action damping

    # ---------------------------------------------------------

    def step(self):

        """Execute one simulation step with integrated flow-aware action damping."""

        try:

            self.emergency_moves = 0

           

            # Remove dead agents

            self.agents = [a for a in self.agents if a.energy > 0]

           

            # Maintain minimum population

            if len(self.agents) < CONFIG["min_population"]:

                missing = CONFIG["min_population"] - len(self.agents)

                for _ in range(missing):

                    self.id_counter += 1

                    self.agents.append(Agent(

                        self.id_counter,

                        np.random.randint(0, self.size),

                        np.random.randint(0, self.size)

                    ))

            # --- INTEGRATION: Control policy (computed once per step) ---

            alpha = self._damping_alpha() # Risk + Flow aware

            survival_mode = self._force_survival_now() # Risk + Flow aware

           

            # Collect agent actions

            moves = defaultdict(list)

            cleaners = []

            breeders = []

           

            for agent in self.agents:

                agent.is_emergency_move = False

               

                # Emergency behavior when energy is low

                if agent.energy <= CONFIG["move_cost"] + 0.1:

                    current_dirt = self.grid[agent.row, agent.col]

                    if current_dirt > 0.1:

                        agent.planned_action = 4 # Clean

                    elif agent.energy >= CONFIG["emergency_move_cost"] + 0.1:

                        best_dir, best_dirt = self.find_dirty_neighbor(agent)

                        if best_dir is not None and best_dirt > 0.2:

                            agent.planned_action = best_dir

                            agent.is_emergency_move = True

                            self.emergency_moves += 1

                            self.isolation_saves += 1

                        else:

                            agent.planned_action = 5 # Rest/Reproduce

                    else:

                        agent.planned_action = 5

                else:

                    # Normal perception and decision

                    agent.perceive(self.get_view(agent.row, agent.col))

               

                # ---------------------------------------------------------

                # 🔥 INTEGRATION: Risk/Flow -> Action Damping

                # - survival_mode: force safe action always

                # - otherwise: with probability (1 - alpha), suppress risky actions

                # risky: move(0-3), reproduce(5) [clean(4) is safe]

                # ---------------------------------------------------------

                if survival_mode:

                    agent.planned_action = self._safe_action(agent)

                else:

                    # Probabilistic suppression based on alpha

                    if self.np_random.random() > alpha:

                        suppress_moves = bool(CONFIG["control"].get("suppress_moves", True))

                        suppress_rep = bool(CONFIG["control"].get("suppress_reproduce", True))

                       

                        current_action = agent.planned_action

                        if (suppress_moves and current_action < 4) or \

                           (suppress_rep and current_action == 5):

                            agent.planned_action = self._safe_action(agent)

               

                # Process the (possibly damped) action

                act = agent.planned_action

                target = (agent.row, agent.col)

               

                if act < 4: # Movement

                    d_row, d_col = [(-1,0), (1,0), (0,-1), (0,1)][act]

                    target = ((agent.row + d_row) % self.size,

                             (agent.col + d_col) % self.size)

                    moves[target].append(agent)

                elif act == 4: # Clean

                    moves[target].append(agent)

                    cleaners.append(agent)

                elif act == 5: # Rest/Reproduce

                    moves[target].append(agent)

                    # Reproduction may be suppressed by damping

                    if agent.energy > CONFIG["reproduce_threshold"]:

                        breeders.append(agent)

            # Resolve conflicts (multiple agents trying to move to same cell)

            new_positions = {}

            for pos, candidates in moves.items():

                winner = max(candidates, key=lambda a: a.energy)

                new_positions[winner.id] = pos

                for loser in candidates:

                    if loser != winner:

                        loser.energy -= 0.1

            # Update agent positions and apply costs

            for agent in self.agents:

                if agent.id in new_positions:

                    nr, nc = new_positions[agent.id]

                    if (nr, nc) != (agent.row, agent.col):

                        cost = (CONFIG["emergency_move_cost"]

                                if agent.is_emergency_move

                                else CONFIG["move_cost"])

                        agent.energy -= cost

                    agent.row, agent.col = nr, nc

               

                agent.energy -= 0.1 # Base metabolism

                agent.age += 1

            # Add entropy (dirt) to grid

            noise = np.random.rand(self.size, self.size) * 0.005

            self.grid = np.clip(self.grid + noise, 0, 1)

            # Cleaning action

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

                    if count > 0:

                        reward = (dirt_to_clean[agent.row, agent.col] *

                                 CONFIG["clean_reward"])

                        agent.energy += reward / count

            # Reproduction (may be suppressed by damping)

            if len(self.agents) < CONFIG["max_population"]:

                breeders.sort(key=lambda a: a.energy, reverse=True)

                for parent in breeders:

                    if len(self.agents) >= CONFIG["max_population"]:

                        break

                   

                    if parent.energy > CONFIG["reproduce_cost"]:

                        parent.energy -= CONFIG["reproduce_cost"]

                        self.id_counter += 1

                        child = Agent(self.id_counter, parent.row, parent.col)

                       

                        # Inherit brain with mutations

                        with torch.no_grad():

                            for cp, pp in zip(child.brain.parameters(),

                                             parent.brain.parameters()):

                                cp.data.copy_(pp.data)

                                fan_in = cp.size(0) if cp.ndim > 1 else 1

                                scale = (CONFIG["mutation_rate"] /

                                        np.sqrt(max(fan_in, 1)))

                                cp.add_(torch.randn_like(cp) * scale)

                       

                        self.agents.append(child)

            self.gen += 1

            return self.grid, self.agents

           

        except Exception as e:

            print(f"🚨 Critical Error in Engine Step: {e}")

            traceback.print_exc()

            return self.grid, self.agents

    # ---------------------------------------------------------

    # Logging with flow metrics

    # ---------------------------------------------------------

    def log_to_buffer(self, avg_energy, total_entropy):

        """Log simulation data with flow and control metrics."""

        population = len(self.agents)

       

        # Calculate dynamic metrics

        entropy_gradient = float(total_entropy - self.prev_total_entropy)

        population_volatility = int(abs(population - self.prev_population))

        emergency_ratio = float(self.emergency_moves / max(population, 1))

       

        # Get risk, verdict, and flow (flow calculated inside)

        risk, verdict, flow = self._compute_risk_and_verdict(

            entropy_gradient,

            population_volatility,

            emergency_ratio

        )

        # Update state for next step

        self.prev_total_entropy = float(total_entropy)

        self.prev_population = int(population)

       

        # Format log entry with all metrics

        self.log_buffer.append(

            f"{self.gen},{population},{avg_energy:.2f},{total_entropy:.2f},"

            f"{self.emergency_moves},{self.isolation_saves},"

            f"{entropy_gradient:.3f},{population_volatility},{emergency_ratio:.4f},"

            f"{risk:.4f},{flow:.4f},{verdict}\n"

        )

       

        # Flush buffer if full

        if len(self.log_buffer) >= CONFIG["log_buffer_size"]:

            self.flush_buffer()

    def flush_buffer(self):

        """Write buffered log data to file."""

        if self.log_buffer:

            try:

                with open(CONFIG["log_file"], 'a') as f:

                    f.writelines(self.log_buffer)

                self.log_buffer.clear()

            except IOError as e:

                print(f"⚠️ Warning: Logging failed - {e}")

# ---------------------------------------------------------

# [Visualization] with integrated metrics display

# ---------------------------------------------------------

def run_simulation():

    """Main simulation loop with visualization."""

    engine = ResoneticsEngine()

   

    # Clear existing log file

    if os.path.exists(CONFIG["log_file"]):

        os.remove(CONFIG["log_file"])

   

    # Write CSV header with all metrics

    with open(CONFIG["log_file"], 'w') as f:

        f.write(

            "Gen,Population,Avg_Energy,Total_Entropy,Emergency_Moves,Isolation_Saves,"

            "Entropy_Gradient,Population_Volatility,Emergency_Ratio,"

            "Collapse_Risk,Flow,Verdict\n"

        )

   

    # Setup visualization

    fig, ax = plt.subplots(figsize=(9, 9))

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

   

    im = ax.imshow(engine.grid, cmap='magma', vmin=0, vmax=1, animated=True)

    scatter = ax.scatter([], [], s=40, edgecolors='white', animated=True)

   

    # HUD text with all metrics including damping status

    text = ax.text(

        0.02, 0.95, '', transform=ax.transAxes, color='white',

        fontsize=11, fontweight='bold',

        bbox=dict(facecolor='black', alpha=0.6)

    )

    def update(frame):

        """Animation update function."""

        grid, agents = engine.step()

        im.set_data(grid)

       

        # Update agent visualization

        if agents:

            offsets = np.array([[a.col, a.row] for a in agents])

            energies = np.array([a.energy for a in agents])

            normalized_e = np.clip(energies / 100, 0, 1)

            colors = plt.cm.RdYlGn(normalized_e)

            scatter.set_offsets(offsets)

            scatter.set_facecolors(colors)

        else:

            scatter.set_offsets(np.zeros((0, 2)))

       

        # Calculate metrics

        avg_e = np.mean([a.energy for a in agents]) if agents else 0.0

        total_entropy = float(np.sum(grid))

       

        # Log data (includes flow calculation)

        engine.log_to_buffer(avg_e, total_entropy)

       

        population = len(agents)

        emergency_ratio = engine.emergency_moves / max(population, 1)

       

        # Compute damping alpha for display

        damping_alpha = engine._damping_alpha()

        survival_mode = engine._force_survival_now()

       

        # Enhanced HUD display with control status

        stats = (

            f"GENERATION: {engine.gen:04d}\n"

            f"POPULATION: {population:02d}\n"

            f"AVG ENERGY: {avg_e:6.1f}\n"

            f"ENTROPY : {total_entropy:7.1f}\n"

            f"EMERGENCIES: {engine.emergency_moves:02d}\n"

            f"SAVED : {engine.isolation_saves:04d}\n"

            f"RISK : {engine.last_risk:6.3f}\n"

            f"FLOW : {engine.last_flow:6.3f}\n"

            f"VERDICT : {engine.last_verdict.upper()}\n"

            f"DAMPING α : {damping_alpha:6.3f}\n"

            f"SURVIVAL : {'ON' if survival_mode else 'OFF'}\n"

            f"EMG_RATIO : {emergency_ratio:6.3f}"

        )

        text.set_text(stats)

       

        return im, scatter, text

   

    print("🌱 [Resonetics v3.3.3 Complete] Simulation Started...")

    print("📈 System Flow + Action Damping Integrated")

    print(" • Flow measures system sensitivity")

    print(" • Damping suppresses risky actions")

    print(" • Survival mode prevents collapse")

   

    # Start animation

    ani = animation.FuncAnimation(fig, update, frames=None, interval=50, blit=True)

    plt.title("Resonetics v3.3.3 Complete: Flow + Damping")

    plt.axis('off')

   

    try:

        plt.show()

    except KeyboardInterrupt:

        print("\n🛑 Simulation stopped by user.")

    finally:

        engine.flush_buffer()

        print("✅ Simulation ended cleanly.")

        print(f"📊 Total isolation saves: {engine.isolation_saves}")

        print(f"📈 Final Flow: {engine.last_flow:.3f}")

        print(f"🎛️ Control system active")

# ---------------------------------------------------------

# Batch mode for experiments (optional)

# ---------------------------------------------------------

def run_batch_experiment(iterations=1000):

    """

    Run simulation without visualization for data collection.

    """

    engine = ResoneticsEngine()

   

    if os.path.exists(CONFIG["log_file"]):

        os.remove(CONFIG["log_file"])

   

    with open(CONFIG["log_file"], 'w') as f:

        f.write(

            "Gen,Population,Avg_Energy,Total_Entropy,Emergency_Moves,Isolation_Saves,"

            "Entropy_Gradient,Population_Volatility,Emergency_Ratio,"

            "Collapse_Risk,Flow,Verdict\n"

        )

   

    print("Running batch experiment...")

    for i in range(iterations):

        engine.step()

       

        # Log every 10 steps

        if i % 10 == 0:

            avg_e = np.mean([a.energy for a in engine.agents]) if engine.agents else 0.0

            total_entropy = float(np.sum(engine.grid))

            engine.log_to_buffer(avg_e, total_entropy)

       

        # Progress indicator

        if i % 100 == 0:

            print(f" Step {i}/{iterations}, Risk: {engine.last_risk:.3f}, Flow: {engine.last_flow:.3f}")

   

    engine.flush_buffer()

    print(f"✅ Experiment completed. Data saved to {CONFIG['log_file']}")

    return engine

if __name__ == "__main__":

    # Run with visualization

    run_simulation()

   

    # Uncomment for batch mode (no visualization)

    # run_batch_experiment(iterations=2000)
