```python
# ==============================================================================
# File: resonetics_k8s_v4_5_fixed.py
# Version: 4.5 (Optimized Gardener)
# Description: RL Environment with improved reward balance and vectorization
# Author: red1239109-cmd
# Copyright (c) 2025 Resonetics Project
#
# License: AGPL-3.0
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys

class KubernetesSmartTensorEnv(gym.Env):
    """Kubernetes cluster management environment with balanced rewards"""
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}
    
    # Movement directions
    MOVE_DELTAS = [
        (-1, 0),  # Move up
        (1, 0),   # Move down
        (0, -1),  # Move left
        (0, 1),   # Move right
    ]

    def __init__(self, config=None):
        super().__init__()
        
        # Grid configuration
        self.grid_w = 4
        self.grid_h = 4
        self.channels = 3  # 0:CPU, 1:MEM, 2:PRIORITY
        
        # Resource management
        self.max_budget = 100.0
        self.budget_recovery = 0.5
        
        # Action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Observation space: 3x3x3(27) + Position(2) + Budget(1) = 30
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(30,), dtype=np.float32
        )
        
        # Debug mode
        self.debug = config.get('debug', False) if config else False
        
        # Precompute neighbor indices for 3x3 view (optimization)
        dy, dx = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing='ij')
        self._neighbor_dy = dy.ravel()
        self._neighbor_dx = dx.ravel()
        
        # Initialize state variables
        self.grid = None
        self.agent_y = 0
        self.agent_x = 0
        self.budget = 0.0
        self.steps = 0

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize grid
        self.grid = np.zeros((self.grid_h, self.grid_w, self.channels), dtype=np.float32)
        
        # CPU: Random initial load (0-30%)
        self.grid[:, :, 0] = self.np_random.random((self.grid_h, self.grid_w)) * 0.3
        
        # Memory: Random initial usage (0-10%)
        self.grid[:, :, 1] = self.np_random.random((self.grid_h, self.grid_w)) * 0.1
        
        # Priority: Discrete priority levels
        priority_levels = np.array([0.2, 0.5, 1.0], dtype=np.float32)
        self.grid[:, :, 2] = self.np_random.choice(priority_levels, size=(self.grid_h, self.grid_w))
        
        # Random agent starting position
        self.agent_y = self.np_random.integers(0, self.grid_h)
        self.agent_x = self.np_random.integers(0, self.grid_w)
        
        # Initialize budget
        self.budget = self.max_budget
        self.steps = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Get current observation (3x3 neighborhood view)"""
        # Calculate neighbor indices with wrap-around
        y_indices = (self.agent_y + self._neighbor_dy) % self.grid_h
        x_indices = (self.agent_x + self._neighbor_dx) % self.grid_w
        
        # Extract 3x3x3 view
        view = self.grid[y_indices, x_indices, :].ravel()
        
        # Additional state information
        extras = np.array([
            self.agent_y / self.grid_h,      # Normalized y position
            self.agent_x / self.grid_w,      # Normalized x position
            self.budget / self.max_budget    # Normalized budget
        ], dtype=np.float32)
        
        return np.concatenate([view, extras])

    def step(self, action):
        """Execute one environment step"""
        # Start with small time penalty
        reward = -0.01
        
        # Execute action based on type
        if action < 4:  # Movement action
            dy, dx = self.MOVE_DELTAS[action]
            self.agent_y = (self.agent_y + dy) % self.grid_h
            self.agent_x = (self.agent_x + dx) % self.grid_w
            
        elif action == 4:  # Clean current node
            reward += self._execute_clean_action()
            
        elif action == 5:  # Scale hotspot nodes
            reward += self._execute_scale_action()
        
        # Apply physics simulation
        self._apply_physics()
        
        # Check for OOM conditions
        oom_nodes, critical_oom = self._check_oom_conditions()
        
        # Apply penalties for OOM
        if oom_nodes > 0:
            reward -= oom_nodes * 2.0  # Reduced penalty for balance
            reward -= critical_oom * 3.0
        
        # Budget recovery
        self.budget = min(self.max_budget, self.budget + self.budget_recovery)
        
        # Update step counter
        self.steps += 1
        
        # Calculate termination conditions
        avg_load = np.mean(self.grid[:, :, :2])
        terminated = self._check_termination(avg_load, oom_nodes)
        truncated = self.steps >= 500
        
        # Info dictionary
        info = {
            'avg_load': float(avg_load),
            'oom_nodes': int(oom_nodes),
            'critical_oom': int(critical_oom),
            'budget': float(self.budget)
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _execute_clean_action(self):
        """Execute clean action on current node"""
        additional_reward = 0.0
        clean_cost = 0.5
        
        if self.budget >= clean_cost:
            # Deduct cost
            self.budget -= clean_cost
            
            # Store current values
            cpu_before = self.grid[self.agent_y, self.agent_x, 0]
            mem_before = self.grid[self.agent_y, self.agent_x, 1]
            priority = self.grid[self.agent_y, self.agent_x, 2]
            
            # Apply cleaning
            self.grid[self.agent_y, self.agent_x, 0] = max(0, cpu_before - 0.5)
            self.grid[self.agent_y, self.agent_x, 1] = 0.0
            
            # Calculate cleaned amount
            cpu_cleaned = cpu_before - self.grid[self.agent_y, self.agent_x, 0]
            mem_cleaned = mem_before
            total_cleaned = cpu_cleaned + mem_cleaned
            
            # Calculate reward based on effectiveness
            if total_cleaned < 0.2:
                additional_reward = -0.1  # Waste penalty
            else:
                # Balanced reward formula
                effectiveness = min(1.0, total_cleaned / 1.5)
                priority_bonus = 1.0 + priority
                additional_reward = effectiveness * priority_bonus
        else:
            # Insufficient budget penalty
            additional_reward = -0.2
        
        return additional_reward

    def _execute_scale_action(self):
        """Execute scale action on hotspot nodes"""
        additional_reward = 0.0
        
        # Find hotspots (high CPU or memory)
        hotspot_mask = (self.grid[:, :, 0] > 0.8) | (self.grid[:, :, 1] > 0.8)
        hotspot_indices = np.argwhere(hotspot_mask)
        hotspot_count = len(hotspot_indices)
        
        if hotspot_count > 0:
            unit_cost = 5.0
            max_affordable = min(hotspot_count, int(self.budget // unit_cost))
            
            if max_affordable > 0:
                # Deduct cost
                self.budget -= max_affordable * unit_cost
                
                # Scale the hotspots (vectorized for performance)
                if max_affordable == hotspot_count:
                    # Scale all hotspots
                    targets = hotspot_indices
                else:
                    # Scale only the worst ones
                    loads = self.grid[hotspot_mask, 0] + self.grid[hotspot_mask, 1]
                    worst_indices = np.argsort(-loads)[:max_affordable]
                    targets = hotspot_indices[worst_indices]
                
                # Apply scaling (vectorized operation)
                self.grid[targets[:, 0], targets[:, 1], 0] = np.maximum(
                    0.0, self.grid[targets[:, 0], targets[:, 1], 0] - 0.3
                )
                self.grid[targets[:, 0], targets[:, 1], 1] = np.maximum(
                    0.0, self.grid[targets[:, 0], targets[:, 1], 1] - 0.3
                )
                
                # Balanced reward
                additional_reward = 0.8 * (max_affordable / max(1, hotspot_count))
            else:
                # Cannot afford scaling
                additional_reward = -0.5
        else:
            # No hotspots to scale
            additional_reward = -0.2
        
        return additional_reward

    def _apply_physics(self):
        """Apply physics simulation to cluster"""
        # CPU: Random fluctuations
        self.grid[:, :, 0] = np.clip(
            self.grid[:, :, 0] + self.np_random.normal(0, 0.02, (self.grid_h, self.grid_w)),
            0.0, 1.0
        )
        
        # Memory: Slow drift with random noise
        self.grid[:, :, 1] = np.clip(
            self.grid[:, :, 1] + self.np_random.normal(0.005, 0.01, (self.grid_h, self.grid_w)),
            0.0, 1.0
        )

    def _check_oom_conditions(self):
        """Check for OOM (Out of Memory) conditions"""
        # Nodes with memory above threshold
        oom_mask = self.grid[:, :, 1] >= 0.99
        oom_count = np.sum(oom_mask)
        
        # Critical nodes in OOM state
        critical_oom_mask = oom_mask & (self.grid[:, :, 2] > 0.8)
        critical_oom_count = np.sum(critical_oom_mask)
        
        return int(oom_count), int(critical_oom_count)

    def _check_termination(self, avg_load, oom_nodes):
        """Check if episode should terminate"""
        # Terminate if system overloaded or any OOM occurs
        return bool(avg_load > 0.85 or oom_nodes > 0)

    def render(self):
        """Render environment state as ASCII"""
        output_lines = []
        output_lines.append("\n" + "="*40)
        output_lines.append("K8s Cluster Status")
        output_lines.append("="*40)
        output_lines.append(f"Step: {self.steps:4d} | Budget: {self.budget:6.1f}/{self.max_budget}")
        
        # Check for OOM warning
        oom_count, crit_oom = self._check_oom_conditions()
        if oom_count > 0:
            output_lines.append(f"⚠️  OOM Nodes: {oom_count} ({crit_oom} critical)")
        
        output_lines.append("\nCluster Grid:")
        
        for y in range(self.grid_h):
            row_chars = []
            for x in range(self.grid_w):
                # Agent position
                if y == self.agent_y and x == self.agent_x:
                    row_chars.append("🤖")
                else:
                    cpu = self.grid[y, x, 0]
                    mem = self.grid[y, x, 1]
                    
                    # Determine character based on state
                    if mem > 0.9:
                        row_chars.append("🔥")  # OOM risk
                    elif cpu > 0.8:
                        row_chars.append("♨️")   # High CPU
                    else:
                        row_chars.append("🟩")   # Normal
                        
            output_lines.append(" ".join(row_chars))
        
        # Add statistics
        output_lines.append("\nStatistics:")
        output_lines.append(f"Avg CPU: {np.mean(self.grid[:,:,0]):.1%}")
        output_lines.append(f"Avg Mem: {np.mean(self.grid[:,:,1]):.1%}")
        output_lines.append(f"Agent: ({self.agent_y}, {self.agent_x})")
        
        print("\n".join(output_lines))

# Test function
def test_environment():
    """Test the K8s environment"""
    print("Testing K8s Smart Tensor Environment...")
    
    # Create environment
    env = KubernetesSmartTensorEnv()
    
    # Reset and render initial state
    obs, info = env.reset(seed=42)
    print("\nInitial state:")
    env.render()
    
    # Test some actions
    test_actions = [4, 0, 4, 5]  # Clean, Move up, Clean, Scale
    
    print("\nTesting actions:")
    for i, action in enumerate(test_actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action {i}: reward={reward:.3f}, budget={info['budget']:.1f}")
        
        if terminated:
            print("Episode terminated!")
            break
    
    print("\nFinal state:")
    env.render()

if __name__ == "__main__":
    test_environment()
```
