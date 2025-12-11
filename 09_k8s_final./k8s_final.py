# ==============================================================================
# File: resonetics_k8s_final.py
# Version: 4.2 (Standard Compliant)
# Description: Fixed seeding for reproducibility & Gymnasium API alignment
# ==============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class KubernetesSmartTensorEnv(gym.Env):
    """
    Kubernetes Smart Tensor Env (v4.2 Final)
    - 3D Tensor State: CPU, MEM, PRIORITY
    - Smart Budgeting: Cost-Benefit Analysis & Partial Scaling
    - Standard Compliance: Proper Seeding & API
    """
    
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, config=None):
        super().__init__()
        
        # Grid Setup (4x4)
        self.grid_w = 4
        self.grid_h = 4
        self.channels = 3 # 0:CPU, 1:MEM, 2:PRI
        
        # Budget Setup
        self.max_budget = 100.0
        self.budget_recovery = 0.5
        
        # Actions: Move(4) + Clean(1) + Scale(1)
        self.action_space = spaces.Discrete(6)
        
        # Observation Space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(30,), dtype=np.float32
        )
        
        self.debug = config.get('debug', False) if config else False
        
        self.grid = None
        self.agent_y = 0
        self.agent_x = 0
        self.budget = 0
    
    def reset(self, seed=None, options=None):
        # [Reflect 4] Use Gymnasium's seeding mechanism
        super().reset(seed=seed) 
        
        # Initialize 3D Grid
        self.grid = np.zeros((self.grid_h, self.grid_w, self.channels), dtype=np.float32)
        
        # Use self.np_random for reproducibility
        self.grid[:, :, 0] = self.np_random.random((self.grid_h, self.grid_w)) * 0.3 # CPU
        self.grid[:, :, 1] = self.np_random.random((self.grid_h, self.grid_w)) * 0.1 # MEM
        
        # PRIORITY: Fixed choices
        pri_choices = np.array([0.2, 0.5, 1.0], dtype=np.float32)
        self.grid[:, :, 2] = self.np_random.choice(pri_choices, size=(self.grid_h, self.grid_w))
        
        self.agent_y = self.np_random.integers(0, self.grid_h)
        self.agent_x = self.np_random.integers(0, self.grid_w)
        
        self.budget = self.max_budget
        self.steps = 0
        self.total_reward = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        view = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                y = (self.agent_y + dy) % self.grid_h
                x = (self.agent_x + dx) % self.grid_w
                view.extend(self.grid[y, x])
        
        view.append(self.agent_y / self.grid_h)
        view.append(self.agent_x / self.grid_w)
        view.append(self.budget / self.max_budget)
        return np.array(view, dtype=np.float32)
    
    def step(self, action):
        reward = -0.01 
        
        # === 1. Movement ===
        if action < 4:
            if action == 0: self.agent_y = (self.agent_y - 1) % self.grid_h
            elif action == 1: self.agent_y = (self.agent_y + 1) % self.grid_h
            elif action == 2: self.agent_x = (self.agent_x - 1) % self.grid_w
            elif action == 3: self.agent_x = (self.agent_x + 1) % self.grid_w
            
        # === 2. Smart Cleaning ===
        elif action == 4:
            cpu = self.grid[self.agent_y, self.agent_x, 0]
            mem = self.grid[self.agent_y, self.agent_x, 1]
            pri = self.grid[self.agent_y, self.agent_x, 2]
            
            clean_cost = 0.5
            if self.budget >= clean_cost:
                self.budget -= clean_cost
                self.grid[self.agent_y, self.agent_x, 0] = max(0, cpu - 0.5)
                self.grid[self.agent_y, self.agent_x, 1] = 0.0 
                
                total_cleaned = (cpu - self.grid[self.agent_y, self.agent_x, 0]) + \
                                (mem - self.grid[self.agent_y, self.agent_x, 1])
                
                # Cost-Benefit check
                if total_cleaned < 0.2:
                    reward -= 0.1
                else:
                    reward += total_cleaned * (1.0 + pri) * 2.0
            else:
                reward -= 0.2 
                
        # === 3. Smart Scaling ===
        elif action == 5:
            hot_mask = (self.grid[:, :, 0] > 0.8) | (self.grid[:, :, 1] > 0.8)
            hot_indices = np.argwhere(hot_mask)
            hot_count = len(hot_indices)
            
            if hot_count > 0:
                unit_cost = 5.0
                total_req = hot_count * unit_cost
                
                if self.budget >= total_req:
                    process_count = hot_count
                    self.budget -= total_req
                else:
                    process_count = int(self.budget // unit_cost)
                    if process_count > 0:
                        self.budget -= process_count * unit_cost
                    else:
                        reward -= 0.5
                        process_count = 0
                
                if process_count > 0:
                    targets = hot_indices[:process_count]
                    for y, x in targets:
                        self.grid[y, x, 0] = max(0, self.grid[y, x, 0] - 0.3)
                        self.grid[y, x, 1] = max(0, self.grid[y, x, 1] - 0.3)
                    
                    reward += process_count * 0.8
                    if process_count < hot_count:
                        reward -= (hot_count - process_count) * 0.05
            else:
                reward -= 0.2
        
        # === 4. Physics (Using self.np_random) ===
        # CPU Noise
        self.grid[:, :, 0] += self.np_random.random((self.grid_h, self.grid_w)) * 0.02
        # MEM Leak
        self.grid[:, :, 1] += self.np_random.random((self.grid_h, self.grid_w)) * 0.01 + 0.005
        
        self.grid = np.clip(self.grid, 0, 1)
        
        # === 5. Danger Check ===
        oom_nodes = np.sum(self.grid[:, :, 1] >= 0.99)
        if oom_nodes > 0:
            crit_oom = np.sum((self.grid[:, :, 1] >= 0.99) & (self.grid[:, :, 2] > 0.8))
            reward -= (oom_nodes * 2.0 + crit_oom * 5.0)
            
        # === 6. Wrap up ===
        self.budget = min(self.max_budget, self.budget + self.budget_recovery)
        avg_load = np.mean(self.grid[:, :, :2])
        self.steps += 1
        self.total_reward += reward
        
        # Termination conditions
        terminated = bool(avg_load > 0.85) # Failed (Collapsed)
        truncated = bool(self.steps >= 500) # Time limit reached
        
        info = {'avg_load': float(avg_load), 'budget': float(self.budget)}
        
        return self._get_obs(), reward, terminated, truncated, info

if __name__ == "__main__":
    # Test Run
    env = KubernetesSmartTensorEnv({'debug': True})
    obs, _ = env.reset(seed=42)
    print("✅ Environment initialized with Seed 42")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Load={info['avg_load']:.2f}")
        if terminated or truncated:
            print("Done!")
            break
