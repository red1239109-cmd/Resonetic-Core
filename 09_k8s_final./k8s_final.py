# ==============================================================================
# File: resonetics_k8s_v4_4_final.py
# Version: 4.4 (Final Gardener)
# Description: RL Environment with ASCII Render & OOM Termination
# ==============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys

class KubernetesSmartTensorEnv(gym.Env):
    """
    Kubernetes Smart Tensor Env (v4.4 Final)
    - Vectorized Observation
    - Realistic Physics
    - Fast Fail (OOM Termination)
    - ASCII Rendering
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(self, config=None):
        super().__init__()
        
        self.grid_w = 4
        self.grid_h = 4
        self.channels = 3 # 0:CPU, 1:MEM, 2:PRI
        
        self.max_budget = 100.0
        self.budget_recovery = 0.5
        
        self.action_space = spaces.Discrete(6)
        
        # Obs: 3x3x3(27) + Pos(2) + Budget(1) = 30
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(30,), dtype=np.float32
        )
        
        self.debug = config.get('debug', False) if config else False
        
        # Indexing cache
        dy, dx = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing='ij')
        self._dy = dy.ravel()
        self._dx = dx.ravel()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.grid_h, self.grid_w, self.channels), dtype=np.float32)
        
        # Init
        self.grid[:, :, 0] = self.np_random.random((self.grid_h, self.grid_w)) * 0.3 # CPU
        self.grid[:, :, 1] = self.np_random.random((self.grid_h, self.grid_w)) * 0.1 # MEM
        
        # Priority
        pri_choices = np.array([0.2, 0.5, 1.0], dtype=np.float32)
        self.grid[:, :, 2] = self.np_random.choice(pri_choices, size=(self.grid_h, self.grid_w))
        
        self.agent_y = self.np_random.integers(0, self.grid_h)
        self.agent_x = self.np_random.integers(0, self.grid_w)
        
        self.budget = self.max_budget
        self.steps = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        y_indices = (self.agent_y + self._dy) % self.grid_h
        x_indices = (self.agent_x + self._dx) % self.grid_w
        
        view = self.grid[y_indices, x_indices, :].ravel()
        
        extras = np.array([
            self.agent_y / self.grid_h,
            self.agent_x / self.grid_w,
            self.budget / self.max_budget
        ], dtype=np.float32)
        
        return np.concatenate([view, extras])
    
    def step(self, action):
        base_reward = -0.01
        
        # 1. Action Logic
        if action < 4: # Move
            if action == 0: self.agent_y = (self.agent_y - 1) % self.grid_h
            elif action == 1: self.agent_y = (self.agent_y + 1) % self.grid_h
            elif action == 2: self.agent_x = (self.agent_x - 1) % self.grid_w
            elif action == 3: self.agent_x = (self.agent_x + 1) % self.grid_w
            
        elif action == 4: # Clean
            clean_cost = 0.5
            if self.budget >= clean_cost:
                self.budget -= clean_cost
                cpu = self.grid[self.agent_y, self.agent_x, 0]
                mem = self.grid[self.agent_y, self.agent_x, 1]
                pri = self.grid[self.agent_y, self.agent_x, 2]
                
                self.grid[self.agent_y, self.agent_x, 0] = max(0, cpu - 0.5)
                self.grid[self.agent_y, self.agent_x, 1] = 0.0
                
                cleaned = (cpu - self.grid[self.agent_y, self.agent_x, 0]) + \
                          (mem - self.grid[self.agent_y, self.agent_x, 1])
                
                if cleaned < 0.2:
                    base_reward -= 0.1 # Waste penalty
                else:
                    base_reward += cleaned * (1.0 + pri) * 2.0
            else:
                base_reward -= 0.2
                
        elif action == 5: # Scale
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
                        base_reward -= 0.5
                        process_count = 0
                
                if process_count > 0:
                    targets = hot_indices[:process_count]
                    for y, x in targets:
                        self.grid[y, x, 0] = max(0, self.grid[y, x, 0] - 0.3)
                        self.grid[y, x, 1] = max(0, self.grid[y, x, 1] - 0.3)
                    base_reward += process_count * 0.8
            else:
                base_reward -= 0.2

        # 2. Physics
        self.grid[:, :, 0] = np.clip(
            self.grid[:, :, 0] + self.np_random.normal(0, 0.02, (self.grid_h, self.grid_w)), 
            0, 1
        )
        self.grid[:, :, 1] = np.clip(
            self.grid[:, :, 1] + self.np_random.normal(0.005, 0.01, (self.grid_h, self.grid_w)), 
            0, 1
        )
        
        # 3. Danger Check & Fast Fail
        oom_nodes = np.sum(self.grid[:, :, 1] >= 0.99)
        if oom_nodes > 0:
            crit_oom = np.sum((self.grid[:, :, 1] >= 0.99) & (self.grid[:, :, 2] > 0.8))
            base_reward -= (oom_nodes * 5.0 + crit_oom * 10.0) # Penalty
            
        # 4. Wrap up
        self.budget = min(self.max_budget, self.budget + self.budget_recovery)
        self.steps += 1
        
        avg_load = np.mean(self.grid[:, :, :2])
        
        # [Terminated Condition Update]
        # Die if OOM happens OR Average load is too high
        terminated = bool(avg_load > 0.85 or oom_nodes > 0)
        truncated = bool(self.steps >= 500)
        
        return self._get_obs(), base_reward, terminated, truncated, {'avg_load': float(avg_load)}

    # [New Feature] ASCII Render
    def render(self):
        output = ["\n--- K8s Cluster Status ---"]
        output.append(f"Steps: {self.steps} | Budget: {self.budget:.1f}")
        
        for y in range(self.grid_h):
            row_str = ""
            for x in range(self.grid_w):
                if y == self.agent_y and x == self.agent_x:
                    char = "🤖"
                else:
                    cpu = self.grid[y, x, 0]
                    mem = self.grid[y, x, 1]
                    if mem > 0.9: char = "🔥" # OOM Risk
                    elif cpu > 0.8: char = "♨️" # High CPU
                    else: char = "🟩" # Normal
                row_str += char + " "
            output.append(row_str)
        
        print("\n".join(output))

if __name__ == "__main__":
    env = KubernetesSmartTensorEnv()
    env.reset()
    print("🖥️ Rendering Test:")
    env.render()
    
    # Test step
    env.step(4) # Clean
    print("\n🖥️ After Cleaning:")
    env.render()
