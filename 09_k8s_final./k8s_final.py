# ==============================================================================
# File: resonetics_k8s_v4_6_hardened.py
# Version: 4.6 (Reviewer-Hardened K8s Env)
# Description: RL environment for Kubernetes-style resource management with
#              improved termination logic, reward shaping, and Gymnasium-compliant rendering.
# Author: red1239109-cmd
# Copyright (c) 2025 Resonetics Project
#
# License: AGPL-3.0
# ==============================================================================

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Optional, Tuple


class KubernetesSmartTensorEnv(gym.Env):
    """
    Kubernetes cluster management environment with balanced rewards.

    Grid: (H, W, C)
      - C0: CPU load   [0, 1]
      - C1: MEM usage  [0, 1]
      - C2: PRIORITY   {0.2, 0.5, 1.0}

    Actions (Discrete=6):
      0: move up
      1: move down
      2: move left
      3: move right
      4: clean current node (reduce CPU, reset MEM)
      5: scale hotspot nodes (reduce CPU/MEM on worst nodes)

    Observation:
      - local 3x3 neighborhood (wrap-around) => 3*3*3 = 27
      - extras: agent_y_norm, agent_x_norm, budget_norm => 3
      total = 30
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    MOVE_DELTAS = [
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        super().__init__()

        config = config or {}
        self.render_mode = render_mode  # "human" | "ansi" | None

        # Grid configuration (can be overridden)
        self.grid_w = int(config.get("grid_w", 4))
        self.grid_h = int(config.get("grid_h", 4))
        self.channels = 3

        # Budget
        self.max_budget = float(config.get("max_budget", 100.0))
        self.budget_recovery = float(config.get("budget_recovery", 0.5))

        # Dynamics tuning
        self.cpu_noise_sigma = float(config.get("cpu_noise_sigma", 0.02))
        self.mem_drift_mu = float(config.get("mem_drift_mu", 0.005))
        self.mem_noise_sigma = float(config.get("mem_noise_sigma", 0.01))

        # Termination tuning (IMPORTANT for learnability)
        # v4.5: terminate on any OOM => episodes often end too early
        # v4.6: allow recovery; terminate only on "critical" conditions
        self.terminate_on_any_oom = bool(config.get("terminate_on_any_oom", False))
        self.oom_terminate_threshold = int(config.get("oom_terminate_threshold", 2))  # >= this count => terminate
        self.critical_oom_terminate = bool(config.get("critical_oom_terminate", True))
        self.avg_load_terminate_threshold = float(config.get("avg_load_terminate_threshold", 0.90))

        # Episode length
        self.max_steps = int(config.get("max_steps", 500))

        # Reward shaping weights
        self.time_penalty = float(config.get("time_penalty", -0.01))
        self.oom_penalty = float(config.get("oom_penalty", 2.0))
        self.critical_oom_penalty = float(config.get("critical_oom_penalty", 3.0))
        self.load_shaping_weight = float(config.get("load_shaping_weight", 0.5))  # potential-based shaping

        # Action/Observation spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(30,),
            dtype=np.float32
        )

        self.debug = bool(config.get("debug", False))

        # Precompute neighbor offsets for 3x3 view (wrap-around)
        dy, dx = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij")
        self._neighbor_dy = dy.ravel()
        self._neighbor_dx = dx.ravel()

        # State
        self.grid: np.ndarray = np.zeros((self.grid_h, self.grid_w, self.channels), dtype=np.float32)
        self.agent_y = 0
        self.agent_x = 0
        self.budget = 0.0
        self.steps = 0

    # -------------------------
    # Gymnasium API
    # -------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        # Initialize grid
        self.grid = np.zeros((self.grid_h, self.grid_w, self.channels), dtype=np.float32)

        # CPU: initial load (0 ~ 30%)
        self.grid[:, :, 0] = (self.np_random.random((self.grid_h, self.grid_w)).astype(np.float32) * 0.3)

        # MEM: initial usage (0 ~ 10%)
        self.grid[:, :, 1] = (self.np_random.random((self.grid_h, self.grid_w)).astype(np.float32) * 0.1)

        # PRIORITY: discrete levels
        priority_levels = np.array([0.2, 0.5, 1.0], dtype=np.float32)
        self.grid[:, :, 2] = self.np_random.choice(priority_levels, size=(self.grid_h, self.grid_w)).astype(np.float32)

        # Agent position
        self.agent_y = int(self.np_random.integers(0, self.grid_h))
        self.agent_x = int(self.np_random.integers(0, self.grid_w))

        # Budget / steps
        self.budget = float(self.max_budget)
        self.steps = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        # Time penalty
        reward = float(self.time_penalty)

        # Potential-based shaping: compare avg load before vs after
        avg_load_before = float(np.mean(self.grid[:, :, :2]))

        # Action
        if action < 4:
            dy, dx = self.MOVE_DELTAS[action]
            self.agent_y = (self.agent_y + dy) % self.grid_h
            self.agent_x = (self.agent_x + dx) % self.grid_w
        elif action == 4:
            reward += float(self._execute_clean_action())
        elif action == 5:
            reward += float(self._execute_scale_action())
        else:
            # Should never happen with Discrete(6), but keep it safe
            reward -= 0.1

        # Physics
        self._apply_physics()

        # OOM checks
        oom_nodes, critical_oom = self._check_oom_conditions()

        # Penalties
        if oom_nodes > 0:
            reward -= float(oom_nodes) * self.oom_penalty
            if critical_oom > 0:
                reward -= float(critical_oom) * self.critical_oom_penalty

        # Potential-based shaping (encourage reducing average load)
        avg_load_after = float(np.mean(self.grid[:, :, :2]))
        reward += (avg_load_before - avg_load_after) * self.load_shaping_weight

        # Budget recovery
        self.budget = float(min(self.max_budget, self.budget + self.budget_recovery))

        # Step count
        self.steps += 1

        # Termination logic (v4.6: less brittle)
        terminated = self._check_termination(avg_load_after, oom_nodes, critical_oom)
        truncated = self.steps >= self.max_steps

        info = {
            "avg_load": avg_load_after,
            "oom_nodes": int(oom_nodes),
            "critical_oom": int(critical_oom),
            "budget": float(self.budget),
        }

        obs = self._get_obs()

        # Gymnasium expects float reward
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        # No external resources, but keep for API completeness
        pass

    # -------------------------
    # Observation
    # -------------------------
    def _get_obs(self) -> np.ndarray:
        y_indices = (self.agent_y + self._neighbor_dy) % self.grid_h
        x_indices = (self.agent_x + self._neighbor_dx) % self.grid_w

        view = self.grid[y_indices, x_indices, :].astype(np.float32).ravel()  # 27

        extras = np.array(
            [
                self.agent_y / max(1, self.grid_h - 1) if self.grid_h > 1 else 0.0,
                self.agent_x / max(1, self.grid_w - 1) if self.grid_w > 1 else 0.0,
                self.budget / max(1e-8, self.max_budget),
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([view, extras], axis=0).astype(np.float32)

        # Safety: enforce bounds (tiny numerical drift)
        np.clip(obs, 0.0, 1.0, out=obs)
        return obs

    # -------------------------
    # Actions
    # -------------------------
    def _execute_clean_action(self) -> float:
        additional_reward = 0.0
        clean_cost = 0.5

        if self.budget < clean_cost:
            return -0.2

        self.budget -= clean_cost

        cpu_before = float(self.grid[self.agent_y, self.agent_x, 0])
        mem_before = float(self.grid[self.agent_y, self.agent_x, 1])
        priority = float(self.grid[self.agent_y, self.agent_x, 2])

        # Clean effect
        self.grid[self.agent_y, self.agent_x, 0] = np.float32(max(0.0, cpu_before - 0.5))
        self.grid[self.agent_y, self.agent_x, 1] = np.float32(0.0)

        cpu_cleaned = cpu_before - float(self.grid[self.agent_y, self.agent_x, 0])
        mem_cleaned = mem_before
        total_cleaned = cpu_cleaned + mem_cleaned

        if total_cleaned < 0.2:
            additional_reward = -0.1  # waste penalty
        else:
            effectiveness = min(1.0, total_cleaned / 1.5)
            priority_bonus = 1.0 + priority  # 1.2 / 1.5 / 2.0
            additional_reward = effectiveness * priority_bonus

        return float(additional_reward)

    def _execute_scale_action(self) -> float:
        # Hotspots: high CPU or MEM
        hotspot_mask = (self.grid[:, :, 0] > 0.8) | (self.grid[:, :, 1] > 0.8)
        hotspot_indices = np.argwhere(hotspot_mask)
        hotspot_count = int(len(hotspot_indices))

        if hotspot_count == 0:
            return -0.2

        unit_cost = 5.0
        max_affordable = int(min(hotspot_count, int(self.budget // unit_cost)))

        if max_affordable <= 0:
            return -0.5

        self.budget -= max_affordable * unit_cost

        if max_affordable == hotspot_count:
            targets = hotspot_indices
        else:
            loads = (self.grid[hotspot_mask, 0] + self.grid[hotspot_mask, 1]).astype(np.float32)
            worst_indices = np.argsort(-loads)[:max_affordable]
            targets = hotspot_indices[worst_indices]

        ty = targets[:, 0]
        tx = targets[:, 1]

        self.grid[ty, tx, 0] = np.maximum(0.0, self.grid[ty, tx, 0] - 0.3).astype(np.float32)
        self.grid[ty, tx, 1] = np.maximum(0.0, self.grid[ty, tx, 1] - 0.3).astype(np.float32)

        # Reward for coverage (balanced)
        additional_reward = 0.8 * (max_affordable / max(1, hotspot_count))
        return float(additional_reward)

    # -------------------------
    # Dynamics / Termination
    # -------------------------
    def _apply_physics(self) -> None:
        # CPU noise
        cpu_noise = self.np_random.normal(0.0, self.cpu_noise_sigma, (self.grid_h, self.grid_w)).astype(np.float32)
        self.grid[:, :, 0] = np.clip(self.grid[:, :, 0] + cpu_noise, 0.0, 1.0).astype(np.float32)

        # MEM drift + noise
        mem_noise = self.np_random.normal(self.mem_drift_mu, self.mem_noise_sigma, (self.grid_h, self.grid_w)).astype(np.float32)
        self.grid[:, :, 1] = np.clip(self.grid[:, :, 1] + mem_noise, 0.0, 1.0).astype(np.float32)

    def _check_oom_conditions(self) -> Tuple[int, int]:
        oom_mask = self.grid[:, :, 1] >= 0.99
        oom_count = int(np.sum(oom_mask))

        critical_oom_mask = oom_mask & (self.grid[:, :, 2] > 0.8)
        critical_oom_count = int(np.sum(critical_oom_mask))

        return oom_count, critical_oom_count

    def _check_termination(self, avg_load: float, oom_nodes: int, critical_oom: int) -> bool:
        if avg_load > self.avg_load_terminate_threshold:
            return True

        if self.terminate_on_any_oom:
            return oom_nodes > 0

        # v4.6 default: terminate only on severe OOM
        if oom_nodes >= self.oom_terminate_threshold:
            return True

        if self.critical_oom_terminate and critical_oom > 0:
            return True

        return False

    # -------------------------
    # Rendering
    # -------------------------
    def render(self):
        # Build output
        lines = []
        lines.append("\n" + "=" * 40)
        lines.append("K8s Cluster Status")
        lines.append("=" * 40)
        lines.append(f"Step: {self.steps:4d} | Budget: {self.budget:6.1f}/{self.max_budget:.1f}")

        oom_count, crit_oom = self._check_oom_conditions()
        if oom_count > 0:
            lines.append(f"⚠️  OOM Nodes: {oom_count} ({crit_oom} critical)")

        lines.append("\nCluster Grid:")
        for y in range(self.grid_h):
            row = []
            for x in range(self.grid_w):
                if y == self.agent_y and x == self.agent_x:
                    row.append("🤖")
                else:
                    cpu = float(self.grid[y, x, 0])
                    mem = float(self.grid[y, x, 1])
                    if mem > 0.9:
                        row.append("🔥")
                    elif cpu > 0.8:
                        row.append("♨️")
                    else:
                        row.append("🟩")
            lines.append(" ".join(row))

        lines.append("\nStatistics:")
        lines.append(f"Avg CPU: {np.mean(self.grid[:, :, 0]):.1%}")
        lines.append(f"Avg Mem: {np.mean(self.grid[:, :, 1]):.1%}")
        lines.append(f"Agent: ({self.agent_y}, {self.agent_x})")

        out = "\n".join(lines)

        if self.render_mode == "ansi":
            return out
        # default: human prints
        print(out)
        return None


# -------------------------
# Quick test
# -------------------------
def test_environment():
    print("Testing KubernetesSmartTensorEnv v4.6...")

    env = KubernetesSmartTensorEnv(
        config={
            "debug": False,
            "terminate_on_any_oom": False,
            "oom_terminate_threshold": 2,
            "avg_load_terminate_threshold": 0.90,
        },
        render_mode="human",
    )

    obs, info = env.reset(seed=42)
    env.render()

    actions = [4, 0, 4, 5, 5, 4]  # clean/move/scale mix
    for i, a in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(a)
        print(f"Action {i}: a={a} reward={reward:+.3f} avg_load={info['avg_load']:.3f} oom={info['oom_nodes']} budget={info['budget']:.1f}")
        if terminated:
            print("Episode terminated.")
            break
        if truncated:
            print("Episode truncated.")
            break

    env.render()
    env.close()


if __name__ == "__main__":
    test_environment()

