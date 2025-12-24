#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: resonetics_grid_v9_0.py
# Product: Resonetics Grid Center (Parallel Processing & Grid Search)
# Notes: Hardened for multiprocessing + reproducibility + large sweeps
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt

# [Dependencies]
# pip install tqdm (Progress Bar)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# [Import Environment]
from resonetics_k8s_v5_1_enterprise import KubernetesSmartTensorEnv

# ==============================================================================
# 0. Utilities
# ==============================================================================
def _safe_json_dump(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _seed_everything(seed: int):
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass

def _action_space_n(action_space) -> int:
    """Support gym-like spaces (.n) or raw int."""
    if hasattr(action_space, "n"):
        return int(action_space.n)
    return int(action_space)

# ==============================================================================
# 1. Learning Tracker (The Black Box Recorder)
# ==============================================================================
class LearningTracker:
    """에피소드별 상세 학습 데이터 기록"""
    def __init__(self):
        self.history = {
            "ep_rewards": [],
            "ep_steps": [],
            "final_epsilon": [],
            "avg_q_change": [],  # Q-Value 변화량 (수렴도)
            "actions": []        # 액션 분포
        }

    def record(self, reward, steps, epsilon, q_change, actions):
        self.history["ep_rewards"].append(float(reward))
        self.history["ep_steps"].append(int(steps))
        self.history["final_epsilon"].append(float(epsilon))
        self.history["avg_q_change"].append(float(q_change))
        self.history["actions"].append(list(map(int, actions)))

# ==============================================================================
# 2. Parameterized Agent (Accepts Hyperparams)
# ==============================================================================
class ConfigurableQAgent:
    def __init__(self, action_space, lr=0.1, gamma=0.95, epsilon_decay=0.995):
        self.action_space = action_space
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = float(epsilon_decay)

        # Derive action count dynamically (avoid 8 hardcode)
        self.nA = _action_space_n(action_space)

        # State: Load(5) x Budget(5) x Risk(5) -> Action(nA)
        self.q_table = np.zeros((5, 5, 5, self.nA), dtype=np.float32)
        self.last_q_change = 0.0

    def _to_state(self, obs):
        # Edge-safe obs access
        if len(obs) < 6:
            raise ValueError(f"Observation too short: len(obs)={len(obs)} (need >= 6)")

        avg_load = (obs[-5] + obs[-4]) / 2.0
        s0 = min(4, max(0, int(avg_load * 5)))
        s1 = min(4, max(0, int(obs[-6] * 5)))
        s2 = min(4, max(0, int(obs[-1] * 5)))
        return (s0, s1, s2)

    def act(self, obs, eval_mode=False):
        if (not eval_mode) and (np.random.rand() < self.epsilon):
            return np.random.randint(0, self.nA)
        return int(np.argmax(self.q_table[self._to_state(obs)]))

    def learn(self, obs, act, rew, next_obs, done: bool):
        s = self._to_state(obs)
        ns = self._to_state(next_obs)

        target = float(rew)
        if not done:
            target += self.gamma * float(np.max(self.q_table[ns]))

        old_q = float(self.q_table[s][act])
        new_q = old_q + self.lr * (target - old_q)
        self.q_table[s][act] = new_q

        self.last_q_change = abs(new_q - old_q)

        # Episode end decay
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ==============================================================================
# 3. Parallel Experiment Manager (The Engine)
# ==============================================================================
def _execute_single_job(job_data):
    """
    개별 프로세스에서 실행될 작업 (Pickle 가능해야 함)
    Top-level function for multiprocessing compatibility
    """
    job_id, name, config, agent_params, n_episodes, save_dir = job_data

    # Reproducibility: seed per job
    base_seed = 1234 + int(job_id)
    _seed_everything(base_seed)

    # Init Env & Agent
    env = KubernetesSmartTensorEnv(config=config)

    # If env supports seeding:
    # Try reset(seed=...) first, else ignore
    def _reset_env():
        try:
            return env.reset(seed=base_seed)
        except TypeError:
            return env.reset()
        except Exception:
            return env.reset()

    agent = ConfigurableQAgent(env.action_space, **agent_params)
    tracker = LearningTracker()

    start_t = time.time()

    for _ in range(int(n_episodes)):
        obs, _info = _reset_env()
        ep_r = 0.0
        steps = 0
        q_deltas = []
        actions = []

        while True:
            act = agent.act(obs)
            next_obs, r, term, trunc, _ = env.step(act)

            # IMPORTANT: done must include truncation too
            done = bool(term) or bool(trunc)

            agent.learn(obs, act, r, next_obs, done)

            obs = next_obs
            ep_r += float(r)
            steps += 1
            q_deltas.append(float(agent.last_q_change))
            actions.append(int(act))

            if done:
                break

        tracker.record(
            reward=ep_r,
            steps=steps,
            epsilon=agent.epsilon,
            q_change=float(np.mean(q_deltas)) if q_deltas else 0.0,
            actions=actions
        )

    duration = time.time() - start_t

    # Save detailed history to file (avoid DataFrame memory blowup)
    save_dir = Path(save_dir)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_path = save_dir / "histories" / f"job_{job_id:03d}_{name}_{run_stamp}.json"
    _safe_json_dump(hist_path, {
        "job_id": job_id,
        "name": name,
        "env_config": config,
        "agent_params": agent_params,
        "seed": base_seed,
        "history": tracker.history,
    })

    # Summary only for parent aggregation
    last_k = min(5, len(tracker.history["avg_q_change"]))
    conv = float(np.mean(tracker.history["avg_q_change"][-last_k:])) if last_k > 0 else 0.0

    return {
        "id": int(job_id),
        "name": name,
        "params": str(agent_params),
        "avg_reward": float(np.mean(tracker.history["ep_rewards"])) if tracker.history["ep_rewards"] else 0.0,
        "avg_steps": float(np.mean(tracker.history["ep_steps"])) if tracker.history["ep_steps"] else 0.0,
        "convergence": conv,
        "duration": float(duration),
        "history_path": str(hist_path),
    }

class ParallelExperimentManager:
    def __init__(self, save_dir="grid_results"):
        self.queue = []
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "histories").mkdir(parents=True, exist_ok=True)

    def schedule(self, name, config, agent_params, n_episodes):
        job_id = len(self.queue)
        self.queue.append((job_id, name, config, agent_params, n_episodes, str(self.save_dir)))

    def create_hyperparameter_sweep(self):
        """하이퍼파라미터 스윕 작업 생성"""
        print("🕸️ Generating Hyperparameter Grid...")

        # Grid Space
        lrs = [0.05, 0.1, 0.2]
        gammas = [0.9, 0.95, 0.99]
        eps_decays = [0.995]  # extend if needed

        for lr in lrs:
            for gamma in gammas:
                for eps_decay in eps_decays:
                    name = f"QL_lr{lr}_g{gamma}_ed{eps_decay}"
                    # Base Environment Config
                    env_cfg = {"max_budget": 100.0, "max_steps": 500}
                    # Agent Params
                    ag_cfg = {"lr": lr, "gamma": gamma, "epsilon_decay": eps_decay}

                    self.schedule(name, env_cfg, ag_cfg, n_episodes=20)  # 20 episodes per setting

    def run_parallel(self, max_workers=None):
        """병렬 실행 엔진"""
        print(f"\n🚀 Launching Grid Compute: {len(self.queue)} Jobs on {max_workers or 'ALL'} Cores")

        results = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_execute_single_job, job) for job in self.queue]

            iterator = tqdm(concurrent.futures.as_completed(futures), total=len(futures)) \
                       if TQDM_AVAILABLE else concurrent.futures.as_completed(futures)

            for future in iterator:
                try:
                    res = future.result()
                    results.append(res)
                    if not TQDM_AVAILABLE:
                        print(f"   ✅ Job Finished: {res['name']}")
                except Exception as e:
                    print(f"   ❌ Job Failed: {e}")

        df = pd.DataFrame(results)
        # Save summary
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.save_dir / f"summary_{stamp}.csv"
        df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"🧾 Summary saved: {summary_path}")
        return df

# ==============================================================================
# 4. Analysis Dashboard
# ==============================================================================
def analyze_grid_results(df: pd.DataFrame):
    """스윕 결과 시각화"""
    if df.empty:
        print("No results to analyze.")
        return

    print("\n📊 Top 3 Configurations:")
    top3 = df.sort_values("avg_reward", ascending=False).head(3)
    print(top3[["name", "avg_reward", "avg_steps", "convergence", "history_path"]])

    # Optional: plot using histories by loading json
    plt.figure(figsize=(12, 5))

    for _, row in top3.iterrows():
        try:
            with open(row["history_path"], "r", encoding="utf-8") as f:
                payload = json.load(f)
            rewards = payload["history"]["ep_rewards"]
            plt.plot(rewards, label=f"{row['name']} (R:{row['avg_reward']:.1f})")
        except Exception:
            pass

    plt.title("Learning Curves (Top 3 Agents)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==============================================================================
# Execution
# ==============================================================================
if __name__ == "__main__":
    center = ParallelExperimentManager()

    center.create_hyperparameter_sweep()

    start_time = time.time()
    results_df = center.run_parallel(max_workers=None)
    print(f"\n⏱️ Total Compute Time: {time.time() - start_time:.2f}s")

    analyze_grid_results(results_df)
