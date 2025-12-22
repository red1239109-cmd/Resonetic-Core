# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: resonetics_grid_v9_0.py
# Product: Resonetics Grid Center (Parallel Processing & Grid Search)
# ==============================================================================

import numpy as np
import pandas as pd
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
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
# 1. Learning Tracker (The Black Box Recorder)
# ==============================================================================
class LearningTracker:
    """에피소드별 상세 학습 데이터 기록"""
    def __init__(self):
        self.history = {
            'ep_rewards': [],
            'ep_steps': [],
            'final_epsilon': [],
            'avg_q_change': [], # Q-Value 변화량 (수렴도)
            'actions': []       # 액션 분포
        }

    def record(self, reward, steps, epsilon, q_change, actions):
        self.history['ep_rewards'].append(reward)
        self.history['ep_steps'].append(steps)
        self.history['final_epsilon'].append(epsilon)
        self.history['avg_q_change'].append(q_change)
        self.history['actions'].append(actions)

# ==============================================================================
# 2. Parameterized Agent (Accepts Hyperparams)
# ==============================================================================
class ConfigurableQAgent:
    def __init__(self, action_space, lr=0.1, gamma=0.95, epsilon_decay=0.995):
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon_decay
        
        # State: Load(5) x Budget(5) x Risk(5) -> Action(8)
        self.q_table = np.zeros((5, 5, 5, 8))
        self.last_q_change = 0.0

    def _to_state(self, obs):
        avg_load = (obs[-5] + obs[-4]) / 2
        return (
            min(4, int(avg_load * 5)),
            min(4, int(obs[-6] * 5)),
            min(4, int(obs[-1] * 5))
        )

    def act(self, obs, eval_mode=False):
        if not eval_mode and np.random.rand() < self.epsilon:
            return np.random.randint(0, 8)
        return np.argmax(self.q_table[self._to_state(obs)])

    def learn(self, obs, act, rew, next_obs, done):
        s = self._to_state(obs)
        ns = self._to_state(next_obs)
        
        target = rew
        if not done:
            target += self.gamma * np.max(self.q_table[ns])
        
        old_q = self.q_table[s][act]
        new_q = old_q + self.lr * (target - old_q)
        self.q_table[s][act] = new_q
        
        self.last_q_change = abs(new_q - old_q)
        
        if done: # Episode end decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ==============================================================================
# 3. Parallel Experiment Manager (The Engine)
# ==============================================================================
def _execute_single_job(job_data):
    """
    개별 프로세스에서 실행될 작업 (Pickle 가능해야 함)
    Top-level function for multiprocessing compatibility
    """
    job_id, name, config, agent_params, n_episodes = job_data
    
    # Init Env & Agent
    env = KubernetesSmartTensorEnv(config=config)
    agent = ConfigurableQAgent(env.action_space, **agent_params)
    tracker = LearningTracker()
    
    start_t = time.time()
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r = 0
        steps = 0
        q_deltas = []
        actions = []
        
        while True:
            act = agent.act(obs)
            next_obs, r, term, trunc, _ = env.step(act)
            agent.learn(obs, act, r, next_obs, term)
            
            obs = next_obs
            ep_r += r
            steps += 1
            q_deltas.append(agent.last_q_change)
            actions.append(act)
            
            if term or trunc: break
            
        tracker.record(
            reward=ep_r,
            steps=steps,
            epsilon=agent.epsilon,
            q_change=np.mean(q_deltas) if q_deltas else 0,
            actions=actions
        )
        
    duration = time.time() - start_t
    
    # Result Summary
    return {
        "id": job_id,
        "name": name,
        "params": str(agent_params),
        "avg_reward": np.mean(tracker.history['ep_rewards']),
        "avg_steps": np.mean(tracker.history['ep_steps']),
        "convergence": np.mean(tracker.history['avg_q_change'][-5:]), # 마지막 5에피소드 수렴도
        "duration": duration,
        "history": tracker.history # 상세 데이터 포함
    }

class ParallelExperimentManager:
    def __init__(self, save_dir="grid_results"):
        self.queue = []
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def schedule(self, name, config, agent_params, n_episodes):
        job_id = len(self.queue)
        self.queue.append((job_id, name, config, agent_params, n_episodes))

    def create_hyperparameter_sweep(self):
        """하이퍼파라미터 스윕 작업 생성"""
        print("🕸️ Generating Hyperparameter Grid...")
        
        # Grid Space
        lrs = [0.05, 0.1, 0.2]
        gammas = [0.9, 0.95, 0.99]
        
        for lr in lrs:
            for gamma in gammas:
                name = f"QL_lr{lr}_g{gamma}"
                # Base Environment Config
                env_cfg = {"max_budget": 100.0, "max_steps": 500}
                # Agent Params
                ag_cfg = {"lr": lr, "gamma": gamma}
                
                self.schedule(name, env_cfg, ag_cfg, n_episodes=20) # 20 episodes per setting

    def run_parallel(self, max_workers=None):
        """병렬 실행 엔진"""
        print(f"\n🚀 Launching Grid Compute: {len(self.queue)} Jobs on {max_workers or 'ALL'} Cores")
        
        results = []
        
        # ProcessPoolExecutor for CPU bound tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # TQDM progress bar integration
            futures = [executor.submit(_execute_single_job, job) for job in self.queue]
            
            iterator = tqdm(concurrent.futures.as_completed(futures), total=len(futures)) \
                       if TQDM_AVAILABLE else concurrent.futures.as_completed(futures)
            
            for future in iterator:
                try:
                    res = future.result()
                    results.append(res)
                    if not TQDM_AVAILABLE: print(f"   ✅ Job Finished: {res['name']}")
                except Exception as e:
                    print(f"   ❌ Job Failed: {e}")

        return pd.DataFrame(results)

# ==============================================================================
# 4. Analysis Dashboard
# ==============================================================================
def analyze_grid_results(df):
    """스윕 결과 시각화"""
    print("\n📊 Top 3 Configurations:")
    top3 = df.sort_values("avg_reward", ascending=False).head(3)
    print(top3[["name", "avg_reward", "avg_steps", "convergence"]])
    
    # Learning Curves for Top 3
    plt.figure(figsize=(12, 5))
    
    for _, row in top3.iterrows():
        rewards = row['history']['ep_rewards']
        plt.plot(rewards, label=f"{row['name']} (R:{row['avg_reward']:.1f})")
        
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
    # 1. Init
    center = ParallelExperimentManager()
    
    # 2. Design Grid Search
    center.create_hyperparameter_sweep()
    
    # 3. Execute (Parallel)
    # workers=None means use all available CPU cores
    start_time = time.time()
    results_df = center.run_parallel(max_workers=None)
    
    print(f"\n⏱️ Total Compute Time: {time.time() - start_time:.2f}s")
    
    # 4. Visualize
    analyze_grid_results(results_df)

