#!/usr/bin/env python3
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
            'actions': []       # 액션 분포(히스토그램)
        }

    def record(self, reward, steps, epsilon, q_change, actions_hist):
        self.history['ep_rewards'].append(float(reward))
        self.history['ep_steps'].append(int(steps))
        self.history['final_epsilon'].append(float(epsilon))
        self.history['avg_q_change'].append(float(q_change))
        self.history['actions'].append(list(actions_hist))


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

        # State: Load(5) x Budget(5) x Risk(5) -> Action(8)
        self.q_table = np.zeros((5, 5, 5, 8), dtype=np.float32)
        self.last_q_change = 0.0

    def _to_state(self, obs):
        avg_load = (obs[-5] + obs[-4]) / 2
        return (
            min(4, int(avg_load * 5)),
            min(4, int(obs[-6] * 5)),
            min(4, int(obs[-1] * 5))
        )

    def act(self, obs, eval_mode=False):
        if (not eval_mode) and (np.random.rand() < self.epsilon):
            return int(np.random.randint(0, 8))
        return int(np.argmax(self.q_table[self._to_state(obs)]))

    def learn(self, obs, act, rew, next_obs, terminal):
        """
        terminal=True only when env terminates naturally.
        truncation(time-limit) should NOT be treated as terminal for learning.
        """
        s = self._to_state(obs)
        ns = self._to_state(next_obs)

        target = float(rew)
        if not terminal:
            target += self.gamma * float(np.max(self.q_table[ns]))

        old_q = float(self.q_table[s][act])
        new_q = old_q + self.lr * (target - old_q)
        self.q_table[s][act] = new_q

        self.last_q_change = abs(new_q - old_q)

        # Epsilon decay at episode end (caller decides what "episode end" means)
        # keep decay outside if you want; leaving here for minimal change:
        if terminal:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ==============================================================================
# 2.5 Simple Evaluation (Greedy)
# ==============================================================================
def evaluate_agent(env, agent: ConfigurableQAgent, n_eval: int = 5) -> float:
    rewards = []
    for _ in range(int(n_eval)):
        obs, _ = env.reset()
        ep_r = 0.0
        while True:
            act = agent.act(obs, eval_mode=True)
            obs, r, term, trunc, _ = env.step(act)
            ep_r += float(r)
            if term or trunc:
                break
        rewards.append(ep_r)
    return float(np.mean(rewards)) if rewards else 0.0


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

    for _ in range(int(n_episodes)):
        obs, _ = env.reset()
        ep_r = 0.0
        steps = 0
        q_deltas = []

        # ✅ actions 전체 저장 대신 히스토그램(메모리/디스크 폭발 방지)
        action_hist = np.zeros(8, dtype=int)

        while True:
            act = agent.act(obs)
            next_obs, r, term, trunc, _ = env.step(act)

            # ✅ (A) truncation은 학습상 terminal로 보지 않음
            terminal_for_learning = bool(term)         # 자연 종료만 terminal
            episode_end = bool(term or trunc)          # 루프 종료 기준

            agent.learn(obs, act, r, next_obs, terminal_for_learning)

            obs = next_obs
            ep_r += float(r)
            steps += 1
            q_deltas.append(float(agent.last_q_change))
            action_hist[int(act)] += 1

            if episode_end:
                # ✅ epsilon decay를 "에피소드 종료"에 맞춰 하고 싶으면 여기서 처리하는 게 더 정확함.
                # agent.learn 내부 terminal 기준 decay는 유지했지만,
                # trunc에서도 decay 원하면 아래를 켜면 됨.
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
                break

        tracker.record(
            reward=ep_r,
            steps=steps,
            epsilon=agent.epsilon,
            q_change=float(np.mean(q_deltas)) if q_deltas else 0.0,
            actions_hist=action_hist.tolist()
        )

    duration = time.time() - start_t

    # ✅ 평가: 탐험 없이 greedy로 몇 판 돌려서 점수 기록
    eval_reward = evaluate_agent(env, agent, n_eval=5)

    # Result Summary
    return {
        "id": int(job_id),
        "name": str(name),
        "params": str(agent_params),
        "avg_reward": float(np.mean(tracker.history['ep_rewards'])) if tracker.history['ep_rewards'] else 0.0,
        "eval_reward": float(eval_reward),
        "avg_steps": float(np.mean(tracker.history['ep_steps'])) if tracker.history['ep_steps'] else 0.0,
        "convergence": float(np.mean(tracker.history['avg_q_change'][-5:])) if tracker.history['avg_q_change'] else 0.0,
        "duration": float(duration),
        "history": tracker.history  # 상세 데이터 포함
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
        epsilon_decays = [0.995]  # 필요하면 확장

        for lr in lrs:
            for gamma in gammas:
                for epsd in epsilon_decays:
                    name = f"QL_lr{lr}_g{gamma}_ed{epsd}"
                    # Base Environment Config
                    env_cfg = {"max_budget": 100.0, "max_steps": 500}
                    # Agent Params
                    ag_cfg = {"lr": lr, "gamma": gamma, "epsilon_decay": epsd}

                    self.schedule(name, env_cfg, ag_cfg, n_episodes=20)  # 20 episodes per setting

    def run_parallel(self, max_workers=None):
        """병렬 실행 엔진"""
        print(f"\n🚀 Launching Grid Compute: {len(self.queue)} Jobs on {max_workers or 'ALL'} Cores")

        results = []

        # ProcessPoolExecutor for CPU bound tasks
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

        # 저장(선택): 결과 CSV
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = self.save_dir / f"grid_results_{ts}.csv"
            df.drop(columns=["history"], errors="ignore").to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"💾 Saved summary CSV: {out_csv}")
        except Exception:
            pass

        return df


# ==============================================================================
# 4. Analysis Dashboard
# ==============================================================================
def analyze_grid_results(df):
    """스윕 결과 시각화"""
    if df is None or df.empty:
        print("No results to analyze.")
        return

    print("\n📊 Top 3 Configurations (by eval_reward):")
    top3 = df.sort_values("eval_reward", ascending=False).head(3)
    print(top3[["name", "avg_reward", "eval_reward", "avg_steps", "convergence", "duration"]])

    # Learning Curves for Top 3
    plt.figure(figsize=(12, 5))

    for _, row in top3.iterrows():
        rewards = row['history']['ep_rewards']
        plt.plot(rewards, label=f"{row['name']} (eval:{row['eval_reward']:.1f})")

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
