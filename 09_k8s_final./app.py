# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd
# ==============================================================================
# File: app.py (Resonetics Final Artifact v13.2 Patched)
# Features: Stable Learning Rate + Graceful Shutdown + Safe Loading + Optimized UI
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
from collections import deque
import random
import io
import gc
import os
from packaging import version 

# [Import Environment]
# 주의: 같은 폴더에 resonetics_k8s_v5_1_enterprise.py 파일이 있어야 합니다.
try:
    from resonetics_k8s_v5_1_enterprise import KubernetesSmartTensorEnv
except ImportError:
    st.error("🚨 필수 파일 누락: 'resonetics_k8s_v5_1_enterprise.py'가 같은 폴더에 있는지 확인하세요.")
    st.stop()

# ==============================================================================
# 1. The Brain (Prophet & Agent)
# ==============================================================================
class ProphetNetwork(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class ProphetGuidedAgent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.prophet = ProphetNetwork(input_dim=5)
        # [수정됨] Prophet 학습률 초기 설정 (안정성을 위해 낮게 설정)
        self.prophet_optim = optim.Adam(self.prophet.parameters(), lr=lr * 0.5)
        
        self.memory = deque(maxlen=5000)
        self.epsilon = 1.0
        self.action_dim = action_dim

    def get_risk(self, obs):
        if len(obs) < 5: return 0.0
        global_stats = torch.FloatTensor(obs[-5:]).unsqueeze(0).cpu()
        with torch.no_grad():
            risk = self.prophet(global_stats).item()
        return risk

    def act(self, obs):
        risk = self.get_risk(obs)
        effective_epsilon = self.epsilon * (1.0 - risk * 0.9)
        if random.random() < effective_epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(state)
        return torch.argmax(q_vals).item()

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size: return None
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, outcomes = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        outcomes = torch.FloatTensor(outcomes).unsqueeze(1)
        
        # DQN Update
        curr_Q = self.q_net(states).gather(1, actions)
        next_Q = self.q_net(next_states).max(1)[0].unsqueeze(1)
        target_Q = rewards + (0.99 * next_Q * (1 - dones))
        loss_q = self.loss_fn(curr_Q, target_Q.detach())
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()
        
        # Prophet Update
        global_stats = states[:, -5:]
        pred_risk = self.prophet(global_stats)
        loss_p = self.loss_fn(pred_risk, outcomes)
        self.prophet_optim.zero_grad()
        loss_p.backward()
        self.prophet_optim.step()
        
        return loss_q.item(), loss_p.item()
        
    def update_lr(self, lr):
        # [Critical Bug Fix] 
        # 이전 코드: g['lr'] = lr * 5.0 (과적합 유발)
        # 수정 코드: g['lr'] = lr * 0.5 (안정적 학습)
        for g in self.optimizer.param_groups: g['lr'] = lr
        for g in self.prophet_optim.param_groups: g['lr'] = lr * 0.5

# ==============================================================================
# 2. Logger
# ==============================================================================
class TrainingLogger:
    def __init__(self, max_logs=50):
        self.logs = deque(maxlen=max_logs)
    def log(self, level, msg):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.logs.append({"ts": ts, "level": level, "msg": msg})

# ==============================================================================
# 3. Streamlit UI (Production Ready)
# ==============================================================================
st.set_page_config(page_title="Resonetics Final Artifact", layout="wide", page_icon="🏛️")

# --- Session Initialization ---
if 'agent' not in st.session_state:
    st.session_state.env = KubernetesSmartTensorEnv(config={"max_steps": 300, "grid_w": 6, "grid_h": 6})
    st.session_state.agent = ProphetGuidedAgent(
        st.session_state.env.observation_space.shape[0], 
        st.session_state.env.action_space.n
    )
    st.session_state.history = {
        'step': deque(maxlen=200), 'reward': deque(maxlen=200), 
        'load': deque(maxlen=200), 'risk': deque(maxlen=200)
    }
    st.session_state.logger = TrainingLogger()
    st.session_state.ep_count = 0
    st.session_state.running = False
    st.session_state.last_gc = time.time()
    st.session_state.start_time = time.time()

# --- Sidebar ---
st.sidebar.title("🏛️ Command Center")

with st.sidebar.expander("💾 Persistence", expanded=True):
    if st.button("Save Checkpoint"):
        checkpoint = {
            'q_net': st.session_state.agent.q_net.state_dict(),
            'prophet': st.session_state.agent.prophet.state_dict(),
            'epsilon': st.session_state.agent.epsilon,
            'ep_count': st.session_state.ep_count,
            'memory': list(st.session_state.agent.memory)[-2000:] 
        }
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        st.download_button("Download .pt", buffer, "resonetics_final.pt")
        st.session_state.logger.log("success", "Checkpoint saved")
        
    # [Secure Load Implementation]
    uploaded = st.file_uploader("Load Checkpoint", type=['pt'])
    if uploaded:
        try:
            # 1순위: 안전 로드 (Weights Only)
            if version.parse(torch.__version__) >= version.parse("1.13"):
                ckpt = torch.load(uploaded, map_location='cpu', weights_only=True)
                st.success("✅ Secure Model Loaded (Weights Only)")
            else:
                st.warning("⚠️ Torch < 1.13: using legacy loader (compatibility mode)")
                ckpt = torch.load(uploaded, map_location='cpu', weights_only=False)
            
            # 상태 복원
            st.session_state.agent.q_net.load_state_dict(ckpt['q_net'])
            st.session_state.agent.prophet.load_state_dict(ckpt['prophet'])
            st.session_state.agent.epsilon = ckpt['epsilon']
            st.session_state.ep_count = ckpt['ep_count']
            st.session_state.agent.memory.clear()
            for m in ckpt['memory']: st.session_state.agent.memory.append(m)
            st.success(f"Restored Ep {ckpt['ep_count']}")
            
        except Exception as e:
            st.error(f"Load Error (Try Legacy?): {e}")

# [CSV Export Logic]
if st.sidebar.button("📊 Export Metrics CSV"):
    h = st.session_state.history
    df = pd.DataFrame({'step': list(h['step']), 'reward': list(h['reward']), 'load': list(h['load']), 'risk': list(h['risk'])})
    st.sidebar.download_button("Download CSV", df.to_csv(index=False), "metrics.csv")

st.sidebar.markdown("---")
max_steps = st.sidebar.slider("Max Steps (Dynamic)", 100, 2000, st.session_state.env.max_steps)
if st.session_state.env.max_steps != max_steps:
    st.session_state.env.max_steps = max_steps

sim_speed = st.sidebar.slider("Delay", 0.0, 0.2, 0.01)
lr = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
st.session_state.agent.update_lr(lr)

# [Auto-shutdown setting]
max_runtime = st.sidebar.number_input("Max Runtime (h, 0=∞)", 0, 72, 0, 1, help="0 = unlimited")

c1, c2 = st.sidebar.columns(2)
if c1.button("▶️ Start"): 
    st.session_state.running = True
    st.session_state.start_time = time.time() # 시작 시 타이머 초기화
if c2.button("⏹️ Stop"): st.session_state.running = False

# --- Main UI ---
st.title("Resonetics v13.2: Final Artifact (Optimized)")
col_m, col_l = st.columns([3, 1])

with col_l:
    st.subheader("System Logs")
    log_c = st.container()
    st.markdown("---")
    m_risk = st.empty()
    m_load = st.empty()

with col_m:
    chart_ph = st.empty()

# --- Training Loop ---
if st.session_state.running:
    env = st.session_state.env
    agent = st.session_state.agent
    history = st.session_state.history
    logger = st.session_state.logger
    
    obs, _ = env.reset()
    episode_buffer = deque(maxlen=env.max_steps)
    
    while st.session_state.running:
        # [Graceful Shutdown Logic]
        if max_runtime > 0:
            max_sec = max_runtime * 3600
            elapsed = time.time() - st.session_state.get('start_time', time.time())
            if elapsed > max_sec:
                st.session_state.running = False
                
                auto_save_name = f"auto_save_ep{st.session_state.ep_count}.pt"
                ckpt = {
                    'q_net': agent.q_net.state_dict(),
                    'prophet': agent.prophet.state_dict(),
                    'epsilon': agent.epsilon,
                    'ep_count': st.session_state.ep_count,
                    'memory': list(agent.memory)[-2000:]
                }
                torch.save(ckpt, auto_save_name)
                
                st.error(f"🛑 Max Runtime ({max_runtime}h) Reached. Loop Stopped.")
                st.write(f"Local backup saved to: `{auto_save_name}`")
                
                with open(auto_save_name, "rb") as f:
                    st.download_button("⬇️ Download Auto-Save", f, file_name=auto_save_name)
                st.stop()
        
        # --- Agent Step ---
        risk = agent.get_risk(obs)
        action = agent.act(obs)
        next_obs, r, term, trunc, info = env.step(action)
        
        episode_buffer.append((obs, action, r, next_obs, term))
        obs = next_obs
        
        history['step'].append(len(history['step']))
        history['reward'].append(r)
        history['load'].append(info['avg_load'])
        history['risk'].append(risk)
        
        # [Optimization] 그래프 업데이트를 여기(Loop 내부)에서 제거하여 속도 향상
        # 실시간 수치만 업데이트
        m_risk.metric("Prophet Risk", f"{risk:.1%}")
        m_load.metric("Avg Load", f"{info['avg_load']:.2f}")

        # 로그 표시
        with log_c:
            st.empty()
            for l in list(logger.logs)[-6:]:
                icon = {"critical":"🚨", "success":"✅"}.get(l['level'], "📝")
                st.caption(f"{icon} **{l['ts']}** {l['msg']}")
        
        if sim_speed > 0:
            time.sleep(sim_speed)
            
        # --- Episode End ---
        if term or trunc:
            st.session_state.ep_count += 1
            is_oom = info.get('oom', 0) > 0
            outcome = 1.0 if (term and is_oom) else 0.0
            
            for exp in episode_buffer:
                agent.memory.append(exp + (outcome,))
            
            # Training
            losses = []
            for _ in range(10):
                res = agent.train_step()
                if res: losses.append(res[0])
            
            agent.epsilon = max(0.05, agent.epsilon * 0.99)
            
            # Log Result
            if is_oom: logger.log("critical", f"Crash (Ep {st.session_state.ep_count})")
            else: logger.log("success", f"Survived (Ep {st.session_state.ep_count})")
            
            # GC
            if time.time() - st.session_state.last_gc > 300:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                st.session_state.last_gc = time.time()
                logger.log("info", "System GC Executed")
                
            # [Fix: Safe CSV Save] 중복 방지를 위해 mode='w' 사용
            LOG_PATH = "metrics_log.csv"
            df = pd.DataFrame({k: list(v) for k, v in history.items()})
            df.to_csv(LOG_PATH, mode='w', index=False)
            
            # [Optimization] 그래프를 에피소드가 끝날 때 한 번만 그림 (성능 최적화)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=list(history['reward']), name='Reward', line=dict(color='#3b82f6')))
            fig.add_trace(go.Scatter(y=list(history['load']), name='Load', line=dict(color='#f59e0b'), yaxis='y2'))
            fig.add_trace(go.Scatter(y=list(history['risk']), name='Risk', line=dict(color='#ef4444', dash='dot'), yaxis='y2'))
            
            fig.update_layout(
                height=350, margin=dict(t=0,b=0,l=0,r=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis2=dict(overlaying='y', side='right', range=[0, 1]),
                legend=dict(orientation="h", y=1.1)
            )
            chart_ph.plotly_chart(fig, use_container_width=True, key=f"chart_{st.session_state.ep_count}")
            
            break
