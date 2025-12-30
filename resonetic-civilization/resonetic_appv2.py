# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 red1239109-cmd
# ==============================================================================
# File: resonetic_civilization_v4_9_ultimate.py
# Product: Resonetic Civilization v4.9 (Ultimate Edition)
# Status: Robust, Fail-safe, Production Grade
# ==============================================================================

import streamlit as st
import random
import numpy as np
import pandas as pd
import time
import uuid
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ------------------------------------------------------------------------------
# 1. The Neural Brain
# ------------------------------------------------------------------------------
class Brain(nn.Module):
    def __init__(self, input_size, output_size):
        super(Brain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

# ------------------------------------------------------------------------------
# 2. System Components
# ------------------------------------------------------------------------------
class PolicyEngine:
    def __init__(self, allocation_config):
        self.ratios = allocation_config if allocation_config else {
            'Education': 0.3, 'Welfare': 0.2, 'Tech': 0.3, 
            'Green': 0.1, 'Defense': 0.1
        }
        self.budgets = {}
        
    def allocate(self, total_revenue):
        for policy, ratio in self.ratios.items():
            self.budgets[policy] = total_revenue * ratio
            
    def get_budget(self, name):
        return self.budgets.get(name, 0.0)

class AdvancedMonitor:
    def calculate_gini(self, citizens):
        if not citizens: return 0.0
        wealths = sorted([max(0, c.wealth) for c in citizens]) 
        n = len(wealths)
        if n == 0 or sum(wealths) == 0: return 0.0
        cumulative = sum(w * (i + 1) for i, w in enumerate(wealths))
        return (2 * cumulative) / (n * sum(wealths)) - (n + 1) / n

# ------------------------------------------------------------------------------
# 3. Neural Agent
# ------------------------------------------------------------------------------
class NeuralCitizen:
    def __init__(self, generation=1, initial_wealth=5.0):
        self.id = str(uuid.uuid4())[:4]
        self.wealth = initial_wealth
        self.happiness = 0.5
        self.age = 0
        self.generation = generation
        
        self.input_dim = 4 
        self.action_dim = 2 
        
        self.brain = Brain(self.input_dim, self.action_dim)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.005)
        self.loss_fn = nn.MSELoss()
        
        self.epsilon = 0.3
        self.gamma = 0.9

    def get_state_tensor(self, global_trust):
        safe_wealth = max(0.01, self.wealth)
        wealth_norm = np.log1p(safe_wealth) / 5.0 
        
        # [Patch 3] Scaling Stability (Clamp inputs)
        # 2.0을 넘어가면 2.0으로 고정하여 신경망 입력 폭주 방지
        wealth_norm = min(wealth_norm, 2.0) 
        
        state = [
            global_trust,
            min(1.0, wealth_norm),
            self.happiness,
            min(1.0, self.age / 80.0)
        ]
        return torch.FloatTensor(state)

    def decide(self, global_trust):
        state = self.get_state_tensor(global_trust)
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        with torch.no_grad():
            q_values = self.brain(state)
            return torch.argmax(q_values).item()

    def learn(self, global_trust, action, reward, next_global_trust):
        state = self.get_state_tensor(global_trust)
        next_state = self.get_state_tensor(next_global_trust)
        
        q_values = self.brain(state)
        current_q = q_values[action]
        
        with torch.no_grad():
            next_q_values = self.brain(next_state)
            max_next_q = torch.max(next_q_values)
            clipped_reward = np.clip(reward, -10.0, 10.0)
            target_q = torch.tensor(clipped_reward, dtype=torch.float32) + self.gamma * max_next_q
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.epsilon = max(0.05, self.epsilon * 0.99)

# ------------------------------------------------------------------------------
# 4. Simulation Core
# ------------------------------------------------------------------------------
class Civilization:
    def __init__(self, name, config):
        self.name = name
        self.year = 0
        self.trust = 0.5
        self.tech_level = 1.0
        self.event_log = deque(maxlen=20)
        
        self.monitor = AdvancedMonitor()
        self.policy = PolicyEngine(config.get('budget_allocation'))
        self.tax_rate = config['tax_rate']
        
        self.citizens = [
            NeuralCitizen(initial_wealth=5.0 + random.random())
            for _ in range(config['pop_size'])
        ]
        
        self.history_maxlen = 1000
        self.history = {
            'year': deque(maxlen=self.history_maxlen),
            'trust': deque(maxlen=self.history_maxlen),
            'population': deque(maxlen=self.history_maxlen),
            'avg_wealth': deque(maxlen=self.history_maxlen),
            'gini': deque(maxlen=self.history_maxlen),
            'happiness': deque(maxlen=self.history_maxlen),
            'tech': deque(maxlen=self.history_maxlen)
        }
        
    def _trigger_shocks(self):
        if random.random() < 0.05:
            event_type = random.choice(["Pandemic", "Financial Crisis", "Disinformation"])
            if event_type == "Pandemic":
                deaths = 0
                for c in self.citizens:
                    c.happiness -= 0.3
                    if random.random() < 0.2:
                        c.wealth = -10
                        deaths += 1
                self.event_log.append(f"Year {self.year}: 🦠 PANDEMIC! {deaths} died.")
            elif event_type == "Financial Crisis":
                for c in self.citizens: c.wealth *= 0.6
                self.event_log.append(f"Year {self.year}: 📉 MARKET CRASH!")
            elif event_type == "Disinformation":
                self.trust = max(0.1, self.trust - 0.4)
                self.event_log.append(f"Year {self.year}: 🕵️ FAKE NEWS!")

    def step(self):
        # [Patch 2] Zero Population Handling
        if not self.citizens:
            self.event_log.append(f"Year {self.year}: 💀 CIVILIZATION COLLAPSED!")
            # 멸망했어도 그래프는 그려야 하므로 History에 0으로 기록
            self.history['year'].append(self.year)
            self.history['trust'].append(0)
            self.history['population'].append(0)
            self.history['avg_wealth'].append(0)
            self.history['gini'].append(0)
            self.history['happiness'].append(0)
            self.history['tech'].append(self.tech_level)
            return False

        self.year += 1
        self._trigger_shocks()
        current_trust = self.trust
        
        shuffled_citizens = self.citizens.copy()
        random.shuffle(shuffled_citizens)
        
        pairs = []
        for i in range(len(shuffled_citizens) // 2):
            pairs.append((shuffled_citizens[2*i], shuffled_citizens[2*i+1]))
            
        if len(shuffled_citizens) % 2 != 0:
            leftover = shuffled_citizens[-1]
            partner = random.choice(shuffled_citizens[:-1])
            pairs.append((leftover, partner))

        multiplier = 1.0 + (np.log(self.tech_level) * 0.5)
        rewards_map = {
            (0, 0): (1.5, 1.5), (1, 1): (-0.5, -0.5),
            (1, 0): (3.0, -2.0), (0, 1): (-2.0, 3.0)
        }
        
        crimes = 0
        interactions = []
        
        for p1, p2 in pairs:
            a1 = p1.decide(current_trust)
            a2 = p2.decide(current_trust)
            
            if a1 == 1: crimes += 1
            if a2 == 1: crimes += 1
            
            r1_fin, r2_fin = rewards_map[(a1, a2)]
            w1_change = r1_fin * multiplier
            w2_change = r2_fin * multiplier
            
            p1.wealth += w1_change
            p2.wealth += w2_change
            
            psy_1 = 0.5 if a2 == 0 else -1.5
            psy_2 = 0.5 if a1 == 0 else -1.5
            
            p1.happiness = np.clip(p1.happiness + (w1_change/10.0) + (psy_1*0.1), 0.0, 1.0)
            p2.happiness = np.clip(p2.happiness + (w2_change/10.0) + (psy_2*0.1), 0.0, 1.0)
            
            interactions.append((p1, a1, w1_change + psy_1))
            interactions.append((p2, a2, w2_change + psy_2)) 

        crime_rate = crimes / max(1, len(interactions))
        
        def_budget = self.policy.get_budget('Defense')
        sec_factor = 1.0 / (1.0 + np.log1p(def_budget) * 0.5) if def_budget > 0 else 1.0
        trust_change = 0.05 if crime_rate < 0.2 else -0.1 * sec_factor
        self.trust = np.clip(self.trust + trust_change, 0.0, 1.0)
        
        for agent, action, reward in interactions:
            agent.learn(current_trust, action, reward, self.trust)

        tax_revenue = sum(max(0, c.wealth * self.tax_rate) for c in self.citizens)
        for c in self.citizens: c.wealth -= max(0, c.wealth * self.tax_rate)
        self.policy.allocate(tax_revenue)
        
        welfare = self.policy.get_budget('Welfare')
        poor = [c for c in self.citizens if c.wealth < 3.0]
        if poor and welfare > 0:
            sub = welfare / len(poor)
            for c in poor: c.wealth += sub
            
        tech = self.policy.get_budget('Tech')
        tech_prob = 1.0 - np.exp(-tech / (50.0 * max(0.1, self.tech_level * 0.5)))
        if random.random() < tech_prob:
            self.tech_level *= (1.0 + random.uniform(0.01, 0.05))
        
        survivors = []
        new_borns = []
        for c in self.citizens:
            c.wealth -= 0.5 
            c.age += 1
            age_factor = c.age / 80.0
            survival_prob = 1.0 / (1.0 + np.exp((age_factor - 1.0) * 10))
            if c.wealth > 0 and random.random() < survival_prob:
                survivors.append(c)
                if c.wealth > 15.0:
                    c.wealth -= 5.0
                    child = NeuralCitizen(generation=c.generation+1, initial_wealth=1.0)
                    if random.random() < 0.7:
                        child.brain.load_state_dict(c.brain.state_dict())
                    new_borns.append(child)
        
        self.citizens = survivors + new_borns

        # History Update
        self.history['year'].append(self.year)
        self.history['trust'].append(self.trust)
        self.history['population'].append(len(self.citizens))
        self.history['avg_wealth'].append(np.mean([c.wealth for c in self.citizens]) if self.citizens else 0)
        self.history['gini'].append(self.monitor.calculate_gini(self.citizens))
        self.history['happiness'].append(np.mean([c.happiness for c in self.citizens]) if self.citizens else 0)
        self.history['tech'].append(self.tech_level)
        
        # [Patch 1] Data Consistency Check
        # 데이터 길이가 하나라도 다르면 경고 로그 (보통은 위 로직으로 안 생기지만, 만약 발생하면 즉시 감지)
        lengths = {k: len(v) for k, v in self.history.items()}
        if len(set(lengths.values())) > 1:
            self.event_log.append(f"⚠️ DATA SYNC ERROR: {lengths}")

        # [Patch 4] Stop Conditions (Critical Alerts)
        if len(self.citizens) <= 2:
            self.event_log.append(f"Year {self.year}: ⚠️ Population critical (<=2)!")
            return False
        
        if self.trust <= 0.1:
            self.event_log.append(f"Year {self.year}: 💔 Trust collapsed (<=0.1)!")
            # 멈추지는 않고 경고만 줌 (사용자가 멸망을 관찰할 수도 있으므로)

        return True

# ------------------------------------------------------------------------------
# 5. UI Setup
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Resonetic v4.9 (Ultimate)", layout="wide", page_icon="🛡️")
st.title("🛡️ Resonetic v4.9: The Ultimate Edition")

if 'running' not in st.session_state: st.session_state.running = False
if 'run_requested' not in st.session_state: st.session_state.run_requested = False

with st.sidebar:
    st.header("Experiment Setup")
    mode = st.radio("Mode", ["Single Run", "A/B Test"], disabled=st.session_state.running)
    
    edu = st.slider("Education (Alpha)", 0.0, 1.0, 0.3, disabled=st.session_state.running)
    wel = st.slider("Welfare", 0.0, 1.0, 0.2, disabled=st.session_state.running)
    tech = st.slider("Tech", 0.0, 1.0, 0.3, disabled=st.session_state.running)
    
    alloc = {'Education': edu, 'Welfare': wel, 'Tech': tech, 'Green': 0.1, 'Defense': 0.1}
    total = sum(alloc.values())
    if total == 0: total = 1
    alloc = {k: v/total for k,v in alloc.items()}
    
    config = {"tax_rate": 0.15, "pop_size": 40, "budget_allocation": alloc}

    if st.button("🚀 Initialize", disabled=st.session_state.running):
        if mode == "Single Run":
            st.session_state.civs = [Civilization("Neural World A", config)]
        else:
            conf_b = config.copy()
            conf_b['budget_allocation'] = {'Education': 0.1, 'Welfare': 0.0, 'Tech': 0.8, 'Green':0.0, 'Defense':0.05}
            st.session_state.civs = [
                Civilization("Balanced AI", config),
                Civilization("Tech Jungle", conf_b)
            ]
        st.rerun()

if 'civs' in st.session_state:
    if st.button("⏩ Run 10 Years", disabled=st.session_state.running):
        st.session_state.run_requested = True
        st.rerun()

    if st.session_state.run_requested:
        st.session_state.running = True
        st.session_state.run_requested = False
        
        progress = st.progress(0)
        status_text = st.empty()
        
        for i in range(10):
            status_text.text(f"Simulating Year {i+1}...")
            
            # 모든 문명이 살아있어야 계속 진행
            all_alive = True
            for c in st.session_state.civs: 
                if not c.step(): # step이 False를 반환하면 (멸망/인구부족)
                    all_alive = False
            
            if not all_alive:
                status_text.warning("Simulation stopped early (Collapse/Critical Condition).")
                break
                
            progress.progress((i+1)/10)
            time.sleep(0.01)
            
        st.session_state.running = False
        st.rerun()
    
    civs = st.session_state.civs
    cols = st.columns(len(civs))
    for idx, civ in enumerate(civs):
        with cols[idx]:
            st.subheader(civ.name)
            st.metric("Pop", len(civ.citizens))
            st.metric("Trust", f"{civ.trust:.2f}")
            with st.expander("Recent Events"):
                for e in list(civ.event_log)[-5:]:
                    st.text(e)
    
    df_list = []
    for c in civs:
        # History Length Mismatch 방지 및 빈 데이터 처리
        if len(c.history['year']) > 0:
            history_dict = {k: list(v) for k, v in c.history.items()}
            # 혹시 모를 길이 불일치 발생 시, 가장 짧은 길이에 맞춤 (Safe Truncate)
            min_len = min(len(v) for v in history_dict.values())
            history_dict = {k: v[:min_len] for k, v in history_dict.items()}
            
            d = pd.DataFrame(history_dict)
            d['Civ'] = c.name
            df_list.append(d)
    
    if df_list:
        df_all = pd.concat(df_list)
        fig = px.line(df_all, x='year', y=['trust', 'avg_wealth', 'population', 'tech'], 
                      color='Civ', facet_col='variable', facet_col_wrap=2,
                      height=700,
                      title="Evolution of Civilization (v4.9 Ultimate)")
        fig.update_yaxes(matches=None)
        st.plotly_chart(fig, use_container_width=True)
