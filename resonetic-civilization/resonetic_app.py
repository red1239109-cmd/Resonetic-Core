# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 red1239109-cmd
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
# File: resonetic_civilization_v3_3.py
# Product: Resonetic Civilization v3.3 (Network & Crisis)
# Features: Social Network (Peer Pressure), External Shocks (Black Swans)
# ==============================================================================

import streamlit as st
import random
import numpy as np
import pandas as pd
import time
import uuid
import plotly.express as px
from collections import deque
from scipy import stats

# ------------------------------------------------------------------------------
# 1. System Components
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
        wealths = sorted([c.wealth for c in citizens])
        n = len(wealths)
        if n == 0 or sum(wealths) == 0: return 0.0
        cumulative = sum(w * (i + 1) for i, w in enumerate(wealths))
        return (2 * cumulative) / (n * sum(wealths)) - (n + 1) / n

# ------------------------------------------------------------------------------
# 2. Networked Agent Model
# ------------------------------------------------------------------------------
class NetworkedCitizen:
    def __init__(self, trait='Aggressive', initial_wealth=5.0, generation=1):
        self.id = str(uuid.uuid4())[:4]
        self.trait = trait
        self.mindset = trait
        self.wealth = initial_wealth
        self.age = 0
        self.generation = generation
        self.happiness = 0.5
        self.memory = deque(maxlen=5)
        
        # [New] Social Network
        self.neighbors = [] # List of other citizens
        self.last_action = None # To influence neighbors
        
    def decide(self, global_trust):
        # 1. Peer Pressure (Network Effect)
        peer_effect = 0.0
        if self.neighbors:
            # 이웃들이 지난번에 무엇을 했는가?
            neighbor_actions = [n.last_action for n in self.neighbors if n.last_action]
            if neighbor_actions:
                coop_rate = neighbor_actions.count('COOP') / len(neighbor_actions)
                # 이웃의 60% 이상이 협력자면 나도 협력 성향 증가
                peer_effect = -0.2 if coop_rate > 0.6 else 0.2
        
        # 2. Experience & Happiness
        mem_effect = 0.0
        if self.memory:
            coop_ratio = self.memory.count('COOP') / len(self.memory)
            mem_effect = -0.2 if coop_ratio > 0.6 else 0.2
            
        happy_effect = -0.2 if self.happiness > 0.7 else 0.1
        
        # 3. Base & Poverty
        base_prob = 0.8 if self.mindset == 'Aggressive' else 0.1
        poverty_effect = 0.4 if self.wealth < 3.0 else 0.0
        social_effect = -(global_trust * 0.3)
        
        # Combine all factors
        final_prob = base_prob + poverty_effect + social_effect + mem_effect + happy_effect + peer_effect
        
        decision = "DEFECT" if random.random() < np.clip(final_prob, 0.05, 0.95) else "COOP"
        self.last_action = decision
        return decision

    def update_state(self, outcome, wealth_change):
        self.memory.append(outcome)
        delta_h = (wealth_change / 10.0)
        if self.wealth < 2.0: delta_h -= 0.1
        self.happiness = np.clip(self.happiness + delta_h, 0.0, 1.0)

# ------------------------------------------------------------------------------
# 3. Simulation Core with Shocks
# ------------------------------------------------------------------------------
class Civilization:
    def __init__(self, name, config):
        self.name = name
        self.year = 0
        self.trust = 0.5
        self.tech_level = 1.0
        self.event_log = []
        
        self.monitor = AdvancedMonitor()
        self.policy = PolicyEngine(config.get('budget_allocation'))
        self.tax_rate = config['tax_rate']
        
        # Init Population
        self.citizens = [
            NetworkedCitizen('Aggressive' if random.random() < config['initial_aggressive'] else 'Cooperative')
            for _ in range(config['pop_size'])
        ]
        
        # [New] Build Network (Small World-ish)
        self._build_network(k=4) 
        
        self.history = {
            'year': [], 'trust': [], 'population': [], 'avg_wealth': [], 
            'gini': [], 'happiness': [], 'tech': []
        }
        
    def _build_network(self, k=4):
        """Randomly connect citizens to create a social graph"""
        for c in self.citizens:
            # Pick k random friends (excluding self)
            candidates = [p for p in self.citizens if p != c]
            if len(candidates) >= k:
                c.neighbors = random.sample(candidates, k)
                
    def _trigger_shocks(self):
        """Random Event Generator (Black Swans)"""
        # 5% chance per year
        if random.random() < 0.05:
            event_type = random.choice(["Pandemic", "Financial Crisis", "Disinformation"])
            
            if event_type == "Pandemic":
                # Health Crisis: Population & Happiness drop
                deaths = 0
                for c in self.citizens:
                    c.happiness -= 0.3
                    if random.random() < 0.2: # 20% lethality without healthcare
                        c.wealth = -10 # Mark for death
                        deaths += 1
                self.event_log.append(f"Year {self.year}: 🦠 PANDEMIC! {deaths} died.")
                
            elif event_type == "Financial Crisis":
                # Wealth Shock: Everyone loses 40% wealth
                for c in self.citizens:
                    c.wealth *= 0.6
                self.event_log.append(f"Year {self.year}: 📉 MARKET CRASH! Wealth wiped out.")
                
            elif event_type == "Disinformation":
                # Trust Shock: Trust drops significantly
                self.trust = max(0.1, self.trust - 0.4)
                self.event_log.append(f"Year {self.year}: 🕵️ FAKE NEWS! Trust collapsed.")

    def step(self):
        self.year += 1
        
        # [New] External Shocks
        self._trigger_shocks()
        
        # 1. Interaction Phase
        random.shuffle(self.citizens)
        multiplier = 1.0 + (np.log(self.tech_level) * 0.5)
        rewards = {('COOP','COOP'):(1.5,1.5), ('DEFECT','DEFECT'):(-0.5,-0.5), 
                   ('DEFECT','COOP'):(3.0,-2.0), ('COOP','DEFECT'):(-2.0,3.0)}
        
        crimes = 0
        pairs = len(self.citizens) // 2
        for i in range(pairs):
            p1, p2 = self.citizens[2*i], self.citizens[2*i+1]
            a1, a2 = p1.decide(self.trust), p2.decide(self.trust)
            
            r1, r2 = rewards[(a1, a2)]
            p1.wealth += r1 * multiplier
            p2.wealth += r2 * multiplier
            
            p1.update_state('COOP' if a2=='COOP' else 'DEFECT', r1*multiplier)
            p2.update_state('COOP' if a1=='COOP' else 'DEFECT', r2*multiplier)
            
            if a1=='DEFECT': crimes+=1
            if a2=='DEFECT': crimes+=1
            
        # 2. Government Phase
        tax_revenue = 0
        for c in self.citizens:
            tax = max(0, c.wealth * self.tax_rate)
            c.wealth -= tax
            tax_revenue += tax
            
        self.policy.allocate(tax_revenue)
        
        # Policy Execution (Simplified for brevity, same logic as v3.2)
        green_budget = self.policy.get_budget('Green')
        if green_budget > 0:
            env_bonus = np.log1p(green_budget) * 0.08
            for c in self.citizens: c.happiness = np.clip(c.happiness + env_bonus, 0.0, 1.0)

        welfare_budget = self.policy.get_budget('Welfare')
        poor = [c for c in self.citizens if c.wealth < 3.0]
        if poor and welfare_budget > 0:
            subsidy = welfare_budget / len(poor)
            for c in poor: c.wealth += subsidy
            
        edu_budget = self.policy.get_budget('Education')
        candidates = [c for c in self.citizens if c.mindset == 'Aggressive']
        while edu_budget >= 8.0 and candidates:
            s = candidates.pop()
            edu_budget -= 8.0
            if random.random() < 0.6: s.mindset = 'Cooperative'
        
        tech_budget = self.policy.get_budget('Tech') + edu_budget
        if random.random() < (1.0 - np.exp(-tech_budget/50.0)): self.tech_level += 0.2
            
        # 3. Mortality & Reproduction (with Network Maintenance)
        survivors = []
        new_borns = []
        for c in self.citizens:
            c.wealth -= 0.5
            c.age += 1
            if c.wealth > 0 and random.random() > (c.age/150.0):
                survivors.append(c)
                if c.wealth > 15.0:
                    c.wealth -= 5.0
                    new_borns.append(NetworkedCitizen(c.trait, 1.0, c.generation+1))
        
        # Update citizens and rebuild network for new/removed agents
        self.citizens = survivors + new_borns
        # [Optimization] Only rebuild network if pop changed significantly or randomly
        if random.random() < 0.3: self._build_network(k=4)
        
        # 4. Metrics Update
        crime_rate = crimes / max(1, len(self.citizens))
        def_budget = self.policy.get_budget('Defense')
        sec_factor = 1.0 / (1.0 + np.log1p(def_budget) * 0.5)
        
        trust_change = 0.05 if crime_rate < 0.2 else -0.1 * sec_factor
        self.trust = np.clip(self.trust + trust_change, 0.0, 1.0)
        
        self.history['year'].append(self.year)
        self.history['trust'].append(self.trust)
        self.history['population'].append(len(self.citizens))
        self.history['avg_wealth'].append(np.mean([c.wealth for c in self.citizens]) if self.citizens else 0)
        self.history['gini'].append(self.monitor.calculate_gini(self.citizens))
        self.history['happiness'].append(np.mean([c.happiness for c in self.citizens]) if self.citizens else 0)
        self.history['tech'].append(self.tech_level)
        
        return len(self.citizens) > 2

# ------------------------------------------------------------------------------
# 4. UI & Dashboard
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Resonetic v3.3", layout="wide", page_icon="⚡")
st.title("⚡ Resonetic v3.3 (Crisis Lab)")

with st.sidebar:
    st.header("🔬 Experiment Setup")
    mode = st.radio("Mode", ["Single Run", "A/B Test"])
    
    st.subheader("Policy Budget Allocation")
    p_edu = st.slider("Education", 0.0, 1.0, 0.3)
    p_wel = st.slider("Welfare", 0.0, 1.0, 0.2)
    p_tech = st.slider("Technology", 0.0, 1.0, 0.3)
    p_green = st.slider("Green (Eco)", 0.0, 1.0, 0.1)
    p_def = st.slider("Defense", 0.0, 1.0, 0.1)
    
    total = p_edu + p_wel + p_tech + p_green + p_def
    if total == 0: total = 1.0
    allocation = {
        'Education': p_edu/total, 'Welfare': p_wel/total, 'Tech': p_tech/total,
        'Green': p_green/total, 'Defense': p_def/total
    }
    
    base_config = {
        "tax_rate": 0.15, "initial_aggressive": 0.5, "pop_size": 60,
        "budget_allocation": allocation
    }

    if st.button("🚀 Initialize"):
        if mode == "Single Run":
            st.session_state.civs = [Civilization("World A", base_config)]
        else:
            config_b = base_config.copy()
            # B: High Resilience Policy (High Welfare & Defense)
            config_b['budget_allocation'] = {'Education': 0.2, 'Welfare': 0.4, 'Tech': 0.1, 'Green': 0.1, 'Defense': 0.2}
            st.session_state.civs = [
                Civilization("Growth Focused", base_config),
                Civilization("Resilience Focused", config_b)
            ]

if 'civs' in st.session_state:
    col1, col2 = st.columns([1, 5])
    if col1.button("▶️ Step"):
        for c in st.session_state.civs: c.step()
    if col2.button("⏩ Run 20 Years"):
        progress = st.progress(0)
        for i in range(20):
            for c in st.session_state.civs: c.step()
            progress.progress((i+1)/20)
            time.sleep(0.05)
            
    # Dashboard
    civs = st.session_state.civs
    cols = st.columns(len(civs))
    for idx, civ in enumerate(civs):
        with cols[idx]:
            st.subheader(f"{civ.name}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Pop", len(civ.citizens))
            m2.metric("Trust", f"{civ.trust:.2f}")
            m3.metric("Wealth", f"{civ.history['avg_wealth'][-1]:.1f}" if civ.history['avg_wealth'] else "0.0")
            
            with st.expander("🚨 Event Log", expanded=True):
                if civ.event_log:
                    for e in civ.event_log[-5:]:
                        if "PANDEMIC" in e or "CRASH" in e or "FAKE" in e:
                            st.error(e)
                        else:
                            st.text(e)
                else:
                    st.caption("No events yet.")

    st.divider()
    tabs = st.tabs(["📈 Comparative Trends", "💾 Data Export"])
    
    with tabs[0]:
        df_list = []
        for c in civs:
            d = pd.DataFrame(c.history)
            d['Civilization'] = c.name
            df_list.append(d)
        
        if df_list:
            df_all = pd.concat(df_list)
            fig = px.line(df_all, x='year', y=['trust', 'avg_wealth', 'population'], 
                          color='Civilization', facet_row='variable', height=700,
                          title="Impact of Crisis on Different Societies")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        if df_list:
            final_df = pd.concat(df_list)
            cols = ['year', 'Civilization', 'population', 'trust', 'avg_wealth', 'gini', 'happiness', 'tech']
            valid_cols = [c for c in cols if c in final_df.columns]
            final_df = final_df[valid_cols + [c for c in final_df.columns if c not in valid_cols]]
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Data", csv, "resonetic_crisis_data.csv", "text/csv")
