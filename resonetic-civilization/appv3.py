# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 red1239109-cmd
# ==============================================================================
# File: resonetic_civilization_v12_0_singularity.py
# Product: Resonetic v12.0 (The Quantum Singularity)
# Core: Schrödinger Evolution, Entanglement, Tunneling, Collapse
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from collections import deque
import random
import math

# Constants
IDX_WEALTH = 0; IDX_HAPPINESS = 1; IDX_AGE = 2; IDX_FAITH = 3
IDX_AMBITION = 4; IDX_CONFORMITY = 5; DIM_AGENT = 6

if torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")

# ------------------------------------------------------------------------------
# 1. Singularity Engine (The Quantum Core)
# ------------------------------------------------------------------------------
class SingularityEngine:
    def __init__(self, device):
        self.device = device
        # Observables (Hermitian Operators)
        self.E = torch.eye(2, device=device)
        self.I = torch.eye(2, device=device)
        self.T = torch.eye(2, device=device)
        self.Eco = torch.eye(2, device=device)
        self.wave_function_log = [] # Track collapse events

    def update_states(self, agents, trust, tech, gini):
        # Update observable states based on micro-data
        self.E[0,0] = 1.0-gini; self.E[1,1] = gini
        self.E[0,1] = self.E[1,0] = gini * 0.2 # Coupling
        
        self.I[0,0] = trust; self.I[1,1] = 1.0-trust
        self.I[0,1] = self.I[1,0] = (1.0-trust) * 0.2
        
        nt = math.tanh(tech/100.0)
        self.T[0,0] = 1.0-nt; self.T[1,1] = nt
        
        # Simple Ecology for v12 focus
        self.Eco[0,0] = 0.8; self.Eco[1,1] = 0.2

    def calculate_entanglement(self):
        """[Innovation 1] Von Neumann Entropy of Entanglement"""
        # Density Matrix (Normalized)
        rho_E = self.E / (torch.trace(self.E) + 1e-8)
        rho_I = self.I / (torch.trace(self.I) + 1e-8)
        
        # Kronecker Product (Combined State)
        combined_rho = torch.kron(rho_E, rho_I)
        
        # Partial Trace (Tracing out Ideology to see Economy's entanglement)
        # Reshape to (2,2,2,2) and trace axes 1 and 3
        reshaped = combined_rho.view(2, 2, 2, 2)
        partial_trace = torch.einsum('ijkl->ik', reshaped) # Simplified partial trace
        
        # Von Neumann Entropy: S = -Tr(rho * ln(rho))
        eigvals = torch.linalg.eigvalsh(partial_trace)
        eigvals = torch.clamp(eigvals, min=1e-10) # Avoid log(0)
        entropy = -torch.sum(eigvals * torch.log(eigvals))
        return entropy.item()

    def schrodinger_evolution(self, dt=0.1):
        """[Innovation 2] Time Evolution U(t) = exp(-iHt)"""
        # Hamiltonian (Total Energy Operator)
        H = self.E + self.I + self.T + self.Eco
        
        # Eigen decomposition for matrix exponential
        eigvals, eigvecs = torch.linalg.eigh(H)
        
        # Evolution Operator (Unitary)
        # U = V * diag(e^-iEt) * V_dagger
        # We simulate the flow of the 'Economy' state under this Hamiltonian
        evolution_diag = torch.diag(torch.exp(-1j * eigvals * dt))
        # Complex matrix multiplication: U = P * D * P^-1
        U = torch.matmul(torch.matmul(eigvecs.to(torch.cfloat), evolution_diag), eigvecs.T.to(torch.cfloat))
        
        # Evolve State: E(t+dt) = U * E(t) * U_dagger
        E_complex = self.E.to(torch.cfloat)
        E_next = torch.matmul(torch.matmul(U, E_complex), U.conj().T)
        
        # Project back to Real (Measurement)
        self.E = E_next.real
        return torch.trace(self.E).item()

    def quantum_tunneling(self, agents):
        """[Innovation 3] Tunneling through high uncertainty"""
        # WKB Approximation: Probability ~ exp(-Barrier)
        # Barrier is approximated by 'Social Inertia' (Conformity)
        barrier = agents[:, IDX_CONFORMITY].mean().item() * 10.0
        prob = math.exp(-barrier)
        
        tunneled = False
        if random.random() < prob:
            # Sudden shift in Faith (Tunneling Event)
            mask = torch.rand(len(agents), device=self.device) < 0.1 # 10% affected
            agents[mask, IDX_FAITH] = 1.0 - agents[mask, IDX_FAITH]
            tunneled = True
            
        return tunneled, prob

    def quantum_collapse(self, measurement_type='election'):
        """[Innovation 4] Wave Function Collapse"""
        # Forcing the superposition to pick a diagonal state
        if measurement_type == 'election':
            # Soft Collapse (Decoherence)
            self.I.fill_diagonal_(self.I.trace() / 2.0)
            self.I[0,1] = self.I[1,0] = 0.0 # Remove interference terms
            return "Election: Ideology Collapsed"
        return "No Event"

# ------------------------------------------------------------------------------
# 2. Supporting Micro-Structure (Optimized)
# ------------------------------------------------------------------------------
class SocialEngine:
    def __init__(self, pop_size, device):
        self.graph = nx.barabasi_albert_graph(n=pop_size, m=5, seed=42)
        adj = nx.to_scipy_sparse_array(self.graph, format='coo')
        indices = torch.LongTensor(np.vstack((adj.row, adj.col)))
        values = torch.FloatTensor(adj.data)
        degrees = torch.tensor([d for n, d in self.graph.degree()], dtype=torch.float32, device=device)
        self.degrees = degrees.unsqueeze(1) + 1e-8 
        if device.type == 'cuda':
            indices = indices.cuda(); values = values.cuda()
            self.adj_sparse = torch.sparse_coo_tensor(indices, values, (pop_size, pop_size)).cuda()
        else:
            self.adj_sparse = torch.sparse_coo_tensor(indices, values, (pop_size, pop_size)).to(device)
    def spread_influence(self, traits):
        neighbor_sum = torch.sparse.mm(self.adj_sparse, traits)
        return 0.7 * traits + 0.3 * (neighbor_sum / self.degrees)

class Brain(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(input_dim), nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x): return self.net(x)

# ------------------------------------------------------------------------------
# 3. Civilization V12 (The Singularity)
# ------------------------------------------------------------------------------
class CivilizationV12:
    def __init__(self, pop_size):
        self.pop_size = pop_size; self.device = device
        self.agents = torch.zeros(pop_size, DIM_AGENT, device=device)
        self._init_agents()
        self.year = 0; self.trust = 0.5; self.tech_level = 1.0
        
        self.social = SocialEngine(pop_size, device)
        self.singularity = SingularityEngine(device)
        self.brain = Brain(7).to(device)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)
        
        self.history = {'year':[], 'trust':[], 'entanglement':[], 'tunneling_prob':[]}
        self.payoff = torch.tensor([[1.5, -2.0], [3.0, -0.5]], device=device)

    def _init_agents(self):
        self.agents[:, IDX_WEALTH] = torch.rand(self.pop_size, device=self.device) * 5.0 + 1.0
        self.agents[:, IDX_HAPPINESS] = 0.5
        self.agents[:, IDX_FAITH] = torch.rand(self.pop_size, device=self.device)
        self.agents[:, IDX_AMBITION] = torch.rand(self.pop_size, device=self.device)
        self.agents[:, IDX_CONFORMITY] = torch.rand(self.pop_size, device=self.device)

    def get_states(self):
        t_col = torch.full((self.pop_size, 1), self.trust, device=self.device)
        return torch.cat([t_col, self.agents], dim=1)

    def step(self):
        self.year += 1
        
        # 1. Micro to Macro (Update Observables)
        gini = self.calculate_gini()
        self.singularity.update_states(self.agents, self.trust, self.tech_level, gini)
        
        # 2. Quantum Dynamics
        # A. Entanglement (Social Cohesion)
        entanglement = self.singularity.calculate_entanglement()
        
        # B. Schrödinger Evolution (Flow of History)
        self.singularity.schrodinger_evolution(dt=0.1)
        
        # C. Quantum Tunneling (Crisis/Revolution)
        is_tunneled, t_prob = self.singularity.quantum_tunneling(self.agents)
        event_msg = "Tunneling Event!" if is_tunneled else None
        
        # D. Wave Function Collapse (Election every 4 years)
        if self.year % 4 == 0:
            collapse_msg = self.singularity.quantum_collapse('election')
            event_msg = collapse_msg
            
        # 3. Agent Decision & Social
        traits = self.agents[:, [IDX_HAPPINESS, IDX_FAITH, IDX_AMBITION]]
        self.agents[:, [IDX_HAPPINESS, IDX_FAITH, IDX_AMBITION]] = self.social.spread_influence(traits)
        
        # Simple decision logic for stability in V12
        states = self.get_states()
        with torch.no_grad(): q = self.brain(states)
        action = torch.argmax(q, dim=1)
        
        # Interaction & Learning (Simplified)
        idx = torch.randperm(self.pop_size, device=self.device)
        p1, p2 = idx[:self.pop_size//2], idx[self.pop_size//2:self.pop_size//2*2]
        a1, a2 = action[p1], action[p2]
        r1 = self.payoff[a1, a2]; r2 = self.payoff[a2, a1]
        self.agents[p1, IDX_WEALTH] += r1; self.agents[p2, IDX_WEALTH] += r2
        
        # Train (Target logic omitted for brevity in V12 focus)
        
        self.history['year'].append(self.year)
        self.history['trust'].append(self.trust)
        self.history['entanglement'].append(entanglement)
        self.history['tunneling_prob'].append(t_prob)
        
        return event_msg

    def calculate_gini(self):
        # Quick approximation
        w = self.agents[:, IDX_WEALTH]
        return (w.std() / (w.mean() + 1e-8)).item()

# ------------------------------------------------------------------------------
# 4. UI
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Resonetic v12.0", layout="wide", page_icon="🌌")
    st.title("🌌 Resonetic v12.0: The Quantum Singularity")
    
    if 'civ' not in st.session_state: st.session_state.civ = None
    if 'auto_run' not in st.session_state: st.session_state.auto_run = False
    
    with st.sidebar:
        if st.button("🚀 Big Bang"):
            st.session_state.civ = CivilizationV12(2000)
            st.success("Universe Born.")
        
        if st.button("▶️/⏸️ Toggle Run"):
            st.session_state.auto_run = not st.session_state.auto_run
            st.rerun()

    if st.session_state.civ:
        civ = st.session_state.civ
        msg = None
        
        if st.session_state.auto_run:
            msg = civ.step()
            time.sleep(0.05)
            st.rerun()
            
        # Display
        c1, c2, c3 = st.columns(3)
        c1.metric("Year", civ.year)
        c2.metric("Entanglement", f"{civ.history['entanglement'][-1]:.4f}" if civ.history['entanglement'] else "0")
        c3.metric("Tunneling Prob", f"{civ.history['tunneling_prob'][-1]:.2e}" if civ.history['tunneling_prob'] else "0")
        
        if msg: st.toast(f"⚡ {msg}")
        
        if civ.year > 0:
            df = pd.DataFrame(civ.history)
            fig = px.line(df, x='year', y=['trust', 'entanglement', 'tunneling_prob'], 
                          log_y=True, title="Quantum Historical Trajectories")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
