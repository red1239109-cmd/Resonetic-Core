# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd
# ------------------------------------------------------------------------------
# Resonetics v5.1 [Synergy & Classes] : True Synergy + Special Abilities + HUD
# ------------------------------------------------------------------------------
import torch, torch.nn as nn, numpy as np
import matplotlib.pyplot as plt, matplotlib.animation as animation
from dataclasses import dataclass, field
from copy import deepcopy
from collections import Counter

@dataclass
class Config:
    size: int = 60
    n_agents: int = 40
    view: int = 7
    dim: int = 8
    hidden: int = 16
    
    # [Patch 1] 진정한 시너지 설정
    synergy: dict = field(default_factory=lambda: {
        'base': 0.3,         # 동료 1명당 보너스 30%
        'cleaner_bonus': 0.2 # 클리너는 시너지 더 잘 받음
    })
    
    # [Patch 2] 직업 밸런스
    ability: dict = field(default_factory=lambda: {
        'clean_eff': 1.5,    # Cleaner 청소 효율
        'move_disc': 0.5,    # Explorer 이동 비용 할인 (0.5배)
        'rep_disc': 0.8      # Breeder 번식 비용 할인 (0.8배)
    })

    # [System]
    temperature: float = 2.0 # Softmax 온도 (낮을수록 Bias 영향력 큼)
    evo_interval: int = 50
    mutation_scale: float = 0.02
    
    cost: dict = field(default_factory=lambda: {'move': 0.5, 'emg': 0.1, 'rep': 40.0})
    reward: dict = field(default_factory=lambda: {'clean': 3.0})
    th: dict = field(default_factory=lambda: {'rep': 80.0, 'min': 15, 'max': 100})
    risk: dict = field(default_factory=lambda: {'alpha': 0.15, 'th_col': 0.72})

CFG = Config()

class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, CFG.dim, 3), nn.ReLU(), nn.Flatten())
        with torch.no_grad(): n_flat = self.cnn(torch.zeros(1, 1, CFG.view, CFG.view)).shape[1]
        self.lstm = nn.LSTM(n_flat, CFG.hidden, batch_first=True)
        self.out = nn.Linear(CFG.hidden, 6)

    def forward(self, x, h): 
        feat = self.cnn(x).unsqueeze(1)
        y, h_new = self.lstm(feat, h)
        return self.out(y[:, -1]), h_new # Logits 반환

    def mutate(self, scale=0.01):
        with torch.no_grad():
            for p in self.parameters():
                p.add_(torch.randn_like(p) * scale)

class Agent:
    ROLES = ['Cleaner', 'Explorer', 'Breeder']
    # 시각화용 마커 및 색상
    MARKERS = ['o', '^', 's'] # 원, 삼각형, 사각형
    COLORS = ['#00aaff', '#00ff44', '#ff0044'] 
    
    def __init__(self, id, r, c, role=None):
        self.id, self.r, self.c, self.e, self.age = id, r, c, 50.0, 0
        self.h = (torch.zeros(1, 1, CFG.hidden), torch.zeros(1, 1, CFG.hidden))
        self.act = 0
        self.role_idx = role if role is not None else np.random.randint(0, 3)
        self.role = self.ROLES[self.role_idx]
        
        # [Patch 2] 강화된 편향 (Bias)
        self.bias = torch.zeros(6)
        if self.role_idx == 0:   self.bias[4] = 2.0   # Cleaner ++
        elif self.role_idx == 1: self.bias[:4] = 1.0  # Explorer ++ (이동)
        elif self.role_idx == 2: self.bias[5] = 2.0   # Breeder ++

class Engine:
    def __init__(self):
        self.grid = np.zeros((CFG.size, CFG.size), dtype=np.float32)
        self.brain = Brain()
        self.best_brain = deepcopy(self.brain)
        self.best_score = -9999.0
        self.agents = [Agent(i, np.random.randint(0, CFG.size), np.random.randint(0, CFG.size)) for i in range(CFG.n_agents)]
        self.ids, self.gen, self.risk = CFG.n_agents, 0, 0.0
        self.stats = {'emg': 0, 'ent_prev': 0, 'pop_prev': CFG.n_agents, 'coop': 0.0}

    def get_views(self):
        padded = np.pad(self.grid, CFG.view//2, mode='wrap')
        views = [padded[a.r:a.r+CFG.view, a.c:a.c+CFG.view] for a in self.agents]
        return torch.tensor(np.array(views), dtype=torch.float32).unsqueeze(1)

    def evolve_swarm(self):
        avg_energy = np.mean([a.e for a in self.agents]) if self.agents else 0
        score = avg_energy 
        status = "KEEP"
        if score > self.best_score:
            self.best_score = score
            self.best_brain = deepcopy(self.brain)
            status = "IMPROVED"
        else:
            self.brain = deepcopy(self.best_brain)
            status = "ROLLBACK"
        self.brain.mutate(CFG.mutation_scale)
        return status, score

    def step(self):
        self.gen += 1; self.stats['emg'] = 0
        self.agents = [a for a in self.agents if a.e > 0]
        
        while len(self.agents) < CFG.th['min']:
            self.ids += 1
            self.agents.append(Agent(self.ids, np.random.randint(0, CFG.size), np.random.randint(0, CFG.size)))

        # 1. Perception
        device = next(self.brain.parameters()).device
        views = self.get_views().to(device)
        h_batch = torch.cat([a.h[0] for a in self.agents], dim=1)
        c_batch = torch.cat([a.h[1] for a in self.agents], dim=1)
        
        logits, (hn, cn) = self.brain(views, (h_batch, c_batch))
        biases = torch.stack([a.bias for a in self.agents]).to(device)
        
        # [Patch 3] Temperature Softmax
        probs = torch.softmax((logits + biases) / CFG.temperature, dim=-1)
        
        actions = torch.argmax(probs, dim=1).tolist()
        for i, a in enumerate(self.agents):
            a.act = actions[i]
            a.h = (hn[:, i:i+1], cn[:, i:i+1])

        # 2. Physics & Abilities
        moves = [((-1,0), (1,0), (0,-1), (0,1))[a.act] if a.act < 4 else (0,0) for a in self.agents]
        occupied = {(a.r, a.c): a for a in self.agents}
        cleaners, breeders = [], []
        
        for a, (dr, dc) in zip(self.agents, moves):
            # Cost 계산 (Explorer 할인 적용)
            move_cost = CFG.cost['move'] * (CFG.ability['move_disc'] if a.role_idx == 1 else 1.0)
            
            if a.e <= move_cost + 0.1 and self.grid[a.r, a.c] > 0.1: 
                a.act = 4; self.stats['emg'] += 1
            
            if a.act < 4:
                nr, nc = (a.r + dr) % CFG.size, (a.c + dc) % CFG.size
                if (nr, nc) not in occupied:
                    del occupied[(a.r, a.c)]
                    a.r, a.c = nr, nc
                    occupied[(nr, nc)] = a
                    a.e -= move_cost
                else: a.e -= 0.1
            elif a.act == 4: cleaners.append(a)
            elif a.act == 5 and a.e > CFG.th['rep']: breeders.append(a)
            a.e -= 0.1; a.age += 1

        # 3. Environment & True Synergy
        self.grid = np.clip(self.grid + np.random.rand(CFG.size, CFG.size) * 0.005, 0, 1)
        
        clean_groups = Counter([(a.r, a.c) for a in cleaners])
        coop_events = sum(1 for cnt in clean_groups.values() if cnt > 1)
        self.stats['coop'] = coop_events / max(len(cleaners), 1) if cleaners else 0.0

        for a in cleaners:
            cnt = clean_groups[(a.r, a.c)]
            dirt = min(self.grid[a.r, a.c], 0.5)
            
            # [Patch 2] Cleaner 효율 증가
            efficiency = CFG.ability['clean_eff'] if a.role_idx == 0 else 1.0
            dirt_removed = dirt * efficiency
            
            # [Patch 1] 진정한 시너지 (개인당 보상 증가)
            synergy = min(1.0, CFG.synergy['base'] * (cnt - 1))
            if a.role_idx == 0: synergy += CFG.synergy['cleaner_bonus']
            
            # 보상 = (흙 * 기본보상) * (1 + 시너지)
            # N빵 하지 않음! 같이 하면 무조건 이득!
            reward = (dirt_removed * CFG.reward['clean']) * (1.0 + synergy)
            
            # 흙은 N빵으로 줄어듦 (물리적 제거)
            self.grid[a.r, a.c] -= dirt_removed / cnt 
            a.e += reward

        # 4. Evolution
        evo_log = ""
        if self.gen % CFG.evo_interval == 0:
            status, score = self.evolve_swarm()
            evo_log = f"EVO: {status} ({score:.1f})"

        # 5. Reproduction (Breeder Discount)
        if len(self.agents) < CFG.th['max']:
            for p in sorted(breeders, key=lambda x: x.e, reverse=True):
                if len(self.agents) >= CFG.th['max']: break
                
                # [Patch 2] Breeder 비용 할인
                rep_cost = CFG.cost['rep'] * (CFG.ability['rep_disc'] if p.role_idx == 2 else 1.0)
                
                if p.e > rep_cost:
                    p.e -= rep_cost; self.ids += 1
                    role = p.role_idx if np.random.rand() > 0.1 else None
                    self.agents.append(Agent(self.ids, p.r, p.c, role=role))

        # 6. Metrics
        ent = np.sum(self.grid)
        self.stats['grad'] = abs(ent - self.stats['ent_prev'])
        self.stats['vol'] = abs(len(self.agents) - self.stats['pop_prev'])
        s = lambda x: 1 / (1 + np.exp(-x))
        emg_ratio = self.stats['emg'] / max(len(self.agents), 1)
        raw_risk = (0.55 * s(3*(self.stats['grad']/15 - 0.5)) + 
                    0.25 * s(3*(self.stats['vol']/8 - 0.5)) + 
                    0.20 * s(4*(emg_ratio - 0.25)))
        self.risk = (1 - CFG.risk['alpha']) * self.risk + CFG.risk['alpha'] * raw_risk
        self.stats.update({'ent_prev': ent, 'pop_prev': len(self.agents)})
        
        return self.grid, self.agents, evo_log

# ---------------------------------------------------------
# Visualization (HUD Upgrade)
# ---------------------------------------------------------
def run():
    sim = Engine()
    # [Vis] 서브플롯: 왼쪽(그리드), 오른쪽(통계)
    fig = plt.figure(figsize=(12, 6), facecolor='#222')
    ax_grid = fig.add_subplot(1, 2, 1)
    ax_stat = fig.add_subplot(1, 2, 2)
    
    # Grid View
    ax_grid.axis('off')
    im = ax_grid.imshow(sim.grid, cmap='magma', vmin=0, vmax=1)
    
    # [Vis] 직업별 스캐터 (모양 다르게 하기 위해 3개 생성)
    scats = []
    for i in range(3): # 0:Cleaner(o), 1:Explorer(^), 2:Breeder(s)
        sc = ax_grid.scatter([], [], s=80, marker=Agent.MARKERS[i], 
                             edgecolors='w', c=Agent.COLORS[i], label=Agent.ROLES[i])
        scats.append(sc)
    
    # Stat View (Bar Chart)
    ax_stat.set_facecolor('#222')
    ax_stat.spines['bottom'].set_color('white')
    ax_stat.spines['left'].set_color('white')
    ax_stat.tick_params(axis='x', colors='white')
    ax_stat.tick_params(axis='y', colors='white')
    
    bars = ax_stat.bar(Agent.ROLES, [0,0,0], color=Agent.COLORS)
    ax_stat.set_ylim(0, CFG.th['max'])
    ax_stat.set_title("Role Distribution", color='white')
    
    txt = ax_grid.text(0.02, 0.95, '', transform=ax_grid.transAxes, color='w', fontfamily='monospace', fontweight='bold')
    evo_txt = ax_grid.text(0.02, 0.02, '', transform=ax_grid.transAxes, color='cyan', fontsize=9, fontfamily='monospace')

    def update(f):
        g, ag, evo_log = sim.step()
        im.set_data(g)
        if evo_log: evo_txt.set_text(evo_log)
        
        # Scatter Update by Role
        counts = [0, 0, 0]
        for i in range(3):
            role_agents = [a for a in ag if a.role_idx == i]
            counts[i] = len(role_agents)
            if role_agents:
                scats[i].set_offsets(np.c_[[a.c for a in role_agents], [a.r for a in role_agents]])
                # Risk Edge Color
                ec = 'lime'
                if sim.risk > CFG.risk['th_col']: ec = 'red'
                elif sim.risk > 0.45: ec = 'yellow'
                scats[i].set_edgecolors(ec)
            else:
                scats[i].set_offsets(np.zeros((0, 2)))

        # Bar Chart Update
        for bar, h in zip(bars, counts):
            bar.set_height(h)
        
        verdict = "COLLAPSE" if sim.risk > CFG.risk['th_col'] else "STABLE"
        status = (f"GEN: {sim.gen:<4} POP: {len(ag):<3}\n"
                  f"RISK: {sim.risk:.3f} [{verdict}]\n"
                  f"COOP: {sim.stats['coop']:.2f} EMG: {sim.stats['emg']}")
        txt.set_text(status)
        return [im, txt, evo_txt] + scats + list(bars)

    ani = animation.FuncAnimation(fig, update, interval=20, blit=False) # blit=False required for ax redraw
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()

