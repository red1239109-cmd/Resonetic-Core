# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 red1239109-cmd
# ------------------------------------------------------------------------------
# Resonetics v4.2 [Evolving Swarm] : Global Evolution + Coop Metric + Visual Risk
# ------------------------------------------------------------------------------
import torch, torch.nn as nn, numpy as np
import matplotlib.pyplot as plt, matplotlib.animation as animation
from dataclasses import dataclass, field
from copy import deepcopy
from collections import Counter

@dataclass
class Config:
    size: int = 60
    n_agents: int = 30
    view: int = 7
    dim: int = 8
    hidden: int = 16
    # [Evolution]
    evo_interval: int = 50     # 진화 주기
    mutation_scale: float = 0.02
    
    cost: dict = field(default_factory=lambda: {'move': 0.5, 'emg': 0.1, 'rep': 40.0})
    reward: dict = field(default_factory=lambda: {'clean': 3.0})
    th: dict = field(default_factory=lambda: {'rep': 80.0, 'min': 15, 'max': 80})
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
        return torch.softmax(self.out(y[:, -1]), dim=-1), h_new

    def mutate(self, scale=0.01):
        with torch.no_grad():
            for p in self.parameters():
                p.add_(torch.randn_like(p) * scale)

class Agent:
    def __init__(self, id, r, c):
        self.id, self.r, self.c, self.e, self.age = id, r, c, 50.0, 0
        # [Fix 1] LSTM Hidden 초기화 누수 방지: 생성 시 명시적 할당
        self.h = (torch.zeros(1, 1, CFG.hidden), torch.zeros(1, 1, CFG.hidden)) 
        self.act = 0

class Engine:
    def __init__(self):
        self.grid = np.zeros((CFG.size, CFG.size), dtype=np.float32)
        
        # [Fix 2] Global Evolution System
        self.brain = Brain()          # 현재 뇌
        self.best_brain = deepcopy(self.brain) # 백업용(롤백)
        self.best_score = -9999.0
        
        self.agents = [Agent(i, np.random.randint(0, CFG.size), np.random.randint(0, CFG.size)) for i in range(CFG.n_agents)]
        self.ids, self.gen, self.risk = CFG.n_agents, 0, 0.0
        self.stats = {'emg': 0, 'ent_prev': 0, 'pop_prev': CFG.n_agents, 'coop': 0.0}

    def get_views(self):
        padded = np.pad(self.grid, CFG.view//2, mode='wrap')
        views = [padded[a.r:a.r+CFG.view, a.c:a.c+CFG.view] for a in self.agents]
        return torch.tensor(np.array(views), dtype=torch.float32).unsqueeze(1)

    def evolve_swarm(self):
        # [Bonus Tip] 진화하는 Swarm: 점수가 안 좋으면 롤백, 좋으면 저장 후 돌연변이
        avg_energy = np.mean([a.e for a in self.agents]) if self.agents else 0
        score = avg_energy # 평가 기준: 평균 에너지 (생존력)
        
        status = "KEEP"
        if score > self.best_score:
            self.best_score = score
            self.best_brain = deepcopy(self.brain) # 좋은 뇌 저장
            status = "IMPROVED"
        else:
            self.brain = deepcopy(self.best_brain) # 나쁜 뇌 폐기 (롤백)
            status = "ROLLBACK"
            
        # 돌연변이 시도
        self.brain.mutate(CFG.mutation_scale)
        return status, score

    def step(self):
        self.gen += 1; self.stats['emg'] = 0
        self.agents = [a for a in self.agents if a.e > 0]
        
        # Repopulate
        while len(self.agents) < CFG.th['min']:
            self.ids += 1
            self.agents.append(Agent(self.ids, np.random.randint(0, CFG.size), np.random.randint(0, CFG.size)))

        # 1. Perception (Batch)
        device = next(self.brain.parameters()).device
        views = self.get_views().to(device)
        
        # [Fix 1] 이미 초기화된 hidden 사용
        h_batch = torch.cat([a.h[0] for a in self.agents], dim=1)
        c_batch = torch.cat([a.h[1] for a in self.agents], dim=1)
            
        probs, (hn, cn) = self.brain(views, (h_batch, c_batch))
        
        actions = torch.argmax(probs, dim=1).tolist()
        for i, a in enumerate(self.agents):
            a.act = actions[i]
            a.h = (hn[:, i:i+1], cn[:, i:i+1])

        # 2. Physics & Logic
        moves = [((-1,0), (1,0), (0,-1), (0,1))[a.act] if a.act < 4 else (0,0) for a in self.agents]
        occupied = {(a.r, a.c): a for a in self.agents}
        cleaners, breeders = [], []
        
        for a, (dr, dc) in zip(self.agents, moves):
            # Emergency Override
            if a.e <= CFG.cost['move'] + 0.1 and self.grid[a.r, a.c] > 0.1: 
                a.act = 4; self.stats['emg'] += 1
            
            if a.act < 4: # Move
                nr, nc = (a.r + dr) % CFG.size, (a.c + dc) % CFG.size
                if (nr, nc) not in occupied:
                    del occupied[(a.r, a.c)]
                    a.r, a.c = nr, nc
                    occupied[(nr, nc)] = a
                    a.e -= CFG.cost['move']
                else: a.e -= 0.1 # Collision penalty
            
            elif a.act == 4: cleaners.append(a)
            elif a.act == 5 and a.e > CFG.th['rep']: breeders.append(a)
            a.e -= 0.1; a.age += 1

        # 3. Environment & Coop Metric
        self.grid = np.clip(self.grid + np.random.rand(CFG.size, CFG.size) * 0.005, 0, 1)
        
        # [Bonus Tip] 협력 지표: 같은 셀을 동시에 청소하는 경우
        clean_counts = Counter([(a.r, a.c) for a in cleaners])
        coop_events = sum(1 for count in clean_counts.values() if count > 1)
        self.stats['coop'] = coop_events / max(len(cleaners), 1) if cleaners else 0.0

        for a in cleaners:
            dirt = min(self.grid[a.r, a.c], 0.5)
            # 협력 시 효율 증가 로직은 없지만, 보상은 나눠가짐 (경쟁적 협력)
            cnt = clean_counts[(a.r, a.c)]
            self.grid[a.r, a.c] -= dirt / cnt 
            a.e += (dirt * CFG.reward['clean']) / cnt

        # 4. Evolution Step (Global)
        evo_log = ""
        if self.gen % CFG.evo_interval == 0:
            status, score = self.evolve_swarm()
            evo_log = f"EVO: {status} ({score:.1f})"

        # 5. Reproduction (Swarm Clone)
        if len(self.agents) < CFG.th['max']:
            for p in sorted(breeders, key=lambda x: x.e, reverse=True):
                if len(self.agents) >= CFG.th['max']: break
                p.e -= CFG.cost['rep']; self.ids += 1
                self.agents.append(Agent(self.ids, p.r, p.c)) # 뇌는 공유되므로 몸만 복제

        # 6. Risk Metrics
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
# Visualization
# ---------------------------------------------------------
def run():
    sim = Engine()
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#222')
    ax.axis('off')
    im = ax.imshow(sim.grid, cmap='magma', vmin=0, vmax=1)
    # [Fix 4] 시각화 단조 개선: 테두리 색상으로 위험도 표시
    scat = ax.scatter([], [], s=60, c=[], edgecolors=[], linewidths=1.5, cmap='RdYlGn', vmin=0, vmax=100)
    txt = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='w', fontfamily='monospace', fontweight='bold')
    
    # [Bonus Tip] Evolution Log 표시용 텍스트
    evo_txt = ax.text(0.02, 0.02, '', transform=ax.transAxes, color='cyan', fontsize=9, fontfamily='monospace')

    def update(f):
        g, ag, evo_log = sim.step()
        im.set_data(g)
        
        if evo_log: evo_txt.set_text(evo_log) # 진화 로그 업데이트
        
        if ag:
            scat.set_offsets(np.c_[[a.c for a in ag], [a.r for a in ag]])
            scat.set_array(np.array([a.e for a in ag]))
            
            # [Fix 4] Risk에 따른 Edge Color 변화 (Green -> Yellow -> Red)
            edge_c = 'lime'
            if sim.risk > CFG.risk['th_col']: edge_c = 'red'
            elif sim.risk > 0.45: edge_c = 'yellow'
            scat.set_edgecolors(edge_c)
        
        verdict = "COLLAPSE" if sim.risk > CFG.risk['th_col'] else "STABLE"
        # [Bonus Tip] COOP 지표 추가
        status = (f"GEN: {sim.gen:<4} POP: {len(ag):<3} ENT: {np.sum(g):.1f}\n"
                  f"RISK: {sim.risk:.3f} [{verdict}]\n"
                  f"COOP: {sim.stats['coop']:.2f}  EMG: {sim.stats['emg']}")
        txt.set_text(status)
        return im, scat, txt, evo_txt

    # [Bonus Tip] 속도 튜닝 (interval 30 -> 10)
    ani = animation.FuncAnimation(fig, update, interval=10, blit=True)
    plt.show()

if __name__ == "__main__":
    run()

