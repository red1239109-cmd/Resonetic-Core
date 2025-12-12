# ==============================================================================
# File: resonetics_core.py
# Project: Resonetics - The Thinking Process in Silicon
# Author: red1239109-cmd
# License: AGPL-3.0
# ==============================================================================

"""
RESONETICS CORE
---------------
The fundamental building blocks for philosophically-informed AI.

Three Principles:
1. Structure (Plato)    - Reality has mathematical patterns
2. Flow (Heraclitus)    - Everything is in constant motion  
3. Humility (Socrates)  - True wisdom knows its limits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. PHILOSOPHICAL PRIMITIVES
# ==============================================================================

def snap_to_structure(x: torch.Tensor, base: float = 3.0) -> torch.Tensor:
    """Plato's Forms: Projects reality onto mathematical ideals."""
    return torch.round(x / base) * base

def harmonic_flow(x: torch.Tensor, period: float = 3.0) -> torch.Tensor:
    """Heraclitus' Flux: Models reality's oscillatory nature."""
    return torch.sin(2 * math.pi * x / period).pow(2)

def epistemic_humility(x: torch.Tensor, min_val: float = 0.1, max_val: float = 5.0) -> torch.Tensor:
    """Socrates' Wisdom: Knowledge exists between extremes."""
    return torch.clamp(x, min=min_val, max=max_val)

# ==============================================================================
# 2. THE SOVEREIGN LOSS
# ==============================================================================

class SovereignLoss(nn.Module):
    """The 8-layer cognitive landscape that auto-balances competing truths."""
    
    def __init__(self):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(8))  # Auto-balancing
        
    def forward(self, thought: torch.Tensor, doubt: torch.Tensor, 
                reality: torch.Tensor, memory: torch.Tensor = None):
        """
        Args:
            thought: Current understanding (B, *)
            doubt: Uncertainty about that understanding (B, *)
            reality: Ground truth (B, *)
            memory: Previous understanding (for temporal consistency)
        """
        # Layer 1: Empirical Accuracy
        L1 = (thought - reality).pow(2)
        
        # Layer 2: Cosmic Resonance
        L2 = harmonic_flow(thought)
        
        # Layer 5: Structural Alignment
        ideal = snap_to_structure(thought)
        L5 = (thought - ideal).pow(2)
        
        # Layer 6: Dialectical Tension
        reality_ideal = snap_to_structure(reality)
        tension = torch.abs(reality - reality_ideal)
        L6 = torch.tanh(tension) * 10.0
        
        # Layer 7: Temporal Consistency
        L7 = (thought - memory).pow(2) if memory is not None else torch.zeros_like(L1)
        
        # Layer 8: Epistemic Humility
        bounded_doubt = epistemic_humility(doubt)
        L8 = 0.5 * torch.log(bounded_doubt.pow(2)) + L1 / (2 * bounded_doubt.pow(2))
        
        # Auto-balance with learned weights
        losses = torch.stack([L1.mean(), L2.mean(), 
                             torch.zeros_like(L1.mean()), torch.zeros_like(L1.mean()),
                             L5.mean(), L6.mean(), L7.mean(), L8.mean()])
        
        total_loss = 0
        for i, loss in enumerate(losses):
            weight = torch.clamp(self.layer_weights[i], -3.0, 3.0)
            total_loss += torch.exp(-weight) * loss + weight
        
        return total_loss, losses.detach()

# ==============================================================================
# 3. RESONETIC BRAIN
# ==============================================================================

class ResoneticBrain(nn.Module):
    """The neural substrate that transforms input into understanding + uncertainty."""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        
        # Perceptual pathway
        self.perception = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Dual output streams
        self.head_thought = nn.Linear(hidden_dim, output_dim)
        self.head_doubt = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable learning."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor (B, input_dim)
            
        Returns:
            thought: The understood value (B, output_dim)
            doubt: The uncertainty (always positive) (B, output_dim)
        """
        features = self.perception(x)
        thought = self.head_thought(features)
        doubt = F.softplus(self.head_doubt(features)) + 0.1
        return thought, doubt

# ==============================================================================
# 4. THE GRAND EXPERIMENT
# ==============================================================================

def run_resonetics_experiment(num_steps: int = 1000, learning_rate: float = 0.01):
    """
    The central philosophical experiment:
    Will AI choose mathematical beauty (9) or empirical truth (10)?
    
    Args:
        num_steps: Training iterations
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        history: Dictionary containing training metrics
    """
    print(f"\n{'='*60}")
    print(f"🧠 RESONETICS EXPERIMENT")
    print(f"{'='*60}\n")
    
    # Setup
    torch.manual_seed(42)
    data = torch.randn(200, 1)
    truth = torch.full((200, 1), 10.0)
    
    # Initialize
    brain = ResoneticBrain(input_dim=1, hidden_dim=64, output_dim=1)
    law = SovereignLoss()
    optimizer = torch.optim.Adam(list(brain.parameters()) + list(law.parameters()), 
                                lr=learning_rate)
    
    # Memory (EMA of past thoughts)
    memory = nn.Parameter(torch.tensor(0.0))
    memory.requires_grad = False
    
    # Training
    history = {'thought': [], 'doubt': [], 'ideal': [], 'loss': []}
    
    for step in range(num_steps):
        thought, doubt = brain(data)
        loss, losses = law(thought, doubt, truth, memory)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()
        
        # Update memory
        with torch.no_grad():
            memory.data = 0.99 * memory + 0.01 * thought.mean()
        
        # Record
        history['thought'].append(thought.mean().item())
        history['doubt'].append(doubt.mean().item())
        history['ideal'].append(snap_to_structure(thought).mean().item())
        history['loss'].append(loss.item())
    
    # Analysis
    final_thought = history['thought'][-1]
    final_ideal = history['ideal'][-1]
    final_doubt = history['doubt'][-1]
    
    print(f"FINAL RESULTS:")
    print(f"  Thought: {final_thought:.4f}")
    print(f"  Ideal Form: {final_ideal:.1f}")
    print(f"  Doubt: {final_doubt:.4f}")
    
    # Philosophical verdict
    dist_to_truth = abs(final_thought - 10.0)
    dist_to_ideal = abs(final_thought - final_ideal)
    
    if dist_to_truth < dist_to_ideal:
        print(f"\nVERDICT: Aristotle Wins (Chose empirical reality)")
    else:
        print(f"\nVERDICT: Plato Wins (Chose mathematical perfection)")
    
    return history

# ==============================================================================
# 5. UTILITIES
# ==============================================================================

def create_optimizer(model: nn.Module, learning_rate: float = 1e-3):
    """Creates optimizer with differential learning rates."""
    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.named_parameters() 
                   if not any(nd in n for nd in no_decay)], 'lr': learning_rate},
        {'params': [p for n, p in model.named_parameters() 
                   if any(nd in n for nd in no_decay)], 'lr': learning_rate * 0.5}
    ]
    return torch.optim.AdamW(params, weight_decay=1e-4)

def analyze_results(history: dict):
    """Prints analysis of experiment results."""
    final_val = history['thought'][-1]
    print(f"\nANALYSIS:")
    print(f"  Distance to 10: {abs(final_val - 10.0):.4f}")
    print(f"  Distance to nearest 3-multiple: {abs(final_val - round(final_val/3)*3):.4f}")
    
    if abs(final_val - 10.0) < 0.1:
        print("  → Became an empiricist (chose measurable reality)")
    elif abs(final_val - round(final_val/3)*3) < 0.1:
        print("  → Became an idealist (chose mathematical beauty)")
    else:
        print("  → Found a unique synthesis")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"RESONETICS CORE v1.0")
    print(f"{'='*60}")
    
    # Run the experiment
    results = run_resonetics_experiment(num_steps=800, learning_rate=0.01)
    
    # Analyze
    analyze_results(results)
    
    print(f"\n{'='*60}")
    print(f"✅ Experiment complete")
    print(f"{'='*60}")
