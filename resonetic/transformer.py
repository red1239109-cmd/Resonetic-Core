# Copyright (c) 2025 red1239109-cmd
# Licensed under AGPL-3.0. See LICENSE file for details.

"""
Resonetic Transformer (Unit 02)
-------------------------------
"My thinking process, now in silicon."

Unlike traditional Transformers that rely solely on statistical correlations,
this architecture incorporates structural constraints (R-Grammar) and
physical principles (Resonance/Burgers' Eq) directly into its neural pathways.

[Key Features]
1. R-Grammar Encoder: Projects input into S/R/T/G 4-dimensional semantic space.
2. Resonance Attention: Introduces 'Phase Bias' to model truthfulness as resonance.
3. Boundary Layer: Detects cognitive 'shock waves' (discontinuities) using Burgers' logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RGrammarEncoder(nn.Module):
    """
    Encodes tokens into the 4-layer R-Grammar space.
    Output: [Surface, Structural, Topological, Generative] weights
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 4) # S / R / T / G

    def forward(self, x):
        # Sigmoid ensures the grammar weights are between 0 and 1
        return torch.sigmoid(self.proj(x))

class ResonanceAttention(nn.Module):
    """
    The Core Innovation: Attention mechanism with Physical Resonance.
    
    Standard Attention: Dot(Q, K)
    Resonance Attention: Dot(Q, K) - Phase_Difference^2
    
    If phases are out of sync (high difference), attention resonates less (logic breaks).
    """
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        
        # Learnable Phase parameter for each token head
        self.phase = nn.Linear(d_model, n_heads)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # 1. Standard Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Base Attention Score
        attn = (q @ k.transpose(-2, -1)) / (self.d_h ** 0.5)

        # 3. Apply Resonance Bias (Phase Difference)
        # Each token has a 'Phase' (phi). We penalize high phase difference.
        phase_val = self.phase(x).transpose(1, 2)  # (B, H, L)
        
        # Broadcasting to calculate (phi_i - phi_j)^2
        pi = phase_val.unsqueeze(-1) # (B, H, L, 1)
        pj = phase_val.unsqueeze(-2) # (B, H, 1, L)
        
        # Resonance logic: closer phase = higher attention
        resonance_bias = -(pi - pj) ** 2
        attn = attn + resonance_bias

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        
        # 4. Output
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out), attn

class BoundaryLayer(nn.Module):
    """
    Inspired by Burgers' Equation Shock Waves.
    Acts as a gatekeeper to detect sudden discontinuities (trauma/lies) in the thought process.
    """
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(), # Tanh often models wave-like behaviors better
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # Returns a gating value (0~1). 
        # If shock is detected (low value), it suppresses the signal.
        return torch.sigmoid(self.mlp(x).squeeze(-1))

class ResoneticTransformer(nn.Module):
    """
    The Full Architecture: "Thinking in Silicon"
    """
    def __init__(self, vocab_size=30000, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 1024, d_model)) # Simple learnable position encoding
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'rg': RGrammarEncoder(d_model),
                'attn': ResonanceAttention(d_model, n_heads),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model*4), 
                    nn.GELU(), 
                    nn.Linear(d_model*4, d_model)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
            }) for _ in range(n_layers)
        ])
        
        self.boundary = BoundaryLayer(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids, mask=None):
        B, L = ids.shape
        
        # 1. Embedding + Position
        x = self.embed(ids) + self.pos_enc[:, :L, :]
        
        # 2. Resonetic Layers
        for blk in self.blocks:
            # R-Grammar constraint injection
            rg_weights = blk['rg'](x) 
            
            # Inject grammar 'thought' into the stream (simple modulation)
            # This forces the model to 'think' about structure before attention
            x_modulated = x * (1 + rg_weights.mean(-1, keepdim=True))
            
            # Resonance Attention
            attn_out, _ = blk['attn'](self.norm1(x_modulated))
            x = blk['norm1'](x + attn_out)
            
            # Feed Forward
            x = blk['norm2'](x + blk['ff'](x))
            
        # 3. Boundary Check (Shock Wave Filter)
        # Before speaking, check for internal contradictions (shocks)
        boundary_gate = self.boundary(x)
        
        # Apply gate: If boundary fails (shock), output is suppressed/modified
        final_thought = x * boundary_gate.unsqueeze(-1)
        
        # 4. Synthesis (Logits)
        logits = self.head(final_thought)
        
        return {
            'logits': logits,
            'boundary_state': boundary_gate,
            'grammar_state': rg_weights
        }

# ==============================
# Demo Execution
# ==============================
if __name__ == "__main__":
    print("⚡️ Initializing Unit 02: Resonetic Transformer...")
    
    # 1. Initialize Model
    model = ResoneticTransformer(vocab_size=1000, d_model=128, n_heads=4, n_layers=2)
    print("✅ Model Architecture Built in Silicon.")
    
    # 2. Dummy Input (Batch=1, Length=10)
    dummy_input = torch.randint(0, 1000, (1, 10))
    
    # 3. Forward Pass (Thinking Process)
    print("🧠 Simulating Thought Process...")
    output = model(dummy_input)
    
    print("\n--- [Internal State Analysis] ---")
    print(f"Logits Shape: {output['logits'].shape} (Vocabulary Projection)")
    print(f"Boundary Gate (Shock Filter): {output['boundary_state'].mean().item():.4f} (1.0 = Flow, 0.0 = Shock)")
    print(f"R-Grammar Activation: {output['grammar_state'].mean().item():.4f} (Structural Awareness)")
    
    print("\n🚀 Unit 02 is operational.")
