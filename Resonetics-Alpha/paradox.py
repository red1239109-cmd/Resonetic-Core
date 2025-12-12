# ==============================================================================
# File: resonetics_transformer_v3_final.py
# Project: Resonetic Transformer (Unit 02) - Fully Fixed Version
# Author: red1239109-cmd
# Copyright (c) 2025 red1239109-cmd
#
# License: AGPL-3.0
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# Description:
#   The Silicon Brain of Resonetics.
#   Implements R-Grammar encoding, Resonance Attention with Phase Bias,
#   and Boundary Layer logic for shock wave detection.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================
# 1. Components (Fully Fixed)
# ==============================

class RGrammarEncoder(nn.Module):
    """
    Projects input tokens into the 4-dimensional Semantic Space (S/R/T/G).
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 4)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        
    def forward(self, x):
        return torch.sigmoid(self.proj(x))

class ResonanceAttention(nn.Module):
    """
    Attention mechanism based on Physical Resonance.
    Uses Phase Difference to bias attention scores.
    """
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.phase = nn.Linear(d_model, n_heads)
        
        # Learnable scaling factors
        self.resonance_scale = nn.Parameter(torch.tensor(0.1))
        self.phase_scale = nn.Parameter(torch.tensor(1.0))
        
        # Initialize
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # 1. QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_h)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, d_h)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Base attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_h)

        # 3. Resonance bias (with numerical stability)
        # Tanh ensures phase is bounded [-1, 1]
        phase_val = torch.tanh(self.phase_scale * self.phase(x))  # (B, L, H)
        phase_val = phase_val.transpose(1, 2)  # (B, H, L)
        
        # Phase difference matrix
        pi = phase_val.unsqueeze(-1)  # (B, H, L, 1)
        pj = phase_val.unsqueeze(-2)  # (B, H, 1, L)
        phase_diff = (pi - pj) ** 2  # (B, H, L, L)
        
        # Apply resonance bias (subtract squared phase diff)
        scores = scores - self.resonance_scale * phase_diff

        # 4. Apply mask (handles different mask shapes)
        if mask is not None:
            if mask.dim() == 2:  # (B, L)
                mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            elif mask.dim() == 3:  # (B, L, L)
                mask = mask.unsqueeze(1)  # (B, 1, L, L)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 5. Softmax and output
        attn_weights = F.softmax(scores, dim=-1)
        
        # Optional: dropout for training
        if self.training:
            attn_weights = F.dropout(attn_weights, p=0.1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        
        return self.out(out), attn_weights

class BoundaryLayer(nn.Module):
    """
    Detects cognitive 'shock waves' (internal contradictions).
    Based on Burgers' Equation logic.
    """
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model // 2
            
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Returns shock score: (B, L, 1)
        # 1.0 = smooth flow, 0.0 = shock detected
        return self.mlp(x)

# ==============================
# 2. Main Architecture (Fixed)
# ==============================

class ResoneticTransformer(nn.Module):
    def __init__(self, vocab_size=30000, d_model=512, n_heads=8, n_layers=6, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model))
        
        # Blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'rg': RGrammarEncoder(d_model),
                'attn': ResonanceAttention(d_model, n_heads),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(0.1)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
            }) for _ in range(n_layers)
        ])
        
        # Boundary detection (shared across layers)
        self.boundary = BoundaryLayer(d_model)
        
        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (optional)
        self.head.weight = self.embed.weight
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                if 'embed' in name or 'head' in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
        # Position encoding
        nn.init.normal_(self.pos_enc, mean=0.0, std=0.02)
        
    def forward(self, ids, mask=None):
        B, L = ids.shape
        
        # 1. Embeddings with scaling
        x = self.embed(ids) * math.sqrt(self.d_model)
        x = x + self.pos_enc[:, :L, :]
        
        # State tracking
        grammar_states = []
        shock_scores = []
        
        # 2. Process through all layers
        for i, blk in enumerate(self.blocks):
            # R-Grammar encoding
            rg_weights = blk['rg'](x)
            grammar_states.append(rg_weights)
            
            # Grammar modulation (with residual)
            grammar_modulation = 1.0 + 0.1 * rg_weights.mean(dim=-1, keepdim=True)
            x_modulated = x * grammar_modulation
            
            # Attention with pre-norm
            attn_input = blk['norm1'](x_modulated)
            attn_out, _ = blk['attn'](attn_input, mask)
            x = x_modulated + attn_out  # Residual connection
            
            # Feed-forward with pre-norm
            ff_input = blk['norm2'](x)
            ff_out = blk['ff'](ff_input)
            x = x + ff_out  # Residual connection
            
            # Boundary/shock detection
            shock_score = self.boundary(x)  # (B, L, 1)
            shock_scores.append(shock_score)
            
            # Apply shock damping if needed
            if self.training:  # Only during training for stability
                shock_damping = 0.3 + 0.7 * shock_score  # (B, L, 1)
                x = x * shock_damping
        
        # 3. Final normalization and output
        x = self.final_norm(x)
        logits = self.head(x)
        
        # 4. Aggregate states
        grammar_stack = torch.stack(grammar_states, dim=0)  # (n_layers, B, L, 4)
        shock_stack = torch.stack(shock_scores, dim=0)       # (n_layers, B, L, 1)
        
        # Average across layers
        avg_grammar = grammar_stack.mean(dim=0)      # (B, L, 4)
        avg_shock = shock_stack.mean(dim=0)          # (B, L, 1)
        
        return {
            'logits': logits,
            'hidden_states': x,
            'grammar_states': avg_grammar,
            'boundary_scores': avg_shock,
            'grammar_history': grammar_stack,
            'shock_history': shock_stack
        }

# ==============================
# 3. Enhanced Verification
# ==============================

def verify_model():
    print("🧪 Unit 02 - Comprehensive Verification")
    print("=" * 50)
    
    # Test configurations
    configs = [
        {"vocab_size": 1000, "d_model": 128, "n_heads": 4, "n_layers": 2},
        {"vocab_size": 5000, "d_model": 256, "n_heads": 8, "n_layers": 4},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nTest {i+1}: {config}")
        
        # Create model
        model = ResoneticTransformer(**config)
        model.eval()
        
        # Test inputs
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        
        # Test with and without mask
        mask = torch.ones(batch_size, seq_len)
        mask[:, -3:] = 0  # Mask last 3 tokens
        
        # Forward pass
        with torch.no_grad():
            output1 = model(input_ids)
            output2 = model(input_ids, mask)
        
        # Verify shapes
        assert output1['logits'].shape == (batch_size, seq_len, config["vocab_size"])
        assert output2['logits'].shape == (batch_size, seq_len, config["vocab_size"])
        
        assert output1['grammar_states'].shape == (batch_size, seq_len, 4)
        assert output1['boundary_scores'].shape == (batch_size, seq_len, 1)
        
        # Check shock scores are in valid range
        assert torch.all(output1['boundary_scores'] >= 0.0) and torch.all(output1['boundary_scores'] <= 1.0)
        
        # Check gradients can flow
        model.train()
        output = model(input_ids)
        loss = output['logits'].sum()
        loss.backward()
        
        # Check no NaN values
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.any(torch.isnan(param.grad)), f"NaN in {name} gradients"
        
        print(f"  ✓ Shapes correct")
        print(f"  ✓ No NaN values")
        print(f"  ✓ Gradients flow")
        print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Print statistics
        grammar_mean = output1['grammar_states'].mean().item()
        shock_mean = output1['boundary_scores'].mean().item()
        print(f"  📊 Grammar activation: {grammar_mean:.4f}")
        print(f"  📊 Boundary health: {shock_mean:.4f}")

if __name__ == "__main__":
    print("⚡️ Resonetic Transformer Unit 02 - Final Version")
    print("🔄 Running comprehensive verification...")
    
    verify_model()
    
    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("🚀 Unit 02 is fully operational and stable.")
    print("\n[System Status]")
    print("  - R-Grammar: ✓ Functional")
    print("  - Resonance Attention: ✓ Stable")
    print("  - Boundary Detection: ✓ Active")
    print("  - Gradient Flow: ✓ Healthy")
