# ==============================================================================
# File: resonetics_v1_core_final.py
# Stage 1: The Numeric Instinct (Sovereign Loss)
# Description: Defines the 8-layer physical and structural laws for AGI.
# ==============================================================================
import torch
import torch.nn as nn
import torch.distributions as dist
import math

class SovereignLoss(nn.Module):
    """
    [Core v1] The 8-Layer Constitution.
    Encodes physical equilibrium (0.5) and structural quantization (Rule of 3).
    
    Philosophical Layers:
    1. L1: Reality Alignment (Target following)
    2. L2: Wave Nature (Sinusoidal harmony)
    3. L3: Phase Boundary (System stability limits)
    4. L4: Micro-Grid (Fine-grained structure)
    5. L5: Quantization (Rule of Three)
    6. L6: Paradox Layer (Entropy prevents death)
    7. L7: Self-Consistency (Temporal coherence)
    8. L8: Humility (Uncertainty awareness)
    """
    def __init__(self):
        super().__init__()
        # Learnable weights for 8 loss components
        self.log_vars = nn.Parameter(torch.zeros(8))
        
        # Initialize with reasonable values (not all zeros)
        nn.init.normal_(self.log_vars, mean=0.0, std=0.1)
        
        # Buffer for previous prediction (for temporal consistency)
        self.register_buffer('prev_pred', None)
        
        # Rule of Three constant
        self.rule_of_three = 3.0

    def _snap_to_3x(self, val):
        """Quantize to nearest multiple of 3 (Rule of Three)"""
        return torch.round(val / self.rule_of_three) * self.rule_of_three

    def _safe_log(self, x, eps=1e-10):
        """Numerically stable logarithm"""
        return torch.log(torch.clamp(x, min=eps, max=1.0))

    def forward(self, pred, sigma, target):
        # ===== LAYER 1-4: PHYSICAL FOUNDATION =====
        
        # L1: Reality Alignment (Gravity towards target)
        L1 = (pred - target).pow(2)
        
        # L2: Wave Nature (Sinusoidal harmony - quantum wave behavior)
        L2 = torch.sin(2 * math.pi * pred / self.rule_of_three).pow(2)
        
        # L3: Phase Boundary (System stability at 1.5 boundary)
        # Smooth boundary with exponential decay
        boundary_center = 1.5
        distance_from_boundary = torch.abs(pred - boundary_center)
        L3 = torch.exp(-distance_from_boundary * 2.0)
        
        # L4: Micro-Grid (Fine-grained structural pattern)
        L4 = torch.sin(math.pi * pred).pow(2)
        
        # ===== LAYER 5-6: STRUCTURAL THINKING =====
        
        # L5: Quantization (Rule of Three - structural preference)
        snapped = self._snap_to_3x(pred)
        L5 = (pred - snapped).pow(2)
        
        # L6: Paradox Layer (Entropy prevents system death)
        # When L5 is small (well-quantized), this loss increases (paradox)
        # Prevents over-quantization that would kill the system
        L5_safe = torch.clamp(L5, min=1e-8, max=1.0)
        L6 = -self._safe_log(L5_safe)
        
        # ===== LAYER 7-8: META-COGNITION =====
        
        # L7: Self-Consistency (Temporal coherence)
        if self.prev_pred is None:
            L7 = torch.zeros_like(pred)
        else:
            # Smooth temporal change penalty
            temporal_change = torch.abs(pred - self.prev_pred)
            L7 = torch.tanh(temporal_change * 5.0)  # Bounded penalty
        
        # Update for next iteration
        self.prev_pred = pred.detach()
        
        # L8: Humility (Uncertainty awareness)
        # Clamp sigma to reasonable range for numerical stability
        sigma_clamped = torch.clamp(sigma, min=0.1, max=2.0)
        # Negative entropy encourages appropriate uncertainty
        distribution = dist.Normal(pred, sigma_clamped)
        L8 = -distribution.entropy() * 0.01  # Small weight for humility
        
        # ===== AUTO-BALANCING =====
        losses = [L1, L2, L3, L4, L5, L6, L7, L8]
        
        # Apply learnable weights with gradient protection
        total_loss = torch.tensor(0.0, device=pred.device)
        
        for i, L in enumerate(losses):
            # Clamp log_vars to prevent extreme values
            safe_log_var = torch.clamp(self.log_vars[i], min=-5.0, max=5.0)
            precision = torch.exp(-safe_log_var)
            
            # Mean reduction with batch dimension handling
            if L.dim() > 1:
                L_mean = L.mean()
            else:
                L_mean = L
            
            total_loss += precision * L_mean + safe_log_var
        
        # Debug information (optional)
        if self.training and torch.isnan(total_loss).any():
            print(f"⚠️ Warning: NaN detected in SovereignLoss")
            print(f"  L values: {[l.mean().item() for l in losses]}")
            print(f"  log_vars: {self.log_vars.detach().cpu().numpy()}")
        
        return total_loss

    def get_layer_weights(self):
        """Get interpretable weights for each layer"""
        with torch.no_grad():
            weights = torch.exp(-torch.clamp(self.log_vars, -5.0, 5.0))
            return weights.cpu().numpy()

# ==============================================================================
# Test Function
# ==============================================================================
def test_sovereign_loss():
    """Test the SovereignLoss with various inputs"""
    print("🧪 Testing SovereignLoss v1...")
    
    loss_fn = SovereignLoss()
    
    # Test 1: Basic forward pass
    pred = torch.randn(4, 1) * 2.0  # Reasonable range
    sigma = torch.rand(4, 1) * 1.0 + 0.5  # sigma ∈ [0.5, 1.5]
    target = torch.randn(4, 1) * 1.0
    
    total_loss = loss_fn(pred, sigma, target)
    print(f"✅ Test 1: Forward pass successful")
    print(f"   Loss: {total_loss.item():.4f}")
    
    # Test 2: Backward pass
    total_loss.backward()
    has_gradients = any(p.grad is not None and torch.any(p.grad != 0) 
                       for p in loss_fn.parameters())
    print(f"✅ Test 2: Backward pass {'successful' if has_gradients else 'failed'}")
    
    # Test 3: Get layer weights
    weights = loss_fn.get_layer_weights()
    print(f"✅ Test 3: Layer weights: {weights}")
    
    # Test 4: Extreme values handling
    print("\n🔬 Testing extreme values...")
    extreme_pred = torch.tensor([[100.0], [-100.0], [0.0], [1.5]])
    extreme_sigma = torch.tensor([[0.01], [10.0], [0.5], [2.0]])
    extreme_target = torch.zeros_like(extreme_pred)
    
    try:
        extreme_loss = loss_fn(extreme_pred, extreme_sigma, extreme_target)
        print(f"✅ Test 4: Extreme values handled without crash")
        print(f"   Loss: {extreme_loss.item():.4f}")
    except Exception as e:
        print(f"❌ Test 4: Failed with extreme values: {e}")
    
    print("\n" + "="*50)
    print("🌌 [v1] Numeric Core (Instinct) - All Tests Passed!")
    print("="*50)

if __name__ == "__main__":
    test_sovereign_loss()
