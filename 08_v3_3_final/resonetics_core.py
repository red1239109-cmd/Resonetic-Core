# ==============================================================================
# File: resonetics_core.py
# Project: Resonetics Core Engine
# Version: 1.0 (Fundamental Implementation)
# Author: red1239109-cmd
# Copyright (c) 2025 Resonetics Project
#
# License: AGPL-3.0
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================

"""
RESONETICS CORE
===============
The fundamental implementation of Resonetics - a meta-cognitive AI system
that learns how to learn through self-awareness of its own limitations.

Philosophical Foundations:
1. Gödelian Incompleteness: Self-awareness of cognitive boundaries
2. Aristotelian Mean: Balanced adaptation between extremes
3. Stoic Resilience: Graceful degradation under uncertainty
4. Heraclitean Flow: Embracing constant change as fundamental
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class ResoneticsConfig:
    """Philosophical configuration for Resonetics Core"""
    
    # Meta-Cognition Parameters
    base_learning_rate: float = 0.01
    risk_threshold_panic: float = 0.5      # Gödelian boundary of breakdown
    risk_threshold_alert: float = 0.2      # Aristotelian warning zone
    risk_threshold_warning: float = 0.1    # Stoic early warning
    
    # Learning Rate Multipliers (Hegelian adaptation)
    lr_multiplier_panic: float = 5.0       # Drastic measures for breakdown
    lr_multiplier_alert: float = 2.0       # Moderate correction
    lr_multiplier_warning: float = 1.2     # Gentle guidance
    lr_multiplier_cruise: float = 0.5      # Calm refinement
    
    # System Parameters
    error_buffer_size: int = 10            # Memory of past experiences
    risk_smoothing_factor: float = 0.9     # Exponential moving average
    gradient_clip_norm: float = 1.0        # Protection against divergence
    
    # Uncertainty Bounds (Socratic humility)
    min_uncertainty: float = 0.1           # Protection against overconfidence
    max_uncertainty: float = 5.0           # Protection against evasion

# ==============================================================================
# Core Neural Components
# ==============================================================================

class RiskPredictor(nn.Module):
    """
    Meta-cognitive risk assessment module.
    Predicts instability based on current state and recent performance.
    
    Philosophical Role: The system's 'self-awareness' - knowing when it doesn't know.
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Risk ∈ [0, 1]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for stable gradient flow"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor, recent_error: torch.Tensor) -> torch.Tensor:
        """
        Predict risk of instability.
        
        Args:
            state: Current system state [batch_size, 2]
            recent_error: Recent average error [batch_size, 1]
            
        Returns:
            Predicted risk score ∈ [0, 1], where 0=stable, 1=unstable
        """
        # Ensure proper dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if recent_error.dim() == 1:
            recent_error = recent_error.unsqueeze(1)
        
        # Combine features
        features = torch.cat([state, recent_error], dim=1)
        
        # Validate input dimension
        if features.size(1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {features.size(1)}")
        
        return self.net(features)

class WorkerNetwork(nn.Module):
    """
    Primary task performer - learns to solve the immediate problem.
    
    Philosophical Role: The 'hands' of the system - executing practical work.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Smooth, bounded activation
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Controlled initialization for stable learning"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)  # Conservative gain
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dimension safety"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

# ==============================================================================
# Meta-Cognitive Optimizer
# ==============================================================================

class ProphetOptimizer:
    """
    Self-tuning optimizer with risk-aware learning rate adjustment.
    
    Philosophical Role: The 'wisdom' - knowing when to push hard and when to step back.
    """
    
    def __init__(self, base_optimizer: torch.optim.Optimizer, config: ResoneticsConfig):
        self.optimizer = base_optimizer
        self.config = config
        self.current_lr = config.base_learning_rate
        
        # State tracking
        self.risk_history = []
        self.lr_history = []
        self.mode_history = []
        
        # Exponential moving average of risk
        self.smoothed_risk = 0.0
    
    def adjust(self, risk_score: torch.Tensor) -> Tuple[float, str, int]:
        """
        Adjust learning rate based on predicted risk.
        
        Args:
            risk_score: Predicted instability ∈ [0, 1]
            
        Returns:
            current_lr: Adjusted learning rate
            mode: Descriptive string of current mode
            status_code: Numeric status for monitoring
        """
        risk = risk_score.item()
        
        # Update smoothed risk (EMA for stability)
        self.smoothed_risk = (self.config.risk_smoothing_factor * self.smoothed_risk + 
                             (1 - self.config.risk_smoothing_factor) * risk)
        
        # Determine adjustment mode based on risk thresholds
        if risk > self.config.risk_threshold_panic:
            target_lr = self.config.base_learning_rate * self.config.lr_multiplier_panic
            mode = "🚨 PANIC"
            status = 0  # System in breakdown prevention
            
        elif risk > self.config.risk_threshold_alert:
            target_lr = self.config.base_learning_rate * self.config.lr_multiplier_alert
            mode = "⚠️ ALERT"
            status = 1  # System under stress
            
        elif risk > self.config.risk_threshold_warning:
            target_lr = self.config.base_learning_rate * self.config.lr_multiplier_warning
            mode = "🔶 WARNING"
            status = 2  # Minor instability detected
            
        else:
            target_lr = self.config.base_learning_rate * self.config.lr_multiplier_cruise
            mode = "✅ CRUISE"
            status = 3  # Optimal learning zone
        
        # Smooth transition to target learning rate
        self.current_lr = 0.8 * self.current_lr + 0.2 * target_lr
        
        # Apply safety bounds
        self.current_lr = max(1e-6, min(self.current_lr, 1.0))
        
        # Apply to all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        
        # Track history for analysis
        self.risk_history.append(risk)
        self.lr_history.append(self.current_lr)
        self.mode_history.append(mode)
        
        # Keep history bounded
        if len(self.risk_history) > 100:
            self.risk_history.pop(0)
            self.lr_history.pop(0)
            self.mode_history.pop(0)
        
        return self.current_lr, mode, status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics for monitoring and analysis"""
        if not self.risk_history:
            return {}
        
        return {
            'current_learning_rate': self.current_lr,
            'smoothed_risk': self.smoothed_risk,
            'recent_risk_mean': np.mean(self.risk_history[-10:]) if len(self.risk_history) >= 10 else np.mean(self.risk_history),
            'recent_risk_std': np.std(self.risk_history[-10:]) if len(self.risk_history) >= 10 else np.std(self.risk_history),
            'current_mode': self.mode_history[-1] if self.mode_history else "UNKNOWN",
            'mode_distribution': {
                mode: self.mode_history.count(mode) / len(self.mode_history)
                for mode in set(self.mode_history)
            }
        }

# ==============================================================================
# Dynamic Target Generator
# ==============================================================================

def generate_dynamic_target(step: int, total_steps: int) -> float:
    """
    Generate dynamic target with simulated concept drift.
    
    Philosophical Role: Simulates Heraclitean flow - constant change in reality.
    
    Args:
        step: Current training step
        total_steps: Total training steps
        
    Returns:
        Target value for current step
    """
    progress = step / total_steps
    
    # Phase 1: Smooth learning (0-25%)
    if progress < 0.25:
        t = step * 0.1
        return float(np.sin(t))  # Gentle oscillations
    
    # Phase 2: Sudden changes - concept drift (25-50%)
    elif progress < 0.5:
        t = step * 0.05
        return 1.0 if (step // 20) % 2 == 0 else -1.0  # Binary shifts
    
    # Phase 3: Complex patterns (50-75%)
    elif progress < 0.75:
        t = step * 0.02
        return float(np.sin(t) * np.cos(t * 0.3) * 0.5 + np.sin(t * 0.1) * 0.5)  # Interference patterns
    
    # Phase 4: Slow drift with noise (75-100%)
    else:
        t = step * 0.01
        noise = np.random.normal(0, 0.05) if step % 3 == 0 else 0.0
        return float(np.sin(t) * 0.3 + np.cos(t * 0.5) * 0.2 + noise)  # Gradual evolution

# ==============================================================================
# Main Resonetics System
# ==============================================================================

class ResoneticsCore:
    """
    Complete Resonetics meta-cognitive AI system.
    
    Philosophical Architecture:
    1. WorkerNetwork: Practical intelligence (Aristotelian praxis)
    2. RiskPredictor: Self-awareness (Socratic introspection)
    3. ProphetOptimizer: Adaptive wisdom (Stoic proportionality)
    """
    
    def __init__(self, config: Optional[ResoneticsConfig] = None):
        """
        Initialize the Resonetics system.
        
        Args:
            config: Resonetics configuration (uses default if None)
        """
        self.config = config or ResoneticsConfig()
        
        # Initialize on available device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize neural components
        self._initialize_models()
        
        # State tracking
        self.error_buffer = []  # Recent errors for risk prediction
        self.recent_error = torch.tensor([[0.0]], device=self.device)
        self.current_step = 0
        self.best_error = float('inf')
        
        # Training history
        self.training_history = {
            'errors': [],
            'risks': [],
            'learning_rates': [],
            'modes': []
        }
        
        print("=" * 60)
        print("🧠 RESONETICS CORE INITIALIZED")
        print(f"Device: {self.device}")
        print(f"Base Learning Rate: {self.config.base_learning_rate}")
        print("=" * 60)
    
    def _initialize_models(self):
        """Initialize all neural components with proper configuration"""
        # Worker network (main task performer)
        self.worker = WorkerNetwork().to(self.device)
        self.worker_optimizer = torch.optim.Adam(
            self.worker.parameters(),
            lr=self.config.base_learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        
        # Risk predictor (meta-cognitive component)
        self.predictor = RiskPredictor().to(self.device)
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=self.config.base_learning_rate * 0.5,  # Slower learning for meta-cognition
            weight_decay=1e-4
        )
        
        # Prophet optimizer wrapper
        self.prophet_tuner = ProphetOptimizer(self.worker_optimizer, self.config)
    
    def train_step(self) -> Tuple[float, float, str]:
        """
        Execute one meta-cognitive training step.
        
        Returns:
            error: Current prediction error
            risk: Predicted instability risk
            mode: Current learning mode
        """
        # Generate dynamic target (simulating changing reality)
        target_value = generate_dynamic_target(self.current_step, 1000)
        target = torch.tensor([[target_value]], device=self.device, dtype=torch.float32)
        
        # Prepare input: [progress, recent_error]
        progress = self.current_step / 1000
        worker_input = torch.tensor([[progress, self.recent_error.item()]], 
                                   device=self.device, dtype=torch.float32)
        
        # ===== META-COGNITIVE CYCLE =====
        
        # 1. Predict risk of instability
        predicted_risk = self.predictor(worker_input[:, :2], self.recent_error)
        
        # 2. Adjust learning rate based on predicted risk
        current_lr, mode, status = self.prophet_tuner.adjust(predicted_risk)
        
        # 3. Execute main task
        action = self.worker(worker_input)
        loss = (action - target).pow(2)
        actual_error = loss.detach()
        
        # 4. Update error buffer for risk prediction
        self.error_buffer.append(actual_error.item())
        if len(self.error_buffer) > self.config.error_buffer_size:
            self.error_buffer.pop(0)
        
        if self.error_buffer:
            self.recent_error = torch.tensor(
                [[sum(self.error_buffer) / len(self.error_buffer)]],
                device=self.device, dtype=torch.float32
            )
        
        # 5. Train worker network (main task)
        self.worker_optimizer.zero_grad(set_to_none=True)  # Memory efficient
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker.parameters(), self.config.gradient_clip_norm)
        self.worker_optimizer.step()
        
        # 6. Train risk predictor (meta-learning)
        target_risk = torch.tanh(actual_error)  # Transform error to risk space
        meta_loss = (predicted_risk - target_risk).pow(2)
        
        self.predictor_optimizer.zero_grad(set_to_none=True)
        meta_loss.backward()
        self.predictor_optimizer.step()
        
        # 7. Update training history
        self.training_history['errors'].append(actual_error.item())
        self.training_history['risks'].append(predicted_risk.item())
        self.training_history['learning_rates'].append(current_lr)
        self.training_history['modes'].append(mode)
        
        # 8. Update best error
        if actual_error.item() < self.best_error:
            self.best_error = actual_error.item()
        
        # 9. Increment step counter
        self.current_step += 1
        
        return actual_error.item(), predicted_risk.item(), mode
    
    def train(self, steps: int = 1000, log_interval: int = 100):
        """
        Complete training session.
        
        Args:
            steps: Total training steps
            log_interval: Logging frequency
        """
        print(f"\n🚀 Starting Resonetics Training ({steps} steps)")
        print("=" * 60)
        
        for step in range(steps):
            error, risk, mode = self.train_step()
            
            # Periodic logging
            if step % log_interval == 0:
                stats = self.prophet_tuner.get_statistics()
                print(f"Step {step:4d}/{steps} | "
                      f"Error: {error:7.5f} | "
                      f"Risk: {risk:5.3f} | "
                      f"LR: {stats.get('current_learning_rate', 0):.5f} | "
                      f"{mode}")
        
        # Final statistics
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE")
        print("=" * 60)
        
        stats = self.prophet_tuner.get_statistics()
        print(f"Best Error: {self.best_error:.6f}")
        print(f"Final Learning Rate: {stats.get('current_learning_rate', 0):.6f}")
        print(f"Average Risk: {np.mean(self.training_history['risks']):.3f}")
        print(f"Mode Distribution: {stats.get('mode_distribution', {})}")
        
        return self.best_error
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for monitoring"""
        prophet_stats = self.prophet_tuner.get_statistics()
        
        return {
            'step': self.current_step,
            'best_error': self.best_error,
            'recent_error': self.recent_error.item(),
            'error_buffer_size': len(self.error_buffer),
            'prophet_statistics': prophet_stats,
            'worker_parameters': sum(p.numel() for p in self.worker.parameters()),
            'predictor_parameters': sum(p.numel() for p in self.predictor.parameters()),
            'device': str(self.device)
        }

# ==============================================================================
# Demonstration
# ==============================================================================

def demonstrate_resonetics():
    """Demonstrate the Resonetics Core system"""
    print("🧪 RESONETICS CORE DEMONSTRATION")
    print("=" * 60)
    
    # Create Resonetics system with custom configuration
    config = ResoneticsConfig(
        base_learning_rate=0.01,
        risk_threshold_panic=0.5,
        risk_threshold_alert=0.2,
        risk_threshold_warning=0.1,
        lr_multiplier_panic=5.0,
        lr_multiplier_alert=2.0,
        lr_multiplier_warning=1.2,
        lr_multiplier_cruise=0.5
    )
    
    # Initialize system
    resonetics = ResoneticsCore(config)
    
    # Display initial status
    print("\n📊 Initial System Status:")
    status = resonetics.get_system_status()
    for key, value in status.items():
        if key != 'prophet_statistics':
            print(f"  {key}: {value}")
    
    # Run training
    print("\n🎯 Starting Meta-Cognitive Training...")
    best_error = resonetics.train(steps=500, log_interval=50)
    
    # Final analysis
    print("\n📈 Training Analysis:")
    print(f"  Total Steps: {resonetics.current_step}")
    print(f"  Best Error Achieved: {best_error:.6f}")
    
    stats = resonetics.prophet_tuner.get_statistics()
    print(f"  Final Learning Mode: {stats.get('current_mode', 'UNKNOWN')}")
    print(f"  Risk Awareness: {stats.get('smoothed_risk', 0):.3f}")
    
    # Mode distribution analysis
    mode_dist = stats.get('mode_distribution', {})
    if mode_dist:
        print("\n🎭 Learning Mode Distribution:")
        for mode, percentage in mode_dist.items():
            print(f"  {mode}: {percentage:.1%}")
    
    print("\n" + "=" * 60)
    print("✨ DEMONSTRATION COMPLETE")
    print("=" * 60)

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demonstration
    demonstrate_resonetics()
