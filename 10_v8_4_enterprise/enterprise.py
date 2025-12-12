# ==============================================================================
# File: resonetics_v8_4_enterprise_fixed.py
# Project: Resonetics (The Prophet) - Enterprise Edition
# Version: 8.4.1 (Fixed Production Release)
# Author: red1239109-cmd
# Copyright (c) 2023-2025 Resonetics Project
#
# ==============================================================================
# DUAL LICENSE MODEL:
#
# 1. OPEN SOURCE LICENSE (AGPL-3.0)
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    REQUIREMENTS:
#    - If you use this software over a network, you MUST open-source your
#      entire project under AGPL-3.0
#    - You must preserve all copyright notices and license headers
#    - You must make source code available to all users
#
# 2. COMMERCIAL LICENSE
#    For organizations that wish to use this software in proprietary products
#    without open-sourcing their code, a commercial license is available.
#
#    Contact: red1239109@gmail.com
#    Pricing: Based on organization size and usage
#
# NOTICE: This software includes Prometheus client libraries under Apache 2.0
#         and Flask under BSD 3-Clause licenses. See respective licenses.
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import os
import sys
import threading
import time
import logging
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
import signal
import atexit
from functools import lru_cache

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# DEPENDENCY IMPORTS WITH GRACEFUL FALLBACKS
# ==============================================================================

# Prometheus metrics (optional for basic operation)
try:
    from prometheus_client import start_http_server, Gauge, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Metrics disabled.")

# Flask for health checks (optional)
try:
    from flask import Flask, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not available. Health endpoints disabled.")

# Production web server (optional)
try:
    from waitress import serve
    WAITRESS_AVAILABLE = True
except ImportError:
    WAITRESS_AVAILABLE = False
    logger.warning("Waitress not available. Using development server.")

# ==============================================================================
# ENTERPRISE MONITORING SYSTEM
# ==============================================================================

class EnterpriseMonitor:
    """
    Production-grade monitoring with graceful degradation.
    Provides metrics and health checks even when dependencies are missing.
    """
    
    def __init__(self, enable_prometheus: bool = True, enable_health: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_health = enable_health and FLASK_AVAILABLE
        
        self.metrics = {}
        self.health_thread = None
        self.metrics_thread = None
        
        if self.enable_prometheus:
            self._setup_prometheus()
        
        if self.enable_health:
            self._setup_health_endpoints()
    
    def _setup_prometheus(self):
        """Initialize Prometheus metrics registry"""
        self.registry = CollectorRegistry()
        self.metrics = {
            'learning_rate': Gauge(
                'prophet_learning_rate',
                'Current auto-tuned learning rate',
                registry=self.registry
            ),
            'predicted_risk': Gauge(
                'prophet_predicted_risk',
                'Predicted instability score (0-1)',
                registry=self.registry
            ),
            'actual_error': Gauge(
                'prophet_actual_error',
                'Actual task error (MSE)',
                registry=self.registry
            ),
            'training_step': Gauge(
                'prophet_training_step_total',
                'Total training steps completed',
                registry=self.registry
            ),
            'system_status': Gauge(
                'prophet_system_status',
                'System status (0=panic, 1=alert, 2=cruise)',
                registry=self.registry
            )
        }
    
    def _setup_health_endpoints(self):
        """Setup Flask health check endpoints"""
        self.health_app = Flask(__name__)
        
        @self.health_app.route("/healthz")
        def healthz():
            """Kubernetes liveness probe"""
            return Response("OK\n", status=200, mimetype='text/plain')
        
        @self.health_app.route("/readyz")
        def readyz():
            """Kubernetes readiness probe"""
            return Response("READY\n", status=200, mimetype='text/plain')
        
        @self.health_app.route("/metrics")
        def metrics():
            """Prometheus metrics endpoint"""
            if self.enable_prometheus:
                from prometheus_client import generate_latest
                return Response(generate_latest(self.registry), 
                              mimetype='text/plain')
            return Response("# Metrics disabled\n", status=501)
    
    def start_services(self, metrics_port: int = 8000, health_port: int = 8080):
        """Start monitoring services in background threads"""
        
        # Start Prometheus metrics server
        if self.enable_prometheus:
            def start_prometheus():
                try:
                    start_http_server(metrics_port, registry=self.registry)
                    logger.info(f"Prometheus endpoint: http://localhost:{metrics_port}/metrics")
                except Exception as e:
                    logger.error(f"Failed to start Prometheus: {e}")
            
            self.metrics_thread = threading.Thread(target=start_prometheus, daemon=True)
            self.metrics_thread.start()
        else:
            logger.info("Prometheus metrics disabled")
        
        # Start health check server
        if self.enable_health:
            def start_health_server():
                try:
                    if WAITRESS_AVAILABLE:
                        # Production server
                        serve(self.health_app, 
                              host='0.0.0.0', 
                              port=health_port,
                              threads=4,
                              ident="Resonetics Health")
                    else:
                        # Development server (not for production)
                        self.health_app.run(host='0.0.0.0', 
                                           port=health_port,
                                           debug=False,
                                           use_reloader=False)
                except Exception as e:
                    logger.error(f"Health server failed: {e}")
            
            self.health_thread = threading.Thread(target=start_health_server, daemon=True)
            self.health_thread.start()
            logger.info(f"Health endpoints: http://localhost:{health_port}/{{healthz,readyz,metrics}}")
        else:
            logger.info("Health endpoints disabled")
    
    def update_metrics(self, lr: float, risk: float, error: float, step: int, status: int):
        """Update all metrics atomically"""
        if self.enable_prometheus:
            self.metrics['learning_rate'].set(lr)
            self.metrics['predicted_risk'].set(risk)
            self.metrics['actual_error'].set(error)
            self.metrics['training_step'].set(step)
            self.metrics['system_status'].set(status)

# ==============================================================================
# CONFIGURATION MANAGEMENT
# ==============================================================================

class ConfigManager:
    """Safe configuration loading with validation and defaults"""
    
    DEFAULT_CONFIG = {
        'system': {
            'seed': 42,
            'steps': 2000,
            'log_interval': 100,
            'enable_monitoring': True,
            'save_checkpoints': True,
            'checkpoint_interval': 500,
            'batch_size': 1
        },
        'optimizer': {
            'base_lr': 0.01,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 1e-4
        },
        'prophet': {
            'risk_thresholds': {
                'panic': 0.5,
                'alert': 0.2,
                'warning': 0.1
            },
            'lr_multipliers': {
                'panic': 5.0,    # Drastic measures for high risk
                'alert': 2.0,    # Moderate adjustment
                'warning': 1.2,  # Small adjustment
                'cruise': 0.5    # Normal operation
            },
            'buffer_size': 10,
            'risk_smoothing': 0.9
        },
        'monitoring': {
            'metrics_port': 8000,
            'health_port': 8080,
            'enable_prometheus': True,
            'enable_health': True
        }
    }
    
    @staticmethod
    def load(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file or environment variables.
        Priority: Environment > Config File > Defaults
        """
        # Try environment variable first
        if config_path is None:
            config_path = os.getenv('PROPHET_CONFIG', 'config/prophet.yaml')
        
        config = ConfigManager.DEFAULT_CONFIG.copy()
        
        # Load from file if exists
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    ConfigManager._deep_update(config, file_config)
                logger.info(f"Loaded config from: {config_path}")
            else:
                logger.info(f"Config file not found: {config_path}. Using defaults.")
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")
        
        # Override with environment variables
        ConfigManager._apply_env_overrides(config)
        
        return config
    
    @staticmethod
    def _deep_update(base: Dict, update: Dict):
        """Recursively update nested dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigManager._deep_update(base[key], value)
            else:
                base[key] = value
    
    @staticmethod
    def _apply_env_overrides(config: Dict):
        """Apply environment variable overrides"""
        # System settings
        if os.getenv('PROPHET_SEED'):
            config['system']['seed'] = int(os.getenv('PROPHET_SEED'))
        
        # Optimizer settings
        if os.getenv('PROPHET_BASE_LR'):
            config['optimizer']['base_lr'] = float(os.getenv('PROPHET_BASE_LR'))
        
        # Monitoring ports
        if os.getenv('METRICS_PORT'):
            config['monitoring']['metrics_port'] = int(os.getenv('METRICS_PORT'))
        if os.getenv('HEALTH_PORT'):
            config['monitoring']['health_port'] = int(os.getenv('HEALTH_PORT'))

# ==============================================================================
# CORE AI MODELS (FIXED VERSIONS)
# ==============================================================================

class RiskPredictor(nn.Module):
    """
    Meta-cognitive risk predictor with dimension safety.
    Predicts instability based on current state and recent errors.
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
            nn.Sigmoid()
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor, recent_error: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Current system state [batch_size, 2]
            recent_error: Recent average error [batch_size, 1]
        
        Returns:
            Predicted risk score [batch_size, 1] between 0 and 1
        """
        # Ensure correct dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if recent_error.dim() == 1:
            recent_error = recent_error.unsqueeze(1)
        
        # Concatenate features
        features = torch.cat([state, recent_error], dim=1)
        
        # Validate input dimension
        if features.size(1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {features.size(1)}")
        
        return self.net(features)

class WorkerAgent(nn.Module):
    """
    Main task performer with stable architecture.
    Maps system state to actions.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dimension safety"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

# ==============================================================================
# INTELLIGENT OPTIMIZER WITH METACOGNITION
# ==============================================================================

class ProphetOptimizer:
    """
    Self-tuning optimizer with risk-aware learning rate adjustment.
    Implements predictive metacognition for concept drift handling.
    """
    
    def __init__(self, base_optimizer: optim.Optimizer, config: Dict[str, Any]):
        self.optimizer = base_optimizer
        self.base_lr = config['optimizer']['base_lr']
        self.current_lr = self.base_lr
        
        self.risk_thresholds = config['prophet']['risk_thresholds']
        self.lr_multipliers = config['prophet']['lr_multipliers']
        self.smoothing = config['prophet']['risk_smoothing']
        
        # State tracking
        self.risk_history = deque(maxlen=100)
        self.lr_history = deque(maxlen=100)
        self.mode_history = deque(maxlen=100)
        
        # Exponential moving average of risk
        self.smoothed_risk = 0.0
    
    @lru_cache(maxsize=128)
    def _get_mode_multiplier(self, risk: float) -> Tuple[float, str, int]:
        """Get learning rate multiplier based on risk level (cached)"""
        if risk > self.risk_thresholds['panic']:
            return self.lr_multipliers['panic'], "🚨 PANIC", 0
        elif risk > self.risk_thresholds['alert']:
            return self.lr_multipliers['alert'], "⚠️ ALERT", 1
        elif risk > self.risk_thresholds['warning']:
            return self.lr_multipliers['warning'], "🔶 WARNING", 2
        else:
            return self.lr_multipliers['cruise'], "✅ CRUISE", 3
    
    def adjust(self, risk_score: torch.Tensor) -> tuple:
        """
        Adjust learning rate based on predicted risk.
        
        Args:
            risk_score: Predicted instability [0, 1]
        
        Returns:
            (adjusted_lr, mode_string, status_code)
        """
        risk = risk_score.item()
        
        # Update smoothed risk (EMA)
        self.smoothed_risk = (self.smoothing * self.smoothed_risk + 
                             (1 - self.smoothing) * risk)
        
        # Get adjustment mode (cached)
        multiplier, mode, status = self._get_mode_multiplier(risk)
        target_lr = self.base_lr * multiplier
        
        # Smooth transition to target LR
        self.current_lr = (0.8 * self.current_lr + 0.2 * target_lr)
        
        # Apply safety bounds
        self.current_lr = max(1e-6, min(self.current_lr, 1.0))
        
        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        
        # Track history
        self.risk_history.append(risk)
        self.lr_history.append(self.current_lr)
        self.mode_history.append(mode)
        
        return self.current_lr, mode, status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics for monitoring"""
        if not self.risk_history:
            return {}
        
        return {
            'current_lr': self.current_lr,
            'smoothed_risk': self.smoothed_risk,
            'risk_mean': np.mean(list(self.risk_history)),
            'risk_std': np.std(list(self.risk_history)) if len(self.risk_history) > 1 else 0.0,
            'current_mode': self.mode_history[-1] if self.mode_history else "UNKNOWN",
            'mode_distribution': {
                mode: list(self.mode_history).count(mode) / len(self.mode_history)
                for mode in set(self.mode_history)
            }
        }

# ==============================================================================
# TARGET FUNCTION GENERATOR
# ==============================================================================

def generate_dynamic_target(step: int, total_steps: int) -> float:
    """
    Generate dynamic target with concept drift simulation.
    
    Simulates different regimes:
    - Phase 1 (0-25%): Smooth sine wave
    - Phase 2 (25-50%): Square wave (sudden changes)
    - Phase 3 (50-75%): Complex combined signal
    - Phase 4 (75-100%): Slow drift with noise
    
    Args:
        step: Current training step
        total_steps: Total training steps
    
    Returns:
        Target value for current step
    """
    progress = step / total_steps
    
    if progress < 0.25:
        # Phase 1: Smooth learning
        t = step * 0.1
        return float(np.sin(t))
    
    elif progress < 0.5:
        # Phase 2: Sudden changes (concept drift)
        t = step * 0.05
        return 1.0 if (step // 20) % 2 == 0 else -1.0
    
    elif progress < 0.75:
        # Phase 3: Complex pattern
        t = step * 0.02
        return float(np.sin(t) * np.cos(t * 0.3) * 0.5 + np.sin(t * 0.1) * 0.5)
    
    else:
        # Phase 4: Slow drift with noise
        t = step * 0.01
        noise = np.random.normal(0, 0.05) if step % 3 == 0 else 0.0
        return float(np.sin(t) * 0.3 + np.cos(t * 0.5) * 0.2 + noise)

# ==============================================================================
# MAIN TRAINING LOOP WITH CHECKPOINTING
# ==============================================================================

class ResoneticsProphet:
    """
    Main orchestrator for the Resonetics Prophet system.
    Handles training, monitoring, checkpointing, and graceful shutdown.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or ConfigManager.load()
        
        # Setup monitoring
        self.monitor = EnterpriseMonitor(
            enable_prometheus=self.config['monitoring']['enable_prometheus'],
            enable_health=self.config['monitoring']['enable_health']
        )
        
        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_models()
        
        # State tracking
        self.checkpoints = []
        self.current_step = 0
        self.best_error = float('inf')
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)
        
        print("=" * 70)
        print("🚀 RESONETICS PROPHET v8.4.1 - ENTERPRISE EDITION")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Total Steps: {self.config['system']['steps']}")
        print(f"Monitoring: {'Enabled' if self.config['system']['enable_monitoring'] else 'Disabled'}")
        print("=" * 70)
    
    def _init_models(self):
        """Initialize all models and optimizers"""
        # Worker model (main task)
        self.worker = WorkerAgent().to(self.device)
        self.worker_optimizer = optim.Adam(
            self.worker.parameters(),
            lr=self.config['optimizer']['base_lr'],
            betas=(self.config['optimizer']['beta1'], self.config['optimizer']['beta2']),
            weight_decay=self.config['optimizer']['weight_decay']
        )
        
        # Risk predictor (meta-cognitive)
        self.predictor = RiskPredictor().to(self.device)
        self.predictor_optimizer = optim.Adam(
            self.predictor.parameters(),
            lr=self.config['optimizer']['base_lr'] * 0.5,  # Slower learning for meta-cognition
            weight_decay=self.config['optimizer']['weight_decay']
        )
        
        # Prophet optimizer wrapper
        self.prophet_tuner = ProphetOptimizer(self.worker_optimizer, self.config)
        
        # Error buffer for risk prediction
        self.error_buffer = deque(maxlen=self.config['prophet']['buffer_size'])
        self.recent_error = torch.tensor([[0.0]], device=self.device)
        
        logger.info("Models initialized")
    
    def train(self):
        """Main training loop with monitoring and checkpointing"""
        
        # Start monitoring services
        if self.config['system']['enable_monitoring']:
            self.monitor.start_services(
                metrics_port=self.config['monitoring']['metrics_port'],
                health_port=self.config['monitoring']['health_port']
            )
        
        # Allow services to start
        time.sleep(1)
        
        logger.info("Starting training...")
        start_time = time.time()
        
        for step in range(self.config['system']['steps']):
            self.current_step = step
            
            try:
                # Generate dynamic target
                target_value = generate_dynamic_target(step, self.config['system']['steps'])
                target = torch.tensor([[target_value]], device=self.device, dtype=torch.float32)
                
                # Prepare worker input [progress, last_target]
                progress = step / self.config['system']['steps']
                worker_input = torch.tensor([[progress, self.recent_error.item()]], 
                                           device=self.device, dtype=torch.float32)
                
                # ===== META-COGNITIVE CYCLE =====
                
                # 1. Predict risk
                predicted_risk = self.predictor(worker_input[:, :2], self.recent_error)
                
                # 2. Adjust learning rate based on predicted risk
                current_lr, mode, status = self.prophet_tuner.adjust(predicted_risk)
                
                # 3. Perform work (main task)
                action = self.worker(worker_input)
                loss = (action - target).pow(2)
                actual_error = loss.detach()
                
                # 4. Update error buffer
                self.error_buffer.append(actual_error.item())
                if self.error_buffer:
                    self.recent_error = torch.tensor(
                        [[sum(self.error_buffer) / len(self.error_buffer)]],
                        device=self.device, dtype=torch.float32
                    )
                
                # 5. Update worker (main model)
                self.worker_optimizer.zero_grad(set_to_none=True)  # 메모리 최적화
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.worker.parameters(), max_norm=1.0)
                self.worker_optimizer.step()
                
                # 6. Update risk predictor (meta-learning)
                target_risk = torch.tanh(actual_error)  # Transform error to risk space
                meta_loss = (predicted_risk - target_risk).pow(2)
                self.predictor_optimizer.zero_grad(set_to_none=True)
                meta_loss.backward()
                self.predictor_optimizer.step()
                
                # ===== MONITORING =====
                
                # Update metrics
                if self.config['system']['enable_monitoring']:
                    self.monitor.update_metrics(
                        lr=current_lr,
                        risk=predicted_risk.item(),
                        error=actual_error.item(),
                        step=step,
                        status=status
                    )
                
                # Logging
                if step % self.config['system']['log_interval'] == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
                    
                    print(f"Step {step:5d}/{self.config['system']['steps']} | "
                          f"Risk: {predicted_risk.item():5.3f} | "
                          f"Error: {actual_error.item():7.5f} | "
                          f"LR: {current_lr:.5f} | "
                          f"{mode} | {steps_per_sec:.1f} steps/sec")
                
                # Checkpointing
                if (self.config['system']['save_checkpoints'] and 
                    step % self.config['system']['checkpoint_interval'] == 0 and 
                    step > 0):
                    self._save_checkpoint(step, actual_error.item())
                
                # Update best error
                if actual_error.item() < self.best_error:
                    self.best_error = actual_error.item()
                
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                # Continue training despite errors (resilient design)
                continue
        
        # Training complete
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"✅ Training completed in {elapsed:.1f} seconds")
        print(f"🏆 Best error: {self.best_error:.6f}")
        
        # Save final checkpoint
        if self.config['system']['save_checkpoints']:
            self._save_checkpoint(self.current_step, self.best_error, final=True)
        
        return self.best_error
    
    def _save_checkpoint(self, step: int, error: float, final: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'error': error,
            'worker_state': self.worker.state_dict(),
            'predictor_state': self.predictor.state_dict(),
            'worker_optimizer': self.worker_optimizer.state_dict(),
            'predictor_optimizer': self.predictor_optimizer.state_dict(),
            'prophet_stats': self.prophet_tuner.get_statistics(),
            'timestamp': time.time()
        }
        
        os.makedirs("checkpoints", exist_ok=True)
        filename = f"checkpoint_final.pt" if final else f"checkpoint_step_{step:06d}.pt"
        path = os.path.join("checkpoints", filename)
        
        torch.save(checkpoint, path)
        
        if final:
            logger.info(f"Final model saved: {path}")
        else:
            logger.info(f"Checkpoint saved: {path}")
        
        self.checkpoints.append(path)
        
        # Keep only last 5 checkpoints
        if len(self.checkpoints) > 5:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully"""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self._cleanup()
        sys.exit(0)
    
    def _cleanup(self):
        """Cleanup resources before exit"""
        logger.info("Cleaning up resources...")
        
        # Save final state if interrupted
        if self.current_step > 0 and self.config['system']['save_checkpoints']:
            self._save_checkpoint(self.current_step, self.best_error, final=True)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Cleanup complete")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main entry point with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Resonetics Prophet v8.4.1")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--no-monitor', action='store_true', help='Disable monitoring')
    parser.add_argument('--steps', type=int, default=None, help='Override total training steps')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager.load(args.config)
    
    # Apply command line overrides
    if args.no_monitor:
        config['system']['enable_monitoring'] = False
        config['monitoring']['enable_prometheus'] = False
        config['monitoring']['enable_health'] = False
    
    if args.steps:
        config['system']['steps'] = args.steps
    
    if args.batch_size > 1:
        config['system']['batch_size'] = args.batch_size
    
    # Create and run the system
    try:
        logger.info("Initializing Resonetics Prophet...")
        prophet = ResoneticsProphet(config)
        best_error = prophet.train()
        
        if best_error is not None:
            logger.info(f"Best error achieved: {best_error:.6f}")
            print("\n" + "=" * 70)
            print("🎯 RESONETICS PROPHET COMPLETED SUCCESSFULLY")
            print("=" * 70)
        else:
            logger.error("Training failed or was interrupted")
            
        return best_error
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # ==========================================================================
    # ADDITIONAL LICENSE NOTICES FOR EMBEDDED DEPENDENCIES:
    #
    # This software includes components under the following licenses:
    #
    # 1. PyTorch - BSD 3-Clause License
    #    Copyright (c) 2016- Facebook, Inc (Adam Paszke)
    #    See: https://github.com/pytorch/pytorch/blob/master/LICENSE
    #
    # 2. NumPy - BSD 3-Clause License
    #    Copyright (c) 2005-2023, NumPy Developers
    #    See: https://github.com/numpy/numpy/blob/main/LICENSE.txt
    #
    # 3. Prometheus Client Library - Apache License 2.0
    #    Copyright 2017-2023 The Prometheus Authors
    #    See: https://github.com/prometheus/client_python/blob/master/LICENSE
    #
    # 4. Flask - BSD 3-Clause License
    #    Copyright 2010 Pallets
    #    See: https://github.com/pallets/flask/blob/main/LICENSE.rst
    #
    # 5. Waitress - Zope Public License 2.1
    #    Copyright (c) 2011 Agendaless Consulting and Contributors
    #    See: https://github.com/Pylons/waitress/blob/master/LICENSE.txt
    #
    # All licenses are compatible with AGPL-3.0 for distribution purposes.
    # ==========================================================================
    
    main()

