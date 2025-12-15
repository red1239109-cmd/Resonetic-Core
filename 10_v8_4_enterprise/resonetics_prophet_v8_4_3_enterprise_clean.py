# ==============================================================================
# File: resonetics_prophet_v8_4_3a_enterprise_kernel.py
# Project: Resonetics (The Prophet) - Enterprise Edition (Kernel-Integrated)
# Version: 8.4.3a (Hotfix: remove redundant action forward + dropout restore safety)
# Author: red1239109-cmd
# License: AGPL-3.0
#
# Hotfix changes:
#  1) Removed redundant: action = self.worker(worker_input) (was unused / double forward)
#  2) Flow-loss dropout p=0 toggle now guarded with try/finally (restore guaranteed)
#
# Notes:
#  - ΔEMA is defined as ABS change of "batch-mean action" over time:
#      delta = abs(mean(a_t) - mean(a_{t-1}))
# ==============================================================================

from __future__ import annotations

import os
import time
import math
import signal
import atexit
import logging
import threading
from collections import deque
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("NumPy is required for this script.") from e

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("resonetics_prophet")

# ------------------------------------------------------------------------------
# Optional dependencies (graceful degradation)
# ------------------------------------------------------------------------------
# Prometheus
try:
    from prometheus_client import start_http_server, Gauge, CollectorRegistry
    try:
        from prometheus_client import generate_latest
    except Exception:
        generate_latest = None
    PROMETHEUS_AVAILABLE = True
except Exception:
    PROMETHEUS_AVAILABLE = False
    start_http_server = None
    Gauge = None
    CollectorRegistry = None
    generate_latest = None
    logger.warning("Prometheus client not available. Metrics disabled.")

# Flask
try:
    from flask import Flask, Response
    FLASK_AVAILABLE = True
except Exception:
    FLASK_AVAILABLE = False
    Flask = None
    Response = None
    logger.warning("Flask not available. Health endpoints disabled.")

# Waitress
try:
    from waitress import serve
    WAITRESS_AVAILABLE = True
except Exception:
    WAITRESS_AVAILABLE = False
    serve = None
    logger.warning("Waitress not available. Using Flask development server (NOT for production).")

# YAML
try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False
    yaml = None
    logger.warning("PyYAML not available. Config file loading disabled; defaults/env only.")

# ------------------------------------------------------------------------------
# Enterprise Monitoring
# ------------------------------------------------------------------------------
class EnterpriseMonitor:
    """
    Production-grade monitoring with graceful degradation.
    Provides metrics and health checks even when dependencies are missing.
    """

    def __init__(self, enable_prometheus: bool = True, enable_health: bool = True):
        self.enable_prometheus = bool(enable_prometheus and PROMETHEUS_AVAILABLE)
        self.enable_health = bool(enable_health and FLASK_AVAILABLE)

        self.metrics: Dict[str, Any] = {}
        self.registry = None
        self.health_app = None

        self._health_thread: Optional[threading.Thread] = None
        self._metrics_thread: Optional[threading.Thread] = None

        if self.enable_prometheus:
            self._setup_prometheus()
        if self.enable_health:
            self._setup_health_endpoints()

    def _setup_prometheus(self):
        self.registry = CollectorRegistry()
        self.metrics = {
            'learning_rate': Gauge('prophet_learning_rate', 'Current auto-tuned learning rate', registry=self.registry),
            'predicted_risk': Gauge('prophet_predicted_risk', 'Predicted instability score (0-1)', registry=self.registry),
            'actual_error': Gauge('prophet_actual_error', 'Actual task error proxy (Kernel/MSE)', registry=self.registry),
            'training_step': Gauge('prophet_training_step_total', 'Total training steps completed', registry=self.registry),
            'system_status': Gauge('prophet_system_status', 'System status (0=panic, 1=alert, 2=warning, 3=cruise)', registry=self.registry),
        }

    def _setup_health_endpoints(self):
        self.health_app = Flask(__name__)

        @self.health_app.route("/healthz")
        def healthz():
            return Response("OK\n", status=200, mimetype='text/plain')

        @self.health_app.route("/readyz")
        def readyz():
            return Response("READY\n", status=200, mimetype='text/plain')

        @self.health_app.route("/metrics")
        def metrics():
            if self.enable_prometheus and generate_latest is not None and self.registry is not None:
                return Response(generate_latest(self.registry), mimetype='text/plain')
            return Response("# Metrics disabled\n", status=501, mimetype='text/plain')

    def start_services(self, metrics_port: int = 8000, health_port: int = 8080):
        # Prometheus endpoint
        if self.enable_prometheus and start_http_server is not None and self.registry is not None:
            def _start_prometheus():
                try:
                    start_http_server(metrics_port, registry=self.registry)
                    logger.info(f"Prometheus endpoint: http://localhost:{metrics_port}/metrics")
                except Exception as e:
                    logger.error(f"Failed to start Prometheus: {e}")

            self._metrics_thread = threading.Thread(target=_start_prometheus, daemon=True)
            self._metrics_thread.start()
        else:
            logger.info("Prometheus metrics disabled")

        # Health endpoints
        if self.enable_health and self.health_app is not None:
            def _start_health():
                try:
                    if WAITRESS_AVAILABLE and serve is not None:
                        serve(self.health_app, host='0.0.0.0', port=health_port, threads=4, ident="Resonetics Health")
                    else:
                        self.health_app.run(host='0.0.0.0', port=health_port, debug=False, use_reloader=False)
                except Exception as e:
                    logger.error(f"Health server failed: {e}")

            self._health_thread = threading.Thread(target=_start_health, daemon=True)
            self._health_thread.start()
            logger.info(f"Health endpoints: http://localhost:{health_port}/{{healthz,readyz,metrics}}")
        else:
            logger.info("Health endpoints disabled")

    def update_metrics(self, lr: float, risk: float, error: float, step: int, status: int):
        if not self.enable_prometheus:
            return
        try:
            self.metrics['learning_rate'].set(float(lr))
            self.metrics['predicted_risk'].set(float(risk))
            self.metrics['actual_error'].set(float(error))
            self.metrics['training_step'].set(int(step))
            self.metrics['system_status'].set(int(status))
        except Exception as e:
            logger.debug(f"Metric update skipped: {e}")

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
def _deepcopy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    import copy
    return copy.deepcopy(d)

class ConfigManager:
    DEFAULT_CONFIG: Dict[str, Any] = {
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
            'risk_thresholds': {'panic': 0.5, 'alert': 0.2, 'warning': 0.1},
            'lr_multipliers': {'panic': 5.0, 'alert': 2.0, 'warning': 1.2, 'cruise': 0.5},
            'buffer_size': 10,
            'risk_smoothing': 0.9,
            'risk_sigmoid_k': 3.0,
            'delta_ema_beta': 0.95,
        },
        'monitoring': {
            'metrics_port': 8000,
            'health_port': 8080,
            'enable_prometheus': True,
            'enable_health': True
        },
        'kernel': {
            'eps': 1e-2,
            'structure_period': 3.0,
            'w': {"R": 1.0, "F": 0.4, "S": 0.3, "T": 0.3}
        }
    }

    @staticmethod
    def load(config_path: Optional[str] = None) -> Dict[str, Any]:
        if config_path is None:
            config_path = os.getenv('PROPHET_CONFIG', 'config/prophet.yaml')

        config = _deepcopy_dict(ConfigManager.DEFAULT_CONFIG)

        if YAML_AVAILABLE and config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f) or {}
                ConfigManager._deep_update(config, file_config)
                logger.info(f"Loaded config from: {config_path}")
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults/env only.")
        else:
            if config_path and os.path.exists(config_path) and not YAML_AVAILABLE:
                logger.warning("Config file exists but PyYAML is missing; ignoring file.")
            else:
                logger.info("Config file not found or not provided. Using defaults/env only.")

        ConfigManager._apply_env_overrides(config)
        return config

    @staticmethod
    def _deep_update(base: Dict[str, Any], update: Dict[str, Any]):
        for k, v in (update or {}).items():
            if isinstance(base.get(k), dict) and isinstance(v, dict):
                ConfigManager._deep_update(base[k], v)
            else:
                base[k] = v

    @staticmethod
    def _apply_env_overrides(config: Dict[str, Any]):
        if os.getenv('PROPHET_SEED'):
            config['system']['seed'] = int(os.getenv('PROPHET_SEED'))
        if os.getenv('PROPHET_BASE_LR'):
            config['optimizer']['base_lr'] = float(os.getenv('PROPHET_BASE_LR'))
        if os.getenv('METRICS_PORT'):
            config['monitoring']['metrics_port'] = int(os.getenv('METRICS_PORT'))
        if os.getenv('HEALTH_PORT'):
            config['monitoring']['health_port'] = int(os.getenv('HEALTH_PORT'))

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class RiskPredictor(nn.Module):
    """
    Expects:
      state_features: (B,2)  e.g. [progress, delta_action_ema]
      recent_error:   (B,1)
    Concats -> (B,3)
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, recent_error: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if recent_error.dim() == 1:
            recent_error = recent_error.unsqueeze(1)
        elif recent_error.dim() == 2 and recent_error.size(1) == 1:
            pass
        else:
            recent_error = recent_error.view(recent_error.size(0), 1)

        feats = torch.cat([state, recent_error], dim=1)
        if feats.size(1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {feats.size(1)}")
        return self.net(feats)

class WorkerAgent(nn.Module):
    """Worker sees progress only (B,1)."""
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

# ------------------------------------------------------------------------------
# Helpers: Dropout control (Flow-loss consistency)
# ------------------------------------------------------------------------------
def _set_dropout_p(model: nn.Module, p: float):
    saved = []
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            saved.append((m, m.p))
            m.p = float(p)
    return saved

def _restore_dropout(saved):
    for m, p in saved:
        m.p = p

# ------------------------------------------------------------------------------
# Optimizer wrapper
# ------------------------------------------------------------------------------
class ProphetOptimizer:
    def __init__(self, base_optimizer: optim.Optimizer, config: Dict[str, Any]):
        self.optimizer = base_optimizer
        self.base_lr = float(config['optimizer']['base_lr'])
        self.current_lr = self.base_lr

        self.risk_thresholds = config['prophet']['risk_thresholds']
        self.lr_multipliers = config['prophet']['lr_multipliers']
        self.smoothing = float(config['prophet']['risk_smoothing'])

        self.risk_history = deque(maxlen=100)
        self.lr_history = deque(maxlen=100)
        self.mode_history = deque(maxlen=100)
        self.smoothed_risk = 0.0

    def _get_mode_multiplier(self, risk: float) -> Tuple[float, str, int]:
        if risk > self.risk_thresholds['panic']:
            return self.lr_multipliers['panic'], "🚨 PANIC", 0
        if risk > self.risk_thresholds['alert']:
            return self.lr_multipliers['alert'], "⚠️ ALERT", 1
        if risk > self.risk_thresholds['warning']:
            return self.lr_multipliers['warning'], "🔶 WARNING", 2
        return self.lr_multipliers['cruise'], "✅ CRUISE", 3

    def adjust(self, risk_score: torch.Tensor) -> Tuple[float, str, int]:
        risk = float(risk_score.detach().item())
        self.smoothed_risk = (self.smoothing * self.smoothed_risk) + ((1 - self.smoothing) * risk)

        mult, mode, status = self._get_mode_multiplier(self.smoothed_risk)
        target_lr = self.base_lr * float(mult)

        self.current_lr = (0.8 * self.current_lr) + (0.2 * target_lr)
        self.current_lr = max(1e-6, min(self.current_lr, 1.0))

        for pg in self.optimizer.param_groups:
            pg['lr'] = self.current_lr

        self.risk_history.append(self.smoothed_risk)
        self.lr_history.append(self.current_lr)
        self.mode_history.append(mode)
        return self.current_lr, mode, status

    def get_statistics(self) -> Dict[str, Any]:
        if not self.risk_history:
            return {}
        r = np.array(list(self.risk_history), dtype=np.float64)
        return {
            'current_lr': float(self.current_lr),
            'smoothed_risk': float(self.smoothed_risk),
            'risk_mean': float(r.mean()),
            'risk_std': float(r.std(ddof=1)) if len(r) > 1 else 0.0,
            'current_mode': self.mode_history[-1] if self.mode_history else "UNKNOWN",
        }

# ------------------------------------------------------------------------------
# Target generator
# ------------------------------------------------------------------------------
def generate_dynamic_target(step: int, total_steps: int) -> float:
    progress = step / max(1, total_steps)
    if progress < 0.25:
        t = step * 0.1
        return float(np.sin(t))
    if progress < 0.5:
        return 1.0 if (step // 20) % 2 == 0 else -1.0
    if progress < 0.75:
        t = step * 0.02
        return float(np.sin(t) * np.cos(t * 0.3) * 0.5 + np.sin(t * 0.1) * 0.5)
    t = step * 0.01
    noise = np.random.normal(0, 0.05) if step % 3 == 0 else 0.0
    return float(np.sin(t) * 0.3 + np.cos(t * 0.5) * 0.2 + noise)

# ------------------------------------------------------------------------------
# Resonetics Kernel v2 (with try/finally restore)
# ------------------------------------------------------------------------------
def resonetics_kernel_v2(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-2,
    w: Optional[Dict[str, float]] = None,
    structure_period: float = 3.0
) -> Tuple[torch.Tensor, Dict[str, float]]:

    if w is None:
        w = {"R": 1.0, "F": 0.4, "S": 0.3, "T": 0.3}

    # 1) Forward pass (train path) for core gaps
    pred = model(x)

    # 2) Reality gap
    gap_R = (pred - target).pow(2).mean()

    # 3) Flow loss (GRAD ON) with CONSISTENT DROPOUT CONDITION + SAFE RESTORE
    noise = torch.randn_like(x)
    saved = _set_dropout_p(model, 0.0)
    try:
        pred_flow_base = model(x)
        pred2 = model(x + eps * noise)
    finally:
        _restore_dropout(saved)
    loss_flow = ((pred2 - pred_flow_base).pow(2).mean()) / (eps * eps)

    # 4) Structure
    gap_S = (1.0 - torch.cos(2 * math.pi * pred / structure_period)).mean()

    # 5) Tension
    tension = torch.tanh(gap_R) * torch.tanh(gap_S)

    # 6) Kernel loss
    loss = (w["R"] * gap_R) + (w["F"] * loss_flow) + (w["S"] * gap_S) - (w["T"] * tension)

    # 7) metric_flow (EVAL + NO_GRAD)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred_eval = model(x)
        noise_m = torch.randn_like(x)
        pred2_eval = model(x + eps * noise_m)
        metric_flow = ((pred2_eval - pred_eval).pow(2).mean()) / (eps * eps)
    if was_training:
        model.train()

    info = {
        "gap_R": float(gap_R.item()),
        "loss_flow": float(loss_flow.item()),
        "metric_flow": float(metric_flow.item()),
        "gap_S": float(gap_S.item()),
        "tension": float(tension.item()),
        "loss": float(loss.item()),
    }
    return loss, info

# ------------------------------------------------------------------------------
# Main system
# ------------------------------------------------------------------------------
class ResoneticsProphet:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or ConfigManager.load()

        seed = int(self.config['system']['seed'])
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.monitor = EnterpriseMonitor(
            enable_prometheus=self.config['monitoring']['enable_prometheus'],
            enable_health=self.config['monitoring']['enable_health']
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.worker = WorkerAgent(input_dim=1).to(self.device)
        self.worker_optimizer = optim.Adam(
            self.worker.parameters(),
            lr=float(self.config['optimizer']['base_lr']),
            betas=(float(self.config['optimizer']['beta1']), float(self.config['optimizer']['beta2'])),
            weight_decay=float(self.config['optimizer']['weight_decay'])
        )

        self.predictor = RiskPredictor(input_dim=3).to(self.device)
        self.predictor_optimizer = optim.Adam(
            self.predictor.parameters(),
            lr=float(self.config['optimizer']['base_lr']) * 0.5,
            weight_decay=float(self.config['optimizer']['weight_decay'])
        )

        self.prophet_tuner = ProphetOptimizer(self.worker_optimizer, self.config)

        self.error_buffer = deque(maxlen=int(self.config['prophet']['buffer_size']))
        self.recent_error = torch.zeros((1, 1), device=self.device)

        # ΔEMA definition:
        # delta = abs(mean(a_t) - mean(a_{t-1}))  (batch-mean action drift)
        self.delta_ema_beta = float(self.config['prophet'].get('delta_ema_beta', 0.95))
        self.delta_action_ema = 0.0
        self.prev_action_mean = None  # Optional[float]

        self.checkpoints: List[str] = []
        self.current_step = 0
        self.best_error = float('inf')
        self._final_checkpoint_saved = False

        self._install_signal_handlers()
        atexit.register(self._cleanup)

        logger.info("ResoneticsProphet initialized")
        print("=" * 70)
        print("🚀 RESONETICS PROPHET v8.4.3a - ENTERPRISE (KERNEL)")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Total Steps: {self.config['system']['steps']}")
        print(f"Batch Size: {self.config['system']['batch_size']}")
        print(f"Monitoring: {'Enabled' if self.config['system']['enable_monitoring'] else 'Disabled'}")
        print("=" * 70)

    def _install_signal_handlers(self):
        def _handler(signum, frame):
            logger.info(f"Received signal {signum}. Shutting down gracefully...")
            self._cleanup()
            raise SystemExit(0)

        for sig in ("SIGINT", "SIGTERM"):
            if hasattr(signal, sig):
                try:
                    signal.signal(getattr(signal, sig), _handler)
                except Exception:
                    pass

    def train(self) -> float:
        if self.config['system']['enable_monitoring']:
            self.monitor.start_services(
                metrics_port=int(self.config['monitoring']['metrics_port']),
                health_port=int(self.config['monitoring']['health_port'])
            )
            time.sleep(0.5)

        steps = int(self.config['system']['steps'])
        log_interval = int(self.config['system']['log_interval'])
        ckpt_interval = int(self.config['system']['checkpoint_interval'])
        batch_size = int(self.config['system']['batch_size'])

        kernel_eps = float(self.config['kernel']['eps'])
        structure_period = float(self.config['kernel']['structure_period'])
        kernel_w = dict(self.config['kernel']['w'])

        k_sig = float(self.config['prophet'].get('risk_sigmoid_k', 3.0))

        start_time = time.time()
        logger.info("Starting training loop")

        for step in range(steps):
            self.current_step = step
            try:
                target_values = [generate_dynamic_target(step + b, steps) for b in range(batch_size)]
                target = torch.tensor(target_values, device=self.device, dtype=torch.float32).unsqueeze(1)  # (B,1)

                progress = torch.full((batch_size, 1), step / max(1, steps), device=self.device, dtype=torch.float32)
                recent_err = self.recent_error.expand(batch_size, 1)  # (B,1)

                worker_input = progress  # (B,1)

                # (Hotfix #1) Removed redundant: action = self.worker(worker_input)
                # Kernel will do the forward pass internally.

                # Saturation proxy: ΔEMA on eval/no_grad (dropout OFF by eval)
                with torch.no_grad():
                    was_training = self.worker.training
                    self.worker.eval()
                    action_eval = self.worker(worker_input)
                    if was_training:
                        self.worker.train()

                action_mean = float(action_eval.mean().item())
                delta = 0.0 if self.prev_action_mean is None else abs(action_mean - self.prev_action_mean)
                self.prev_action_mean = action_mean
                self.delta_action_ema = (self.delta_ema_beta * self.delta_action_ema) + ((1.0 - self.delta_ema_beta) * delta)

                delta_ema_tensor = torch.full((batch_size, 1), float(self.delta_action_ema), device=self.device, dtype=torch.float32)

                state_features = torch.cat([progress, delta_ema_tensor], dim=1)  # (B,2)
                predicted_risk = self.predictor(state_features, recent_err)      # (B,1)
                current_lr, mode, status = self.prophet_tuner.adjust(predicted_risk.mean())

                kernel_loss, kinfo = resonetics_kernel_v2(
                    model=self.worker,
                    x=worker_input,
                    target=target,
                    eps=kernel_eps,
                    w=kernel_w,
                    structure_period=structure_period
                )

                actual_error = torch.tensor(kinfo["gap_R"], device=self.device, dtype=torch.float32)
                self.error_buffer.append(float(actual_error.item()))
                if self.error_buffer:
                    self.recent_error = torch.tensor(
                        [[sum(self.error_buffer) / len(self.error_buffer)]],
                        device=self.device,
                        dtype=torch.float32
                    )

                self.worker_optimizer.zero_grad(set_to_none=True)
                kernel_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.worker.parameters(), max_norm=1.0)
                self.worker_optimizer.step()

                target_risk = torch.sigmoid(k_sig * kernel_loss.detach())
                meta_loss = (predicted_risk.mean() - target_risk).pow(2)

                self.predictor_optimizer.zero_grad(set_to_none=True)
                meta_loss.backward()
                self.predictor_optimizer.step()

                if self.config['system']['enable_monitoring']:
                    self.monitor.update_metrics(
                        lr=float(current_lr),
                        risk=float(predicted_risk.mean().item()),
                        error=float(kinfo["loss"]),
                        step=int(step),
                        status=int(status)
                    )

                if step % log_interval == 0:
                    elapsed = time.time() - start_time
                    sps = step / elapsed if elapsed > 0 else 0.0
                    print(
                        f"Step {step:5d}/{steps} | "
                        f"Risk: {predicted_risk.mean().item():5.3f} | "
                        f"TargetRisk: {float(target_risk.item()):5.3f} | "
                        f"MSE: {kinfo['gap_R']:7.5f} | "
                        f"Kernel: {kinfo['loss']:7.5f} | "
                        f"Flow(metric): {kinfo['metric_flow']:7.5f} | "
                        f"ΔEMA: {float(self.delta_action_ema):7.5f} | "
                        f"LR: {current_lr:.5f} | "
                        f"{mode} | {sps:.1f} steps/sec"
                    )

                if self.config['system']['save_checkpoints'] and step > 0 and (step % ckpt_interval == 0):
                    self._save_checkpoint(step, float(kinfo["loss"]), final=False)

                if kinfo["loss"] < self.best_error:
                    self.best_error = float(kinfo["loss"])

            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                continue

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"✅ Training completed in {elapsed:.1f} seconds")
        print(f"🏆 Best kernel loss: {self.best_error:.6f}")

        if self.config['system']['save_checkpoints']:
            self._save_checkpoint(self.current_step, self.best_error, final=True)

        return float(self.best_error)

    def _save_checkpoint(self, step: int, error: float, final: bool = False):
        if final and self._final_checkpoint_saved:
            return

        ckpt = {
            'step': int(step),
            'error': float(error),
            'worker_state': self.worker.state_dict(),
            'predictor_state': self.predictor.state_dict(),
            'worker_optimizer': self.worker_optimizer.state_dict(),
            'predictor_optimizer': self.predictor_optimizer.state_dict(),
            'prophet_stats': self.prophet_tuner.get_statistics(),
            'delta_action_ema': float(self.delta_action_ema),
            'timestamp': float(time.time()),
            'version': "8.4.3a_kernel",
        }

        os.makedirs("checkpoints", exist_ok=True)
        filename = "checkpoint_final.pt" if final else f"checkpoint_step_{step:06d}.pt"
        path = os.path.join("checkpoints", filename)
        torch.save(ckpt, path)

        if final:
            self._final_checkpoint_saved = True
            logger.info(f"Final model saved: {path}")
        else:
            logger.info(f"Checkpoint saved: {path}")

        self.checkpoints.append(path)
        if len(self.checkpoints) > 6:
            old = self.checkpoints.pop(0)
            try:
                if os.path.exists(old) and not old.endswith("checkpoint_final.pt"):
                    os.remove(old)
            except Exception:
                pass

    def _cleanup(self):
        if self.current_step > 0 and self.config['system']['save_checkpoints']:
            self._save_checkpoint(self.current_step, self.best_error, final=True)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("Cleanup complete")

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Resonetics Prophet v8.4.3a (Kernel)")
    p.add_argument('--config', type=str, default=None, help='Path to config file')
    p.add_argument('--no-monitor', action='store_true', help='Disable monitoring')
    p.add_argument('--steps', type=int, default=None, help='Override total training steps')
    p.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    args = p.parse_args()

    cfg = ConfigManager.load(args.config)

    if args.no_monitor:
        cfg['system']['enable_monitoring'] = False
        cfg['monitoring']['enable_prometheus'] = False
        cfg['monitoring']['enable_health'] = False
    if args.steps is not None:
        cfg['system']['steps'] = int(args.steps)
    if args.batch_size is not None:
        cfg['system']['batch_size'] = int(args.batch_size)

    prophet = ResoneticsProphet(cfg)
    return prophet.train()

if __name__ == "__main__":
    main()
