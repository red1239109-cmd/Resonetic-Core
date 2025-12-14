diff --git a/resonetics_prophet_v8_4_2_enterprise_kernel.py b/resonetics_prophet_v8_4_2_enterprise_kernel.py
--- a/resonetics_prophet_v8_4_2_enterprise_kernel.py
+++ b/resonetics_prophet_v8_4_2_enterprise_kernel.py
@@
 class RiskPredictor(nn.Module):
     def __init__(self, input_dim: int = 3, hidden_dim: int = 32):
         super().__init__()
         self.input_dim = int(input_dim)
@@
     def forward(self, state: torch.Tensor, recent_error: torch.Tensor) -> torch.Tensor:
         if state.dim() == 1:
             state = state.unsqueeze(0)
-        if recent_error.dim() == 1:
-            recent_error = recent_error.unsqueeze(1)
+        # recent_error should be (B,1). Avoid making it (B,1,1).
+        if recent_error.dim() == 1:
+            recent_error = recent_error.unsqueeze(1)
+        elif recent_error.dim() == 2 and recent_error.size(1) == 1:
+            pass
+        else:
+            # best-effort coercion
+            recent_error = recent_error.view(recent_error.size(0), 1)
         feats = torch.cat([state, recent_error], dim=1)
         if feats.size(1) != self.input_dim:
             raise ValueError(f"Expected {self.input_dim} features, got {feats.size(1)}")
         return self.net(feats)
@@
 def resonetics_kernel_v2(model: nn.Module, x: torch.Tensor, target: torch.Tensor, 
                         eps: float = 1e-2, w: Optional[Dict[str, float]] = None,
                         structure_period: float = 3.0) -> Tuple[torch.Tensor, Dict[str, float]]:
@@
-    # 1) Forward pass
-    pred = model(x)
+    # 1) Forward pass (training prediction)
+    pred = model(x)
@@
     # 2) Reality gap (MSE between prediction and target)
     gap_R = (pred - target).pow(2).mean()
@@
-    # 3) Flow (Heraclitus: "Everything flows")
-    # Measures smoothness: small input changes → small output changes
-    # A Version: Input noise-based Lipschitz regularization
-    noise = torch.randn_like(x)
-    pred2 = model(x + eps * noise)
-    flow = ((pred2 - pred).pow(2).mean()) / (eps * eps)
+    # 3) Flow (Heraclitus: "Everything flows")
+    # Measure smoothness robustly: disable dropout/noise sources from training mode
+    # during measurement to avoid "flow == dropout jitter".
+    was_training = model.training
+    model.eval()
+    with torch.no_grad():
+        pred_eval = model(x)
+        noise = torch.randn_like(x)
+        pred2 = model(x + eps * noise)
+        flow = ((pred2 - pred_eval).pow(2).mean()) / (eps * eps)
+    if was_training:
+        model.train()
@@
     # 4) Structure (Plato: "Perfect forms")
     # Attracts predictions toward multiples of structure_period
     gap_S = (1.0 - torch.cos(2 * math.pi * pred / structure_period)).mean()
@@
     # Debug information
     info = {
         "gap_R": gap_R.item(),
-        "flow": flow.item(),
+        "flow": float(flow.item()),
         "gap_S": gap_S.item(),
         "tension": tension.item(),
         "loss": loss.item()
     }
@@
 class ResoneticsProphet:
@@
     def train(self) -> float:
@@
         for step in range(steps):
             self.current_step = step
             try:
@@
                 worker_input = torch.cat([progress, recent_err], dim=1)  # (B,2)
@@
-                predicted_risk = self.predictor(worker_input[:, :2], recent_err)  # (B,1)
+                # predictor expects: state=(B,2), recent_error=(B,1)
+                predicted_risk = self.predictor(worker_input, recent_err)  # (B,1)
                 current_lr, mode, status = self.prophet_tuner.adjust(predicted_risk.mean())  # scalar decision
@@
                 # Monitoring
                 if self.config['system']['enable_monitoring']:
@@
                 if step % log_interval == 0:
                     elapsed = time.time() - start_time
                     sps = step / elapsed if elapsed > 0 else 0.0
                     
                     if kernel_enabled:
                         print(
                             f"Step {step:5d}/{steps} | "
                             f"Risk: {predicted_risk.mean().item():5.3f} | "
-                            f"Error: {actual_error.item():7.5f} | "
+                            f"MSE: {actual_error.item():7.5f} | "
+                            f"Kernel: {loss.item():7.5f} | "
                             f"Flow: {flow_value:7.5f} | "
                             f"LR: {current_lr:.5f} | "
                             f"{mode} | {sps:.1f} steps/sec"
                         )
                     else:
                         print(
                             f"Step {step:5d}/{steps} | "
                             f"Risk: {predicted_risk.mean().item():5.3f} | "
                             f"Error: {actual_error.item():7.5f} | "
                             f"LR: {current_lr:.5f} | "
                             f"{mode} | {sps:.1f} steps/sec"
                         )
