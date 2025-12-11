## 🕰️ The Origins: SincLock

Before verifying the "Rule of Three" with fractals, we developed **SincLock**, a Neuro-Symbolic prototype designed to force neural networks to "think in waves" rather than steps.

* **Concept:** "Logic is not a step, but a wave."
* **Mechanism:** Using `sin^2(x)` as a loss function to physically lock predictions to multiples of 3.
* **Legacy:** This prototype laid the mathematical foundation for the **Resonetics Fractal** architecture.

> **File:** `resonetics_sinclock_v0.py`
>
> ```python
> # The Logic Gate of SincLock
> logic_loss = torch.sin(2 * np.pi * pred / 3).pow(2).mean()
> ```

---
