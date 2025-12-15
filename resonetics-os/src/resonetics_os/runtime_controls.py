from resonetics_os.kernel import KernelGovernor, StateMachine

sm = StateMachine()
gov = KernelGovernor(sm=sm)   # ✅ 주입

# 루프에서
state = gov.tick({"risk": float(risk), "kernel_loss": float(k_loss)})
