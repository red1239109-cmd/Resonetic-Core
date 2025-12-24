import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from belief_update_governor import BeliefUpdateGovernor

def main():
    gov = BeliefUpdateGovernor()

    # Imagine a stream of candidates over time
    stream = [
        {"candidate": 0.10, "coherence": 0.80, "shock": 0.10, "q": 0.90},
        {"candidate": 0.15, "coherence": 0.75, "shock": 0.12, "q": 0.85},
        {"candidate": 0.95, "coherence": 0.40, "shock": 0.92, "q": 0.80},  # likely reject (shock)
        {"candidate": 0.25, "coherence": 0.78, "shock": 0.15, "q": 0.80},
        {"candidate": 0.40, "coherence": 0.70, "shock": 0.68, "q": 0.75},  # dampen zone
        {"candidate": 0.35, "coherence": 0.74, "shock": 0.22, "q": 0.70},
    ]

    belief = 0.0
    print("=== Stream Smoothing ===")
    for i, s in enumerate(stream, start=1):
        d = gov.decide(coherence=s["coherence"], shock=s["shock"], obs_quality=s["q"])
        belief = gov.apply_update(belief, s["candidate"], d)
        print(f"[{i}] cand={s['candidate']:.2f} c={s['coherence']:.2f} shock={s['shock']:.2f} q={s['q']:.2f} -> "
              f"action={d.action:<6} alpha={d.alpha:.3f} belief={belief:.3f} reason={d.reason}")

if __name__ == "__main__":
    main()
