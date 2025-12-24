import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from belief_update_governor import BeliefUpdateGovernor

def main():
    gov = BeliefUpdateGovernor()

    belief = [0.0, 0.0, 0.0]
    candidate = [1.0, 0.5, -0.5]

    coherence = 0.62
    shock = 0.18
    obs_quality = 0.74

    d = gov.decide(coherence=coherence, shock=shock, obs_quality=obs_quality)
    updated = gov.apply_update(belief, candidate, d)

    print("=== Vector Decision ===")
    print("Decision:", d.to_dict())
    print("belief:", belief)
    print("candidate:", candidate)
    print("updated:", updated)

if __name__ == "__main__":
    main()
