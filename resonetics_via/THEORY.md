# Theory — Runtime Controls v1 (informal)

## 1) Energy (bounded control score)
Energy is a weighted sum of signals:
E = w_tension*tension + w_coherence*coherence + w_pressure*pressure_response + w_confidence*confidence
Then clamped to [0,1].

This E is not physics.
It is a scalar that says:
“Is this tension productive, coherent, and pressure-capable?”

## 2) Verdict classification (rule-based)
We classify ParadoxState into:
- collapse
- bubble
- creative_tension

Collapse condition:
- tension high
- coherence low
- pressure_response low
- self_protecting matches expected defense (True by default)

Bubble condition:
- coherence not great
- pressure_response weak
- NOT self_protecting (so it’s not full defensive collapse)

Everything else:
- creative_tension

## 3) Risk EMA (memory of danger)
We map verdict to a coarse risk:
- collapse -> 0.95
- bubble -> 0.65
- creative_tension -> 0.25 + 0.50*tension

Then apply EMA:
risk_ema = beta*risk_ema + (1-beta)*instant_risk

beta close to 1 => slow memory
beta smaller => reactive controller

## 4) Damping alpha (risk -> action amplitude)
alpha is the multiplier in [min_alpha, max_alpha].
risk_ema 0 => alpha = max_alpha
risk_ema 1 => alpha = min_alpha

alpha = max_alpha - (max_alpha - min_alpha)*risk_ema

Then damp:
action_damped = action * alpha

## 5) Near-collapse survival switch
We trigger survival if:
- energy <= near_collapse_energy
OR
- coherence <= near_collapse_coherence AND pressure_response <= near_collapse_pressure

This switch returns a directive:
- discrete: prefer stabilizing actions, avoid risky exploration
- continuous: clamp magnitude, avoid aggressive updates

## 6) What is “correct” here?
Correctness is not “true philosophy”.
Correctness is:
- avoids catastrophic spirals
- reduces bubble/collapse frequency
- preserves creative tension when it actually survives pressure

