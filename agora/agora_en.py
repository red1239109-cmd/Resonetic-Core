# SPDX-License-Identifier: MIT
# Copyright (C) 2026 red1239109-cmd
# ==============================================================================
# File: agora_final_en.py
# Project: The Grand Philosophical Agora (Production Ready Edition - English)
# Version: 27.1 (Zombie Killer+ / Strict State / Clean Restart)
#
# [Patch v27.1]
# - Fix: nonlocal placement (was SyntaxError inside finally block)
# - Fix: update guard also checks simulation_task.done()
# - Upgrade: safe_send triggers stop_event + cancels simulation_task on socket failure
# - Upgrade: start/stop awaits cancelled task to avoid stray sends
# ==============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Union
import random, math, re
import asyncio
import json

# --- Server Imports ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# ============================================================
# [SECTION 1] The Engine
# ============================================================

WINDOW_SIZE = 6
WARMUP_CAST_PASSES = 1
RE_ROLL_THRESHOLD = 0.25
RHETORIC_THRESHOLD = 0.75

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def softmax(xs: List[float], temp: float = 1.0) -> List[float]:
    if not xs: return []
    m = max(xs)
    exps = [math.exp((x - m) / max(1e-9, temp)) for x in xs]
    z = sum(exps) or 1.0
    return [e / z for e in exps]

def cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys: return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(a.get(k, 0.0) ** 2 for k in keys)) or 1.0
    nb = math.sqrt(sum(b.get(k, 0.0) ** 2 for k in keys)) or 1.0
    return dot / (na * nb)

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-zA-Z]{2,}", text))

# --- Concept Graph (English) ---
CONCEPT_SYNONYMS = {
    "Truth": {"truth", "true", "fact", "justification", "knowledge", "verity"},
    "Reason": {"reason", "rationality", "logic", "inference", "deduction"},
    "Experience": {"experience", "observation", "case", "experiment", "empirical"},
    "Universal": {"universal", "necessity", "general", "norm", "absolute"},
    "Morality": {"morality", "duty", "imperative", "ethics", "dignity", "good"},
    "Freedom": {"freedom", "autonomy", "will", "liberty"},
    "Power": {"power", "force", "domination", "hierarchy", "strength"},
    "Value": {"value", "evaluation", "good_and_evil", "meaning", "worth"},
    "Language": {"language", "word", "expression", "grammar", "usage", "speech"},
    "Being": {"being", "reality", "substance", "ontology", "existence"},
    "Metaphysics": {"metaphysics", "transcendence", "essence", "idea"},
    "Society": {"society", "institution", "discipline", "politics", "public"},
    "History": {"history", "genealogy", "era", "progress", "epoch"},
    "Subject": {"subject", "self", "consciousness", "ego"},
    "Method": {"method", "critique", "analysis", "dialectic", "deconstruction"},
}

PHILO_GRAPHS = {
    "Plato": {"Metaphysics": {"Truth", "Universal", "Being"}, "Truth": {"Metaphysics", "Universal", "Reason"}, "Universal": {"Truth", "Metaphysics", "Reason"}},
    "Aristotle": {"Experience": {"Truth", "Being", "Method"}, "Being": {"Experience", "Truth", "Method"}, "Method": {"Experience", "Being", "Truth"}},
    "Kant": {"Reason": {"Universal", "Morality", "Method"}, "Universal": {"Reason", "Morality", "Freedom"}, "Freedom": {"Morality", "Universal"}, "Subject": {"Method", "Experience"}, "Method": {"Reason", "Universal"}},
    "Nietzsche": {"Power": {"Value", "Truth", "History"}, "Value": {"Power", "Truth"}, "Truth": {"Power", "Value"}, "History": {"Power", "Value"}},
    "Socrates": {"Method": {"Truth", "Subject", "Reason"}, "Subject": {"Method", "Truth"}, "Truth": {"Method", "Reason"}},
    "Descartes": {"Subject": {"Reason", "Truth"}, "Reason": {"Subject", "Truth"}, "Truth": {"Subject", "Reason"}},
    "Hume": {"Experience": {"Truth", "Method"}, "Method": {"Experience", "Truth"}, "Truth": {"Experience", "Method"}},
    "Spinoza": {"Being": {"Reason", "Universal", "Truth"}, "Reason": {"Being", "Truth"}, "Truth": {"Being", "Universal"}},
    "Leibniz": {"Reason": {"Universal", "Truth", "Metaphysics"}, "Metaphysics": {"Reason", "Truth"}, "Truth": {"Reason", "Universal"}},
    "Locke": {"Experience": {"Subject", "Truth", "Society"}, "Subject": {"Experience", "Truth"}, "Society": {"Experience", "Truth"}},
    "Rousseau": {"Society": {"Freedom", "Morality", "Value"}, "Freedom": {"Society", "Morality"}, "Morality": {"Society", "Freedom"}},
    "Mill": {"Value": {"Morality", "Society", "Truth"}, "Morality": {"Value", "Society"}, "Society": {"Value", "Morality"}},
    "Marx": {"Society": {"History", "Power", "Value"}, "History": {"Society", "Power"}, "Power": {"Society", "History"}},
    "Hegel": {"History": {"Method", "Truth", "Society"}, "Method": {"History", "Truth"}, "Truth": {"History", "Method"}},
    "Schopenhauer": {"Value": {"Being", "Subject"}, "Subject": {"Value", "Being"}, "Being": {"Subject", "Value"}},
    "Kierkegaard": {"Subject": {"Value", "Morality", "Truth"}, "Value": {"Subject", "Morality"}, "Truth": {"Subject", "Value"}},
    "Wittgenstein": {"Language": {"Method", "Truth"}, "Method": {"Language", "Truth"}, "Truth": {"Language", "Method"}},
    "Heidegger": {"Being": {"Subject", "Truth"}, "Subject": {"Being", "Truth"}, "Truth": {"Being", "Subject"}},
    "Sartre": {"Freedom": {"Subject", "Value", "Morality"}, "Subject": {"Freedom", "Value"}, "Value": {"Freedom", "Subject"}},
    "Foucault": {"Society": {"Power", "Truth", "History"}, "Power": {"Society", "Truth"}, "Truth": {"Society", "Power"}},
    "Arendt": {"Society": {"Truth", "History", "Value"}, "Value": {"Society", "Truth"}, "Truth": {"Society", "Value"}},
    "Popper": {"Method": {"Experience", "Truth"}, "Experience": {"Method", "Truth"}, "Truth": {"Method", "Experience"}},
    "Rawls": {"Morality": {"Society", "Freedom", "Universal"}, "Society": {"Morality", "Universal"}, "Universal": {"Morality", "Society"}},
}

def extract_concepts(text: str) -> Set[str]:
    out: Set[str] = set()
    for c, syns in CONCEPT_SYNONYMS.items():
        for s in syns:
            if s in text.lower():
                out.add(c)
                break
    return out

def pick_target_concept_from_other(other_claim: str, fallback: str = "Truth") -> str:
    cs = list(extract_concepts(other_claim))
    if cs: return random.choice(cs)
    return fallback

def concept_graph_score(philo: str, text: str) -> float:
    cset = extract_concepts(text)
    g = PHILO_GRAPHS.get(philo, {})
    if not cset: return 0.0
    
    if g:
        nodes = set(g.keys())
        coverage = (len(cset & nodes) / max(1, len(nodes)))
    else:
        coverage = min(1.0, len(cset) / 4)

    pairs = 0
    hits = 0
    cl = list(cset)
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            a, b = cl[i], cl[j]
            pairs += 1
            if g and (b in g.get(a, set()) or a in g.get(b, set())):
                hits += 1
            elif not g:
                hits += 0.5 

    coherence = hits / max(1, pairs)
    return clamp(0.55 * coverage + 0.45 * coherence, 0.0, 1.0)

@dataclass(frozen=True)
class InferenceTaboo:
    name: str; pattern: re.Pattern; penalty: float; explanation: str; repair_hint: str

def taboo_score(taboo_rules: Union[List[InferenceTaboo], InferenceTaboo, None], text: str) -> Tuple[float, List[InferenceTaboo]]:
    if taboo_rules is None: taboo_rules = []
    if isinstance(taboo_rules, InferenceTaboo): taboo_rules = [taboo_rules]
    s = 0.0
    hits: List[InferenceTaboo] = []
    for r in taboo_rules:
        if r.pattern.search(text):
            s += r.penalty
            hits.append(r)
    return clamp(s, 0.0, 1.0), hits

# English Regex Patterns for Taboos
TABOOS = {
    "kant_empirical_jump": InferenceTaboo("Empirical Jump", re.compile(r"(experience|observation|case).*(therefore|so).*(universal|must|norm|duty)", re.IGNORECASE), 0.65, "Empirical Leap", "Demand 'conditions of possibility' instead of experience."),
    "kant_ends_justify": InferenceTaboo("Ends Justify Means", re.compile(r"(end|goal|result).*(justify|justifies).*(means|method)", re.IGNORECASE), 0.75, "Dignity Clash", "Stop means-ends reasoning; insert 'dignity' constraint."),
    "nietzsche_universal_morals": InferenceTaboo("Universal Morality Assert", re.compile(r"(universal|absolute).*(moral|good|evil|law)", re.IGNORECASE), 0.70, "Moral Dogmatism", "Switch to genealogy: 'Who benefits?'"),
    "nietzsche_truth_worship": InferenceTaboo("Truth Worship", re.compile(r"(truth).*(sacred|absolute|highest|worship)", re.IGNORECASE), 0.55, "Truth Value Query", "Ask 'Why is truth good?'"),
    "hume_necessary_causation": InferenceTaboo("Necessary Causation", re.compile(r"(cause|causality).*(must|necessary|absolute)", re.IGNORECASE), 0.65, "Causality Skepticism", "Reduce explanation to 'habit/expectation'."),
    "witt_metaphysics_assert": InferenceTaboo("Metaphysical Assertion", re.compile(r"(essence|idea|transcendent).*(exists|real|certain)", re.IGNORECASE), 0.70, "Language Limit", "Switch to 'language games' or rules."),
    "foucault_truth_neutral": InferenceTaboo("Neutral Truth", re.compile(r"(truth).*(neutral|pure|objective)", re.IGNORECASE), 0.70, "Power-Knowledge", "Frame it as power/discipline/normalization."),
}

@dataclass
class Lexicon:
    core: List[str]; evidentials: List[str]; hedges: List[str]; intensifiers: List[str]; metaphors: List[str]; taboo_softeners: List[str]

@dataclass
class ReasoningOps:
    ops: List[str]

@dataclass
class StyleProfile:
    rhetoric_bias: float; justification_bias: float; interrogation_bias: float; poetic_bias: float

@dataclass
class Philosopher:
    name: str; era: str; truth_vector: Dict[str, float]; lexicon: Lexicon; reasoning: ReasoningOps; taboo: List[InferenceTaboo]; style: StyleProfile
    phase: Dict[str, float] = field(default_factory=lambda: {"open": 0.5, "attack": 0.5, "synthesize": 0.5})

@dataclass
class ArgGraph:
    claim: str = ""; warrant: str = ""; attack: str = ""; constraint: str = ""; synthesis: str = ""

def pick(xs: List[str]) -> str: return random.choice(xs) if xs else ""
def tension_to_register(t: float) -> str: return "low" if t < 0.33 else "mid" if t < 0.66 else "high"

# --- Ops (English) ---
def op_define_split(p, topic, other, g, t, target=""):
    tgt = target if target else topic; w = pick(p.lexicon.evidentials) or "First,"
    g.claim = f"{w} when we speak of '{tgt}', we confuse 'fact' with 'justification'."
    g.warrant = "If definitions are blurred, debate becomes a slippery slope."

def op_condition_censor(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"For the judgment of '{tgt}' to be possible at all, what are its preconditions?"
    g.warrant = "Experience alone cannot guarantee universality."; g.constraint = "Therefore, we must establish the conditions of possibility first."

def op_empirical_classify(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"Let us classify the cases called '{tgt}': scientific verification, legal proof, and daily trust."
    g.warrant = "Lumping distinct mechanisms together collapses the explanation."; g.constraint = "Taxonomy and causal analysis must come first."

def op_reductio(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.attack = f"Very well. If we assume '{tgt}' is absolute, the distinction between true and false itself collapses."
    g.warrant = "When the distinction falls, the argument loses its footing."

def op_genealogy_expose(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = random.choice([
        f"'{tgt}' appears with an innocent face, but historically it was a tool in someone's hand.",
        f"The moment the word '{tgt}' appears, a certain standard has already won.",
        f"The very way of speaking about '{tgt}' reveals the arrangement of power.",
    ])
    g.attack = random.choice([
        "Who monopolized the right to speak it?",
        "What did that word normalize, and what did it exclude?",
        "Who is positioned as the 'strong' and who as the 'weak' by this discourse?",
    ])
    g.warrant = "If you trace the genealogy, 'Truth' often smells of value and power."

def op_value_invert(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = random.choice([
        f"Why does the premise that '{tgt}' is good pass automatically?",
        f"What are those who worship '{tgt}' actually afraid of?",
        f"Is there evidence that '{tgt}' strengthens life—or does it paralyze it?",
    ])
    g.warrant = "Unless we invert the hierarchy of values, 'Truth' remains an idol."; g.attack = "I choose to shatter the idol."

def op_language_therapy(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"The confusion here might stem from the grammar of the word '{tgt}'."
    g.warrant = "Clarify the usage, and half the problem vanishes."; g.constraint = "Look at the usage rules instead of metaphysical assertions."

def op_power_knowledge(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = "Truth takes the form of knowledge, but knowledge is linked to institutions and discipline."
    g.attack = f"If you don't see what the discourse of '{tgt}' normalizes and excludes, you miss the point."; g.warrant = "The power-knowledge apparatus creates the conditions for 'truth'."

def op_public_world(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"'{tgt}' does not exist solely in the head. It is revealed in the public world through speech and action."
    g.warrant = "Truth is a matter of world-sharing and responsibility."; g.synthesis = "Thus, the debate on truth cannot escape the political and ethical dimensions."

def op_elenchus(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"You say you know '{tgt}'. Then tell me the single minimum condition that constitutes '{tgt}'."
    g.attack = "If that condition shakes, your claim of 'knowing' also shakes."; g.warrant = "If only conviction remains without definition, we are deceived by words."

def op_methodic_doubt(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f" regarding '{tgt}', I will doubt everything that can be doubted."
    g.warrant = "Only that which survives doubt earns the title of certainty."; g.constraint = "So, present a foundation that withstands doubt first."

def op_hume_skeptic(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"When speaking of '{tgt}', we often mistake the expectation from repetition for 'necessity'."
    g.warrant = "Causality, necessity, and certainty are often other names for habit."; g.attack = "Your certainty might be imagination filling the gaps of experience."

def op_dialectic(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"'{tgt}' is not a fixed point, but a process unfolding through conflict."
    g.warrant = "Clinging to one side accumulates contradiction, which opens the next phase."; g.synthesis = "Do not remove the opposite; lift it up to a higher unity (Aufheben)."

def op_pessimism_will(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"'{tgt}' is not the victory of reason, but the Will packaging its own suffering."
    g.warrant = "Life comes first; reason follows to glorify it."; g.attack = "Is your truth a medicine for pain, or a fog hiding reality?"

def op_subjective_truth(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"Let us suspect the belief that '{tgt}' exists like an objective document."
    g.warrant = "What matters is how I am 'entangled' in that truth."; g.attack = "Does your truth change your life, or is it a tool to judge others?"

def op_being_unconceal(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"Do not see '{tgt}' merely as a correct proposition. Truth is unconcealment (Aletheia)."
    g.warrant = "Look at the structure of what is hidden and what is revealed."; g.constraint = "First, ask how 'Being' opens up."

def op_existential_choice(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"To seek '{tgt}' is ultimately a matter of choice."
    g.warrant = "You are not free not to choose. Even silence is a choice."; g.attack = "Does your truth make you take responsibility, or does it help you flee from it?"

def op_marx_ideology(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"'{tgt}' does not fall from the stars. It is produced at the base of production and institutions."
    g.warrant = "Dominant truths tend to justify dominant relations."; g.attack = "Look at who benefits from that truth first."

def op_utilitarian(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"When discussing '{tgt}', we cannot exclude the calculation of consequences."
    g.warrant = "Rules gain persuasion when linked to the aggregate of happiness and suffering."; g.constraint = "Therefore, look at what actual harm or benefit a statement produces."

def op_falsification(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"If you assert '{tgt}', you must first state under what conditions that assertion can be refuted."
    g.warrant = "Certainty without falsifiability is belief, not knowledge."; g.attack = "Does your truth take risks, or is it built to escape?"

def op_original_position(p, topic, other, g, t, target=""):
    tgt = target if target else topic; g.claim = f"If we establish the rules of '{tgt}', would you choose them without knowing who you are in that system?"
    g.warrant = "Fairness is tested behind a veil of ignorance."; g.constraint = "Thus, establish conditions that everyone can accept."

OPS = {
    "define_split": op_define_split, "condition_censor": op_condition_censor, "empirical_classify": op_empirical_classify, "reductio": op_reductio,
    "genealogy_expose": op_genealogy_expose, "value_invert": op_value_invert, "language_therapy": op_language_therapy, "power_knowledge": op_power_knowledge,
    "public_world": op_public_world, "elenchus": op_elenchus, "methodic_doubt": op_methodic_doubt, "hume_skeptic": op_hume_skeptic,
    "dialectic": op_dialectic, "pessimism_will": op_pessimism_will, "subjective_truth": op_subjective_truth, "being_unconceal": op_being_unconceal,
    "existential_choice": op_existential_choice, "marx_ideology": op_marx_ideology, "utilitarian": op_utilitarian, "falsification": op_falsification, "original_position": op_original_position,
}

# --- Helpers ---
def apply_taboo_repairs(p: Philosopher, text: str, hits: List[InferenceTaboo], topic: str) -> str:
    if not hits: return text
    repairs = []
    for h in hits[:2]:
        if "Empirical Jump" in h.name: repairs.append("The key is the 'condition of possibility'.")
        elif "Ends Justify" in h.name: repairs.append("No end can violate dignity.")
        elif "Genealogy" in h.name: repairs.append("Who calls it good and gains power?")
        elif "Truth Value" in h.name: repairs.append("Let us ask again: 'Why is truth good?'")
        elif "Causality" in h.name: repairs.append("'Necessity' might just be a habit.")
        elif "Language" in h.name: repairs.append("That assertion might be a misuse of language.")
        elif "Power-Knowledge" in h.name: repairs.append("Truth is entangled with the apparatus of power.")
    soft = pick(p.lexicon.taboo_softeners)
    add = (" " + soft + " " if soft else " ") + " ".join(repairs)
    return (text + add).strip()

def ensure_concept_anchor(p: Philosopher, text: str) -> str:
    cg = concept_graph_score(p.name, text)
    if cg >= 0.24: return text
    anchors = {
        "Kant": " For universal validity, we must demand conditions of possibility.",
        "Nietzsche": " In the end, that 'Truth' is a mask for values!",
        "Plato": " Without an unchanging standard, truth scatters.",
        "Aristotle": " We must establish words through cases and causal analysis.",
        "Socrates": " Do you truly *know* that?",
        "Descartes": " I think, therefore I am.",
        "Marx": " The point is not to interpret the world, but to change it.",
        "Wittgenstein": " Whereof one cannot speak, thereof one must remain silent.",
    }
    return (text + anchors.get(p.name, "")).strip()

def build_arggraph(p: Philosopher, topic: str, other_claim: str, tension: float, mode: str, entropy_boost: float = 0.0) -> ArgGraph:
    g = ArgGraph()
    raw_ops = p.reasoning.ops
    unique_ops = list(dict.fromkeys(raw_ops))
    
    atk_bias = 0.9 + 0.8 * p.phase.get("attack", 0.5)
    syn_bias = 0.9 + 0.8 * p.phase.get("synthesize", 0.5)
    open_bias = 0.9 + 0.6 * p.phase.get("open", 0.5)
    reg = tension_to_register(tension)
    weights = []
    
    attack_ops = {"reductio", "genealogy_expose", "power_knowledge", "value_invert", "hume_skeptic", "pessimism_will", "marx_ideology", "elenchus"}
    syn_ops = {"public_world", "define_split", "dialectic", "utilitarian"}
    open_ops = {"define_split", "empirical_classify", "condition_censor", "language_therapy", "methodic_doubt", "being_unconceal", "original_position", "falsification"}

    for opn in unique_ops:
        w = 1.0
        if mode == "attack" and opn in attack_ops: w *= atk_bias
        if mode == "synthesize" and opn in syn_ops: w *= syn_bias
        if mode == "open" and opn in open_ops: w *= open_bias
        if reg == "high": w *= (1.0 + 0.6 * p.style.rhetoric_bias)
        elif reg == "low": w *= (1.0 + 0.6 * p.style.justification_bias)
        weights.append(w)

    probs = softmax(weights, temp=0.9 + entropy_boost)
    k = 3 if reg != "low" else 2
    if entropy_boost > 0.0 and len(unique_ops) >= 4: k += 1
        
    chosen = set(); attempts = 0; target_k = min(k, len(unique_ops))
    while len(chosen) < target_k and attempts < 20:
        attempts += 1; r = random.random(); cum = 0.0
        for opn, pr in zip(unique_ops, probs):
            cum += pr
            if r <= cum:
                chosen.add(opn); break
    if len(chosen) < target_k:
        remaining = [op for op in unique_ops if op not in chosen]
        chosen.update(random.sample(remaining, target_k - len(chosen)))

    ordered_chosen = [op for op in unique_ops if op in chosen]
    target = pick_target_concept_from_other(other_claim, fallback=topic)
    for opn in ordered_chosen: OPS[opn](p, topic, other_claim, g, tension, target=target)
    return g

def linearize(p: Philosopher, g: ArgGraph, tension: float) -> str:
    reg = tension_to_register(tension)
    if reg == "low": lead = pick(p.lexicon.hedges) or "Perhaps,"
    elif reg == "mid": lead = "However,"
    else: lead = pick(p.lexicon.intensifiers) or "Decisively,"
    parts = []
    if p.style.interrogation_bias > 0.65: parts.append(f"{lead} let me ask first.")
    if reg == "high" and p.style.rhetoric_bias >= p.style.justification_bias:
        if g.attack: parts.append(g.attack)
        if g.claim: parts.append(g.claim)
        if g.warrant: parts.append(g.warrant)
        if g.constraint: parts.append(g.constraint)
        if g.synthesis: parts.append(g.synthesis)
    else:
        if g.claim: parts.append(f"{lead} {g.claim}")
        if g.warrant: parts.append(g.warrant)
        if g.constraint: parts.append(g.constraint)
        if g.attack: parts.append(g.attack)
        if g.synthesis: parts.append(g.synthesis)
    if p.style.poetic_bias > 0.6 and p.lexicon.metaphors: parts.insert(0, pick(p.lexicon.metaphors))
    text = " ".join([s.strip() for s in parts if s and s.strip()])
    if random.random() < 0.25:
        core_word = pick(p.lexicon.core)
        if core_word: text = f"({core_word}) {text}"
    if p.name == "Nietzsche": text = text.replace(" is ", " IS ").replace(" do ", " DO ") # Stylistic emphasis
    if p.name == "Nietzsche" and reg == "high": text = text.replace(".", "!")
    return text.strip()

@dataclass
class Verdict:
    status: str; note: str; tension: float; coherence: float; novelty: float; is_warmup: bool = False

class Arbiter:
    def __init__(self, cast_size: int):
        self.t_hist = []; self.c_hist = []; self.n_hist = []; self.last = []; self.p_last = {}
        self.warmup_turns = cast_size * WARMUP_CAST_PASSES
        self.window_size = WINDOW_SIZE
        self.turn_count = 0; self.warmup_done = False; self.calibration_snapshot = (0.0, 0.0, 0.0)

    def novelty_check(self, text: str, speaker: str) -> float:
        toks = tokenize(text); glob_n = 1.0
        if self.last:
            window = self.last[-self.window_size:]
            max_sim = max([jaccard(toks, tokenize(prev)) for prev in window], default=0.0)
            glob_n = clamp(1.0 - max_sim, 0.0, 1.0)
        pers_n = 1.0
        if speaker in self.p_last:
            prev_p = tokenize(self.p_last[speaker])
            pers_n = clamp(1.0 - jaccard(toks, prev_p), 0.0, 1.0)
        return (glob_n * 0.4) + (pers_n * 0.6)

    def judge(self, text: str, speaker: str) -> Verdict:
        # Markers adjusted for English context
        markers = ["collapse", "expose", "censor", "exclude", "leap", "hypocrisy", "dominate", "mask", "idol", "monopoly", "struggle"]
        t_raw = sum(1 for m in markers if m in text.lower()) + min(2, text.count("!") // 2)
        t = clamp(t_raw / 6.0, 0.0, 1.0)
        c_density = clamp(len(extract_concepts(text)) / 6.0, 0.0, 1.0); c = clamp(c_density, 0.0, 1.0)
        n = self.novelty_check(text, speaker)

        self.turn_count += 1
        self.t_hist.append(t); self.c_hist.append(c); self.n_hist.append(n)
        self.last.append(text); self.p_last[speaker] = text

        if not self.warmup_done:
            if self.turn_count < self.warmup_turns: return Verdict("CONTINUE", "(Warm-up)", t, c, n, is_warmup=True)
            elif self.turn_count == self.warmup_turns:
                self.calibration_snapshot = (t, c, n)
                self.warmup_done = True
                self.t_hist.clear(); self.c_hist.clear(); self.n_hist.clear(); self.last.clear()
                self.p_last.clear()
                return Verdict("RESET", f"(Calibration Complete) [Snap: T{t:.2f}/C{c:.2f}/N{n:.2f}]", t, c, n, is_warmup=True)

        if len(self.t_hist) >= 5:
            if (sum(1 for x in self.t_hist[-5:] if x > 0.80) >= 4) and (sum(1 for y in self.c_hist[-5:] if y < 0.28) >= 3): return Verdict("MELTDOWN", "Meltdown Imminent.", t, c, n)
        if len(self.n_hist) >= 6:
            if sum(1 for x in self.n_hist[-6:] if x < 0.25) >= 5: return Verdict("DEADLOCK", "Stagnation.", t, c, n)
        if len(self.t_hist) >= 6:
            th = self.t_hist; ch = self.c_hist
            if th[-6] > 0.70 and th[-1] < 0.35 and ch[-1] > 0.45: return Verdict("CONSENSUS", "Resonance Achieved.", t, c, n)
        return Verdict("CONTINUE", "", t, c, n)

    def adapt(self, philos: List[Philosopher], verdict: Verdict):
        if verdict.status == "RESET":
            (t, c, n) = self.calibration_snapshot
            for p in philos: p.phase["open"] = clamp(p.phase["open"] + 0.1 * n, 0.0, 1.0)
            return
        if verdict.is_warmup: return
        for p in philos:
            if verdict.status == "MELTDOWN":
                p.phase["attack"] = clamp(p.phase["attack"] - 0.15, 0.0, 1.0); p.phase["open"] = clamp(p.phase["open"] + 0.10, 0.0, 1.0); p.phase["synthesize"] = clamp(p.phase["synthesize"] + 0.10, 0.0, 1.0)
            elif verdict.status == "DEADLOCK":
                p.phase["open"] = clamp(p.phase["open"] + 0.15, 0.0, 1.0); p.phase["attack"] = clamp(p.phase["attack"] + 0.05, 0.0, 1.0)
            elif verdict.status == "CONSENSUS":
                p.phase["synthesize"] = clamp(p.phase["synthesize"] + 0.15, 0.0, 1.0); p.phase["attack"] = clamp(p.phase["attack"] - 0.10, 0.0, 1.0)
            else:
                for k in p.phase: p.phase[k] = clamp(p.phase[k] * 0.98 + 0.01, 0.0, 1.0)

def build_user_philosopher(user_claim: str) -> Philosopher:
    cs = extract_concepts(user_claim or ""); 
    if not cs: cs = {"Truth"}
    vec = {c: 1.0 for c in cs}
    soft_mix = ["However,", "But,", "I think,", "The point is,", "To summarize,"]
    lex = Lexicon(core=["My Claim", "Intuition", "Case"], evidentials=["In my view,", "From experience,", "If we think about it,"], hedges=["Perhaps", "Possibly", "Initially"], intensifiers=["Certainly", "Clearly", "Strongly"], metaphors=["Words are maps, the world is terrain."], taboo_softeners=soft_mix)
    style = StyleProfile(rhetoric_bias=0.55, justification_bias=0.65, interrogation_bias=0.60, poetic_bias=0.20)
    ops = ["define_split", "empirical_classify", "falsification", "elenchus"]
    return Philosopher(name="User", era="Modern", truth_vector=vec, lexicon=lex, reasoning=ReasoningOps(ops), taboo=[], style=style)

def build_grand_cast() -> List[Philosopher]:
    def L(core, evid, hed, inten, meta, soft): return Lexicon(core, evid, hed, inten, meta, soft)
    def S(rhet, just, inter, poet): return StyleProfile(rhet, just, inter, poet)
    def P(name, era, vec, lex, ops, taboo, style): return Philosopher(name, era, vec, lex, ReasoningOps(ops), taboo, style)
    soft_mix = ["However,", "But,", "Simply,", "Even so,", "Of course,", "Admittedly,"]
    return [
        P("Plato","Ancient",{"Metaphysics":0.90,"Universal":0.92,"Reason":0.80}, L(["Idea","Essence"],["First,"],["Perhaps"],["Decisively"],["We cannot speak truth looking at shadows."],soft_mix), ["define_split","reductio","condition_censor"],[],S(0.55,0.65,0.55,0.35)),
        P("Aristotle","Ancient",{"Experience":0.88,"Reason":0.75,"Method":0.80}, L(["Cause","Taxonomy"],["To observe,"],["Generally"],["Precisely"],["We must place the object on the dissection table."],soft_mix), ["empirical_classify","define_split"],[],S(0.35,0.85,0.35,0.10)),
        P("Kant","Modern",{"Reason":0.90,"Universal":0.88,"Morality":0.82}, L(["Condition of Possibility","Universal Validity"],["If we examine,"],["First,"],["Never"],["Judgment without rules is sailing without a compass."],soft_mix), ["condition_censor","reductio"], [TABOOS["kant_empirical_jump"]], S(0.30,0.95,0.75,0.05)),
        P("Nietzsche","Modern",{"Power":0.90,"Value":0.85,"Truth":0.55}, L(["Hammer","Idol","Power"],["Behold"],["Sometimes"],["Mercilessly"],["Truth is the comfort of the weak and the tool of the strong."],soft_mix), ["genealogy_expose","value_invert","reductio"], [TABOOS["nietzsche_universal_morals"]], S(0.90,0.45,0.55,0.75)),
        P("Socrates","Ancient",{"Method":0.95,"Subject":0.75}, L(["Questioning","Ignorance"],["Then,"],["Maybe"],["Clearly"],["I know that I know nothing."],soft_mix), ["elenchus","define_split"],[],S(0.45,0.60,0.95,0.05)),
        P("Descartes","Modern",{"Subject":0.92,"Reason":0.85}, L(["Clear & Distinct","Doubt"],["Decisively"],["First"],["Certainly"],["Cast away everything that shakes."],soft_mix), ["methodic_doubt","define_split"],[],S(0.40,0.85,0.55,0.10)),
        P("Hume","Modern",{"Experience":0.92,"Method":0.80}, L(["Habit","Impression"],["In experience,"],["Usually"],["Ultimately"],["We learn belief before logic."],soft_mix), ["hume_skeptic","empirical_classify"], [TABOOS["hume_necessary_causation"]], S(0.25,0.80,0.55,0.10)),
        P("Spinoza","Modern",{"Being":0.90,"Reason":0.85}, L(["Necessity","Nature"],["Necessarily"],["Precisely"],["Must"],["Nature is not caprice, but order."],soft_mix), ["define_split","reductio"],[],S(0.35,0.90,0.45,0.10)),
        P("Marx","Modern",{"Society":0.92,"History":0.85,"Power":0.85}, L(["Class","Production"],["Realistically"],["Usually"],["Decisively"],["The point is not to interpret, but to change."],soft_mix), ["marx_ideology","power_knowledge"],[],S(0.70,0.55,0.55,0.20)),
        P("Wittgenstein","Modern",{"Language":0.95,"Method":0.85}, L(["Language Game","Rule"],["To look at,"],["Perhaps"],["Precisely"],["Whereof one cannot speak, thereof one must remain silent."],soft_mix), ["language_therapy","define_split"], [TABOOS["witt_metaphysics_assert"]], S(0.25,0.85,0.55,0.10)),
        P("Hegel","Modern",{"History":0.92,"Method":0.85,"Truth":0.75}, L(["Dialectic","Unfolding"],["If we follow,"],["First"],["Ultimately"],["Truth is a process."],soft_mix), ["dialectic","define_split"],[],S(0.55,0.65,0.45,0.15)),
        P("Schopenhauer","Modern",{"Subject":0.75,"Being":0.70,"Value":0.65}, L(["Suffering","Will"],["Honestly"],["Usually"],["Decisively"],["Life is a pendulum of suffering."],soft_mix), ["pessimism_will","define_split"],[],S(0.55,0.55,0.55,0.25)),
        P("Kierkegaard","Modern",{"Subject":0.90,"Value":0.80}, L(["Existence","Decision"],["First"],["Maybe"],["Ultimately"],["Truth is subjectivity."],soft_mix), ["subjective_truth","existential_choice"],[],S(0.60,0.55,0.70,0.35)),
        P("Heidegger","Modern",{"Being":0.92,"Subject":0.75}, L(["Being","Clearing"],["First"],["Maybe"],["Ultimately"],["Truth is the struggle of unconcealment."],soft_mix), ["being_unconceal","define_split"],[],S(0.55,0.60,0.55,0.25)),
        P("Sartre","Modern",{"Freedom":0.92,"Subject":0.88}, L(["Freedom","Responsibility"],["Decisively"],["Sometimes"],["Ultimately"],["Man is condemned to be free."],soft_mix), ["existential_choice","define_split"],[],S(0.70,0.55,0.55,0.30)),
        P("Foucault","Modern",{"Society":0.90,"Power":0.90}, L(["Discipline","Apparatus"],["If we trace,"],["Usually"],["Clearly"],["Truth is produced within the apparatus."],soft_mix), ["power_knowledge","genealogy_expose"], [TABOOS["foucault_truth_neutral"]], S(0.70,0.55,0.55,0.20)),
        P("Arendt","Modern",{"Society":0.90,"Value":0.78}, L(["Public World","Action"],["To see,"],["Maybe"],["Clearly"],["Truth is carrying the world together."],soft_mix), ["public_world","define_split"],[],S(0.45,0.70,0.55,0.15)),
        P("Popper","Modern",{"Method":0.92,"Experience":0.80}, L(["Falsification","Hypothesis"],["First"],["First of all"],["Clearly"],["Knowledge grows through refutation."],soft_mix), ["falsification","empirical_classify"],[],S(0.35,0.80,0.55,0.05)),
        P("Rawls","Modern",{"Morality":0.90,"Society":0.85}, L(["Fairness","Original Position"],["If we assume,"],["First"],["Clearly"],["Rules must be endured from the position of the weak."],soft_mix), ["original_position","condition_censor"],[],S(0.35,0.85,0.55,0.05)),
        P("Leibniz","Modern",{"Reason":0.88,"Universal":0.85}, L(["Sufficient Reason","Harmony"],["If we calculate,"],["Perhaps"],["Necessarily"],["Why is there something rather than nothing?"],soft_mix), ["condition_censor","define_split"],[],S(0.35,0.85,0.55,0.20)),
    ]

class Agora:
    def __init__(self, cast, seed=None):
        self.philos = cast
        self.seed = seed
        self.arb = Arbiter(len(cast))
        self.positions: Dict[str, str] = {}

    def _pick_other(self, i, r):
        n = len(self.philos)
        if n <= 1:
            return self.philos[i]
        j = (i + r) % n
        if j == i:
            j = (i + r + 1) % n
        return self.philos[j]

    async def run_async_generator(self, topic: str, rounds: int = 5, state_lock: asyncio.Lock = None, stop_event: asyncio.Event = None):
        yield f"🏛️ Grand Agora v27.1 (Production Ready - English) | Topic: {topic}"
        yield f"Participants ({len(self.philos)}): {', '.join(p.name for p in self.philos[:5])}..."
        yield "-" * 50
        yield "\n[🎤 Opening Statements & Calibration in progress...]"

        base_tension = 0.28

        if state_lock:
            async with state_lock:
                if "User" not in self.positions:
                    self.positions["User"] = f"Regarding '{topic}', I believe we need at least refutability (verification/falsification) to approach truth."
        else:
            if "User" not in self.positions:
                self.positions["User"] = f"Regarding '{topic}', I believe we need at least refutability (verification/falsification) to approach truth."

        opening_buffer: Dict[str, str] = {}
        for p in self.philos:
            if stop_event and stop_event.is_set():
                return

            if p.name == "User":
                if state_lock:
                    async with state_lock:
                        s = self.positions["User"]
                else:
                    s = self.positions["User"]
                opening_buffer[p.name] = s
                self.arb.p_last[p.name] = s
                yield f"[Opening] User: {s[:60]}..."
            else:
                g = build_arggraph(p, topic, "", base_tension, "open")
                s = linearize(p, g, base_tension)
                s = ensure_concept_anchor(p, s)
                opening_buffer[p.name] = s
                self.arb.p_last[p.name] = s
                if random.random() < 0.25:
                    yield f"[Opening] {p.name}: {s[:60]}..."
            await asyncio.sleep(0.01)

        if state_lock:
            async with state_lock:
                self.positions.update(opening_buffer)
        else:
            self.positions.update(opening_buffer)

        for r in range(1, rounds + 1):
            if stop_event and stop_event.is_set():
                return
            yield f"\n🌀 Round {r}"
            yield "-" * 50

            for i, me in enumerate(self.philos):
                if stop_event and stop_event.is_set():
                    return

                if me.name == "User":
                    if state_lock:
                        async with state_lock:
                            s = self.positions.get("User", "").strip()
                    else:
                        s = self.positions.get("User", "").strip()

                    if not s:
                        yield "\n🗣️ User: (Passes turn)"
                        continue

                    verdict = self.arb.judge(s, me.name)
                    hud = f"   📊 [Mode:USER->USER   | Fric:0.00] (T:{verdict.tension:.2f} C:{verdict.coherence:.2f} N:{verdict.novelty:.2f})"
                    yield f"\n🗣️ User:\n\"{s}\""
                    yield hud
                    self.arb.adapt(self.philos, verdict)
                    await asyncio.sleep(0.1)
                    continue

                other = self._pick_other(i, r)

                if state_lock:
                    async with state_lock:
                        other_claim = self.positions.get(other.name, "")
                else:
                    other_claim = self.positions.get(other.name, "")

                dist = 1.0 - cosine_sim(me.truth_vector, other.truth_vector)

                if not other_claim.strip():
                    friction = clamp(0.55 * dist + 0.05, 0.0, 1.0)
                else:
                    other_cg = concept_graph_score(me.name, other_claim)
                    taboo_s, _ = taboo_score(me.taboo, other_claim)
                    friction = clamp(0.55 * dist + 0.25 * taboo_s + 0.20 * (1.0 - other_cg), 0.0, 1.0)

                if friction > 0.60:
                    mode = "attack"
                elif friction > 0.35:
                    mode = "open"
                else:
                    mode = "synthesize"

                tension = clamp(0.30 + 0.70 * friction, 0.0, 1.0)
                g = build_arggraph(me, topic, other_claim, tension, mode)
                s = linearize(me, g, tension)

                ts, hits = taboo_score(me.taboo, s)
                if ts > 0.55:
                    s = apply_taboo_repairs(me, s, hits, topic)

                s = ensure_concept_anchor(me, s)

                n0 = self.arb.novelty_check(s, me.name)
                re_roll_msg = ""
                if n0 < RE_ROLL_THRESHOLD and me.style.rhetoric_bias > RHETORIC_THRESHOLD:
                    cands = {"attack", "open", "synthesize"} - {mode}
                    new_mode = random.choice(list(cands))
                    mode_display = f"{mode.upper()}->{new_mode.upper()}"
                    mode = new_mode
                    tension_boost = clamp(tension + 0.25, 0.0, 1.0)
                    g2 = build_arggraph(me, topic, other_claim, tension_boost, mode, entropy_boost=0.3)
                    s2 = linearize(me, g2, tension_boost)
                    ts2, hits2 = taboo_score(me.taboo, s2)
                    if ts2 > 0.55:
                        s2 = apply_taboo_repairs(me, s2, hits2, topic)
                    s = ensure_concept_anchor(me, s2)
                    re_roll_msg = f"   🔄 (Re-roll via {mode_display})"
                else:
                    mode_display = f"{mode.upper()}->{mode.upper()}"

                verdict = self.arb.judge(s, me.name)
                note_str = f" {verdict.note}" if verdict.note else ""
                hud = f"   📊 [Mode:{mode_display:<15} | Fric:{friction:.2f}] (T:{verdict.tension:.2f} C:{verdict.coherence:.2f} N:{verdict.novelty:.2f}){note_str}"

                if re_roll_msg:
                    yield re_roll_msg
                yield f"\n🗣️ {me.name} → {other.name}:\n\"{s}\""
                yield f"{hud} => {verdict.status}" if verdict.status != "CONTINUE" else hud

                self.arb.adapt(self.philos, verdict)

                if state_lock:
                    async with state_lock:
                        self.positions[me.name] = s
                else:
                    self.positions[me.name] = s

                if verdict.status == "MELTDOWN":
                    yield "\n🛑 [SYSTEM] Entropy threshold exceeded. Debate halted due to MELTDOWN."
                    return
                if verdict.status == "CONSENSUS":
                    yield "\n✅ [SYSTEM] Resonance achieved. CONSENSUS reached."
                    return

                await asyncio.sleep(0.1)

# ============================================================
# [SECTION 2] The Server & UI
# ============================================================

app = FastAPI()

html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agora Interactive (EN)</title>
    <style>
        body { font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; display: flex; flex-direction: column; height: 95vh; }
        h1 { color: #58a6ff; text-align: center; margin-bottom: 10px; font-size: 1.5rem; }
        #chat-container { flex: 1; border: 1px solid #30363d; border-radius: 6px; padding: 15px; overflow-y: auto; background: #161b22; margin-bottom: 15px; box-shadow: inset 0 0 10px #000; }
        .message { margin-bottom: 8px; line-height: 1.4; border-bottom: 1px solid #21262d; padding-bottom: 4px; white-space: pre-wrap; }
        .hud { color: #8b949e; font-size: 0.85em; }
        .system { color: #f0883e; font-weight: bold; }
        .speaker { color: #79c0ff; font-weight: bold; }
        .reroll { color: #d2a8ff; font-style: italic; }
        #input-area { display: flex; gap: 10px; flex-direction: column; }
        .input-row { display: flex; gap: 10px; }
        input { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #30363d; background: #0d1117; color: #c9d1d9; font-family: inherit; }
        button { padding: 10px 20px; border-radius: 6px; border: none; background: #238636; color: white; cursor: pointer; font-weight: bold; font-family: inherit; }
        button#stopBtn { background: #da3633; } 
        button:disabled { background: #484f58; cursor: not-allowed; }
        button:hover:not(:disabled) { opacity: 0.9; }
        input:focus { outline: 2px solid #58a6ff; }
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: #0d1117; }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 5px; }
        ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
    </style>
</head>
<body>
    <h1>🏛️ Grand Philosophical Agora (English Edition)</h1>
    <div id="chat-container">
        <div class="message system">SYSTEM: Connected. Enter a topic and your initial stance to begin.</div>
    </div>
    <div id="input-area">
        <div class="input-row">
            <input type="text" id="topicInput" placeholder="Topic: (e.g., What is Justice?)" />
        </div>
        <div class="input-row">
            <input type="text" id="userClaimInput" placeholder="My Stance / Intervention (Press Enter)" />
            <button id="startBtn" onclick="sendAction('start')">Start Debate</button>
            <button id="stopBtn" onclick="sendAction('stop')" disabled>Stop</button>
        </div>
    </div>

    <script>
        let ws;
        const chat = document.getElementById("chat-container");
        const topicInput = document.getElementById("topicInput");
        const claimInput = document.getElementById("userClaimInput");
        const startBtn = document.getElementById("startBtn");
        const stopBtn = document.getElementById("stopBtn");

        function connectWS() {
            const scheme = (location.protocol === "https:") ? "wss://" : "ws://";
            ws = new WebSocket(scheme + window.location.host + "/ws");

            ws.onopen = function() {
                const msg = document.createElement("div");
                msg.className = "message system";
                msg.innerText = "SYSTEM: Connected to server.";
                chat.appendChild(msg);

                // Reset UI State on Reconnect
                startBtn.disabled = false;
                stopBtn.disabled = true;
                startBtn.innerText = "Start Debate";
            };

            ws.onmessage = function(event) {
                const msg = document.createElement("div");
                msg.className = "message";
                let text = event.data;

                if (text.includes("📊")) msg.className += " hud";
                else if (text.includes("SYSTEM") || text.includes("[Opening]")) msg.className += " system";
                else if (text.includes("🔄")) msg.className += " reroll";
                else if (text.includes("🗣️")) msg.className += " speaker";

                msg.innerText = text;
                chat.appendChild(msg);
                chat.scrollTop = chat.scrollHeight;

                // Stop triggers (English keywords)
                if (text.includes("Debate halted") || text.includes("CONSENSUS reached") || text.includes("Debate terminated") || text.includes("Debate ended")) {
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    startBtn.innerText = "New Debate";
                }
            };

            ws.onclose = function() {
                const msg = document.createElement("div");
                msg.className = "message system";
                msg.innerText = "SYSTEM: Connection lost. Reconnecting...";
                chat.appendChild(msg);
                setTimeout(connectWS, 1000);
            };
        }

        connectWS();

        function sendAction(type) {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;

            if (type === 'start') {
                const topic = topicInput.value.trim();
                const userClaim = claimInput.value.trim();
                if (!topic) { alert("Please enter a topic."); return; }
                
                startBtn.disabled = true;
                stopBtn.disabled = false;
                startBtn.innerText = "Running...";
                chat.innerHTML = ""; 
                ws.send(JSON.stringify({ type: 'start', topic: topic, user_claim: userClaim }));
                claimInput.value = "";
                claimInput.placeholder = "You can intervene at any time...";
            } 
            else if (type === 'update') {
                if (!startBtn.disabled) return; 
                const userClaim = claimInput.value.trim();
                if (!userClaim) return;
                ws.send(JSON.stringify({ type: 'update', text: userClaim }));
                claimInput.value = "";
            }
            else if (type === 'stop') {
                ws.send(JSON.stringify({ type: 'stop' }));
            }
        }

        // IME-Safe Key Binding
        claimInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.isComposing) {
                e.preventDefault();
                if (!startBtn.disabled) sendAction('start'); 
                else sendAction('update');
            }
        });
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    agora_instance: Optional[Agora] = None
    simulation_task: Optional[asyncio.Task] = None
    send_lock = asyncio.Lock()
    state_lock = asyncio.Lock()
    stop_event = asyncio.Event() 
    server_state = "IDLE" 

    async def cancel_task_safely():
        nonlocal simulation_task
        if simulation_task and not simulation_task.done():
            simulation_task.cancel()
            try:
                await simulation_task
            except asyncio.CancelledError:
                pass
        simulation_task = None

    async def safe_send(msg: str):
        nonlocal simulation_task
        try:
            async with send_lock:
                await websocket.send_text(msg)
        except:
            stop_event.set()
            if simulation_task and not simulation_task.done():
                simulation_task.cancel()

    def is_running() -> bool:
        return bool(agora_instance and simulation_task and (not simulation_task.done()) and server_state == "RUNNING")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                action_type = payload.get("type", "start")
            except:
                continue

            if action_type == "start":
                stop_event.set()
                await cancel_task_safely()
                stop_event.clear()

                server_state = "RUNNING"

                topic = payload.get("topic", "Truth")
                user_claim = payload.get("user_claim", "") or ""

                cast = build_grand_cast()
                user_p = build_user_philosopher(user_claim) 
                cast.insert(0, user_p)
                agora_instance = Agora(cast, seed=None)
                
                async with state_lock:
                    agora_instance.positions["User"] = (
                        user_claim if user_claim else f"Regarding '{topic}', I believe we need at least refutability (verification/falsification) to approach truth."
                    )

                async def run_sim():
                    nonlocal agora_instance, server_state
                    try:
                        async for line in agora_instance.run_async_generator(topic=topic, rounds=999, state_lock=state_lock, stop_event=stop_event):
                            if stop_event.is_set(): break
                            await safe_send(line)
                        if not stop_event.is_set():
                            await safe_send("\n🏁 [SYSTEM] Debate ended.")
                    except asyncio.CancelledError:
                        pass
                    finally:
                        agora_instance = None
                        server_state = "IDLE"

                simulation_task = asyncio.create_task(run_sim())

            elif action_type == "update":
                if not is_running():
                    await safe_send("\n⚠️ [SYSTEM] Simulation is not running. Start a debate first.")
                    continue
                
                new_claim = (payload.get("text", "") or "").strip()
                if not new_claim: continue

                async with state_lock:
                    if agora_instance:
                        agora_instance.positions["User"] = new_claim
                await safe_send(f"\n✍️ [SYSTEM] User Stance Updated: \"{new_claim[:60]}...\"")

            elif action_type == "stop":
                stop_event.set() 
                await cancel_task_safely()
                
                agora_instance = None
                server_state = "IDLE"
                await safe_send("\n🏁 [SYSTEM] Debate terminated by user.")

    except WebSocketDisconnect:
        stop_event.set()
        await cancel_task_safely()
        agora_instance = None
        server_state = "IDLE"
        print("Client disconnected")
    except Exception as e:
        stop_event.set()
        await cancel_task_safely()
        agora_instance = None
        server_state = "IDLE"
        print(f"Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
