# SPDX-License-Identifier: MIT
# Copyright (C) 2026 red1239109-cmd
# ==============================================================================
# File: agora_final.py
# Project: The Grand Philosophical Agora (Production Ready Edition)
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
    return set(re.findall(r"[가-힣A-Za-z]{2,}", text))

CONCEPT_SYNONYMS = {
    "진리": {"진리", "참", "참됨", "사실", "정당화", "인식"},
    "이성": {"이성", "합리", "논리", "추론", "연역"},
    "경험": {"경험", "관찰", "사례", "실험"},
    "보편": {"보편", "필연", "일반", "규범"},
    "도덕": {"도덕", "의무", "정언", "윤리", "존엄", "선"},
    "자유": {"자유", "자율", "의지"},
    "권력": {"권력", "힘", "지배", "위계"},
    "가치": {"가치", "평가", "선악", "의미"},
    "언어": {"언어", "말", "표현", "문법", "사용"},
    "존재": {"존재", "실재", "실체", "존재론"},
    "형이상": {"형이상", "초월", "본질", "이데아"},
    "사회": {"사회", "제도", "규율", "정치", "공적"},
    "역사": {"역사", "계보", "시대", "발전"},
    "주체": {"주체", "자아", "의식"},
    "방법": {"방법", "비판", "분석", "변증", "해체", "검열"},
}

PHILO_GRAPHS = {
    "플라톤": {"형이상": {"진리", "보편", "존재"}, "진리": {"형이상", "보편", "이성"}, "보편": {"진리", "형이상", "이성"}},
    "아리스토텔레스": {"경험": {"진리", "존재", "방법"}, "존재": {"경험", "진리", "방법"}, "방법": {"경험", "존재", "진리"}},
    "칸트": {"이성": {"보편", "도덕", "방법"}, "보편": {"이성", "도덕", "자유"}, "자유": {"도덕", "보편"}, "주체": {"방법", "경험"}, "방법": {"이성", "보편"}},
    "니체": {"권력": {"가치", "진리", "역사"}, "가치": {"권력", "진리"}, "진리": {"권력", "가치"}, "역사": {"권력", "가치"}},
    "소크라테스": {"방법": {"진리", "주체", "이성"}, "주체": {"방법", "진리"}, "진리": {"방법", "이성"}},
    "데카르트": {"주체": {"이성", "진리"}, "이성": {"주체", "진리"}, "진리": {"주체", "이성"}},
    "흄": {"경험": {"진리", "방법"}, "방법": {"경험", "진리"}, "진리": {"경험", "방법"}},
    "스피노자": {"존재": {"이성", "보편", "진리"}, "이성": {"존재", "진리"}, "진리": {"존재", "보편"}},
    "라이프니츠": {"이성": {"보편", "진리", "형이상"}, "형이상": {"이성", "진리"}, "진리": {"이성", "보편"}},
    "로크": {"경험": {"주체", "진리", "사회"}, "주체": {"경험", "진리"}, "사회": {"경험", "진리"}},
    "루소": {"사회": {"자유", "도덕", "가치"}, "자유": {"사회", "도덕"}, "도덕": {"사회", "자유"}},
    "밀": {"가치": {"도덕", "사회", "진리"}, "도덕": {"가치", "사회"}, "사회": {"가치", "도덕"}},
    "마르크스": {"사회": {"역사", "권력", "가치"}, "역사": {"사회", "권력"}, "권력": {"사회", "역사"}},
    "헤겔": {"역사": {"방법", "진리", "사회"}, "방법": {"역사", "진리"}, "진리": {"역사", "방법"}},
    "쇼펜하우어": {"가치": {"존재", "주체"}, "주체": {"가치", "존재"}, "존재": {"주체", "가치"}},
    "키르케고르": {"주체": {"가치", "도덕", "진리"}, "가치": {"주체", "도덕"}, "진리": {"주체", "가치"}},
    "비트겐슈타인": {"언어": {"방법", "진리"}, "방법": {"언어", "진리"}, "진리": {"언어", "방법"}},
    "하이데거": {"존재": {"주체", "진리"}, "주체": {"존재", "진리"}, "진리": {"존재", "주체"}},
    "사르트르": {"자유": {"주체", "가치", "도덕"}, "주체": {"자유", "가치"}, "가치": {"자유", "주체"}},
    "푸코": {"사회": {"권력", "진리", "역사"}, "권력": {"사회", "진리"}, "진리": {"사회", "권력"}},
    "아렌트": {"사회": {"진리", "역사", "가치"}, "가치": {"사회", "진리"}, "진리": {"사회", "가치"}},
    "포퍼": {"방법": {"경험", "진리"}, "경험": {"방법", "진리"}, "진리": {"방법", "경험"}},
    "롤스": {"도덕": {"사회", "자유", "보편"}, "사회": {"도덕", "보편"}, "보편": {"도덕", "사회"}},
}

def extract_concepts(text: str) -> Set[str]:
    out: Set[str] = set()
    for c, syns in CONCEPT_SYNONYMS.items():
        for s in syns:
            if s in text:
                out.add(c)
                break
    return out

def pick_target_concept_from_other(other_claim: str, fallback: str = "진리") -> str:
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
    name: str
    pattern: re.Pattern
    penalty: float
    explanation: str
    repair_hint: str

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

TABOOS = {
    "kant_empirical_jump": InferenceTaboo("경험→보편 점프", re.compile(r"(경험|관찰|사례).*(그래서|따라서).*(보편|필연|규범|의무)", re.UNICODE), 0.65, "경험 도약", "경험 대신 '가능조건'을 요구하라."),
    "kant_ends_justify": InferenceTaboo("목적이 수단 정당화", re.compile(r"(목적|결과).*(수단).*(정당화|합리화)", re.UNICODE), 0.75, "존엄 충돌", "수단-목적 논증을 중단하고 '존엄' 제약을 삽입하라."),
    "nietzsche_universal_morals": InferenceTaboo("보편 도덕 단언", re.compile(r"(보편|절대).*(도덕|윤리|선|악|법칙)", re.UNICODE), 0.70, "보편 도덕 의심", "'누가 이득을 보는가'로 계보학적 전환을 수행하라."),
    "nietzsche_truth_worship": InferenceTaboo("진리 숭배", re.compile(r"(진리).*(최고|신성|절대|숭배)", re.UNICODE), 0.55, "진리 가치 심문", "'진리가 왜 선인가'를 묻는 가치전도 질문을 삽입하라."),
    "hume_necessary_causation": InferenceTaboo("필연 인과 단언", re.compile(r"(원인|인과).*(반드시|필연|절대)", re.UNICODE), 0.65, "필연성 회의", "'습관/기대'로 설명을 환원하라."),
    "witt_metaphysics_assert": InferenceTaboo("형이상학 단언", re.compile(r"(이데아|초월|본질).*(존재한다|실재한다|확실하다)", re.UNICODE), 0.70, "언어 한계 초과", "'언어 사용 규칙'으로 전환하라."),
    "foucault_truth_neutral": InferenceTaboo("진리 중립성", re.compile(r"(진리).*(중립|순수|무관)", re.UNICODE), 0.70, "권력-지식 망각", "'제도/규율/정상화'를 넣어 권력 프레임으로 전환하라."),
}

@dataclass
class Lexicon:
    core: List[str]
    evidentials: List[str]
    hedges: List[str]
    intensifiers: List[str]
    metaphors: List[str]
    taboo_softeners: List[str]

@dataclass
class ReasoningOps:
    ops: List[str]

@dataclass
class StyleProfile:
    rhetoric_bias: float
    justification_bias: float
    interrogation_bias: float
    poetic_bias: float

@dataclass
class Philosopher:
    name: str
    era: str
    truth_vector: Dict[str, float]
    lexicon: Lexicon
    reasoning: ReasoningOps
    taboo: List[InferenceTaboo]
    style: StyleProfile
    phase: Dict[str, float] = field(default_factory=lambda: {"open": 0.5, "attack": 0.5, "synthesize": 0.5})

@dataclass
class ArgGraph:
    claim: str = ""
    warrant: str = ""
    attack: str = ""
    constraint: str = ""
    synthesis: str = ""

def pick(xs: List[str]) -> str:
    return random.choice(xs) if xs else ""

def tension_to_register(t: float) -> str:
    return "low" if t < 0.33 else "mid" if t < 0.66 else "high"

# --- Ops ---
def op_define_split(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    w = pick(p.lexicon.evidentials) or "우선"
    g.claim = f"{w} '{tgt}'를 말할 때, 우리는 '사실'과 '정당화'를 섞어 말합니다."
    g.warrant = "정의가 흐리면 논쟁은 말의 미끄러짐이 됩니다."

def op_condition_censor(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'를 판단한다는 말 자체가 성립하려면, 무엇이 그것을 가능하게 합니까?"
    g.warrant = "경험만으로는 보편 타당성을 보장할 수 없습니다."
    g.constraint = "따라서 가능한 조건을 먼저 세워야 합니다."

def op_empirical_classify(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'라고 불리는 사례들을 모아봅시다: 과학의 검증, 법정의 증명, 일상의 신뢰."
    g.warrant = "작동 방식이 다른 것들을 하나로 뭉치면 설명이 무너집니다."
    g.constraint = "분류와 원인 분석이 먼저입니다."

def op_reductio(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.attack = f"좋습니다. 만약 '{tgt}'가 절대적이라 가정합시다. 그러면 참과 거짓의 구분 자체가 붕괴합니다."
    g.warrant = "구분이 붕괴하면 주장도 스스로의 발판을 잃습니다."

def op_genealogy_expose(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = random.choice([
        f"'{tgt}'는 순수한 얼굴을 하고 등장하지만, 역사적으로는 누군가의 손에 들린 도구였다.",
        f"'{tgt}'라는 말이 등장하는 순간, 이미 누군가의 기준이 승리한 것이다.",
        f"'{tgt}'를 말하는 방식 자체가 힘의 배치를 드러낸다.",
    ])
    g.attack = random.choice([
        "누가 그것을 말할 권리를 독점했는가?",
        "그 말이 무엇을 정상으로 만들고 무엇을 비정상으로 밀어냈는가?",
        "그 담론이 누구를 ‘강자’로, 누구를 ‘약자’로 배치했는가?",
    ])
    g.warrant = "계보를 따라가면, ‘참’은 종종 가치와 권력의 냄새를 풍긴다."

def op_value_invert(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = random.choice([
        f"왜 '{tgt}'가 선이라는 전제가 자동으로 통과하는가?",
        f"'{tgt}'를 숭배하는 자들은 대체 무엇을 두려워하는가?",
        f"'{tgt}'가 삶을 강화한다는 증거가 있는가—아니면 삶을 마비시키는가?",
    ])
    g.warrant = "가치의 우선순위를 뒤집어 점검하지 않으면, ‘진리’는 우상이 된다."
    g.attack = "나는 우상을 부수는 쪽을 택한다."

def op_language_therapy(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"여기서 혼란은 '{tgt}'라는 단어의 사용 규칙에서 비롯될 수 있습니다."
    g.warrant = "말의 쓰임을 정리하면, 문제의 절반은 사라집니다."
    g.constraint = "형이상학적 단언 대신 사용 규칙을 보세요."

def op_power_knowledge(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = "진리는 지식의 형태를 띠지만, 지식은 제도와 규율과 연결됩니다."
    g.attack = f"'{tgt}' 담론이 무엇을 정상화하고 무엇을 배제하는지 보지 않으면 핵심을 놓칩니다."
    g.warrant = "권력-지식 장치가 '참'의 조건을 만듭니다."

def op_public_world(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'는 개인 머릿속에만 있지 않습니다. 공적 세계에서 말과 행위로 드러납니다."
    g.warrant = "진리는 세계-공유와 책임의 문제입니다."
    g.synthesis = "그래서 진리 논쟁은 정치적·윤리적 차원을 피할 수 없습니다."

def op_elenchus(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'를 안다고 말하는군. 그렇다면 '{tgt}'를 이루는 최소 조건 하나만 말해보게."
    g.attack = "그 조건이 흔들리면, 너의 '안다'는 말도 같이 흔들린다."
    g.warrant = "정의 없이 확신만 남으면, 우리는 말에 속는다."

def op_methodic_doubt(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"나는 '{tgt}'에 대해, 의심 가능한 것은 전부 의심해보겠다."
    g.warrant = "의심을 통과한 것만이 확실성의 자격을 얻는다."
    g.constraint = "그러니 먼저 의심에 견디는 토대를 제시하라."

def op_hume_skeptic(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'를 말할 때, 우리는 반복에서 생긴 기대를 '필연'으로 착각하곤 한다."
    g.warrant = "인과·필연·확실은 종종 습관의 다른 이름이다."
    g.attack = "너의 확신은 경험의 빈틈을 메우는 상상일 수 있다."

def op_dialectic(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'는 고정된 점이 아니라, 충돌 속에서 전개되는 과정이다."
    g.warrant = "한쪽만 붙들면 모순이 쌓이고, 그 모순이 다음 국면을 연다."
    g.synthesis = "그러니 반대항을 제거하지 말고, 더 높은 통일로 들어올려라."

def op_pessimism_will(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'는 이성의 승리가 아니라, 의지가 자기 고통을 포장하는 방식일 수 있다."
    g.warrant = "삶이 먼저이고, 이성은 그 뒤를 따라 미화한다."
    g.attack = "너의 진리는 고통을 덜어주는 약인가, 현실을 가리는 안개인가?"

def op_subjective_truth(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'가 객관적 공문서처럼 존재한다는 믿음부터 의심하자."
    g.warrant = "중요한 것은 내가 그 진리에 어떻게 ‘걸려드는가’다."
    g.attack = "너의 진리는 삶을 바꾸는가, 아니면 남을 심판하는 도구인가?"

def op_being_unconceal(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'를 ‘정확한 명제’로만 보지 마라. 진리는 드러남(알레테이아)이다."
    g.warrant = "무엇이 숨겨지고 무엇이 드러나는지, 그 구조를 보라."
    g.constraint = "먼저 ‘존재’가 어떻게 열리는지 물어야 한다."

def op_existential_choice(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'를 찾겠다는 말은 결국 선택의 문제다."
    g.warrant = "너는 선택하지 않을 자유도 없다. 침묵도 하나의 선택이다."
    g.attack = "너의 진리는 책임을 지게 만드는가, 책임을 회피하게 만드는가?"

def op_marx_ideology(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'는 머리에서 떨어진 별이 아니다. 생산과 제도의 바닥에서 만들어진다."
    g.warrant = "지배적 진리는 지배적 관계를 정당화하는 경향이 있다."
    g.attack = "누가 그 진리로 이득을 보는지부터 보라."

def op_utilitarian(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'를 논할 때, 결과의 파급을 계산에서 제외할 수 없다."
    g.warrant = "규칙은 행복/고통의 총량과 연결될 때 설득력을 얻는다."
    g.constraint = "따라서 어떤 말이 실제로 어떤 피해/이득을 만드는지 보라."

def op_falsification(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'라고 주장한다면, 어떤 조건에서 그 주장이 반박되는지 먼저 말해야 한다."
    g.warrant = "반박 가능성이 없는 확신은 지식이 아니라 신념이다."
    g.attack = "너의 진리는 위험을 감수하는가, 아니면 도망치기 쉬운가?"

def op_original_position(p, topic, other, g, t, target=""):
    tgt = target if target else topic
    g.claim = f"'{tgt}'의 규칙을 세운다면, 내가 누구인지 모르는 상태에서 그 규칙을 선택하겠는가?"
    g.warrant = "공정은 자기 이익을 가리는 장치에서 시험된다."
    g.constraint = "그러니 모두가 받아들일 수 있는 조건을 먼저 세워라."

OPS = {
    "define_split": op_define_split,
    "condition_censor": op_condition_censor,
    "empirical_classify": op_empirical_classify,
    "reductio": op_reductio,
    "genealogy_expose": op_genealogy_expose,
    "value_invert": op_value_invert,
    "language_therapy": op_language_therapy,
    "power_knowledge": op_power_knowledge,
    "public_world": op_public_world,
    "elenchus": op_elenchus,
    "methodic_doubt": op_methodic_doubt,
    "hume_skeptic": op_hume_skeptic,
    "dialectic": op_dialectic,
    "pessimism_will": op_pessimism_will,
    "subjective_truth": op_subjective_truth,
    "being_unconceal": op_being_unconceal,
    "existential_choice": op_existential_choice,
    "marx_ideology": op_marx_ideology,
    "utilitarian": op_utilitarian,
    "falsification": op_falsification,
    "original_position": op_original_position,
}

# --- Helpers ---
def apply_taboo_repairs(p: Philosopher, text: str, hits: List[InferenceTaboo], topic: str) -> str:
    if not hits: return text
    repairs = []
    for h in hits[:2]:
        if "가능조건" in h.repair_hint: repairs.append("중요한 건 '가능 조건'입니다.")
        elif "존엄" in h.repair_hint: repairs.append("어떤 목적도 존엄을 해칠 순 없습니다.")
        elif "계보학" in h.repair_hint: repairs.append("누가 선이라 부르며 힘을 얻는가?")
        elif "가치전도" in h.repair_hint: repairs.append("‘진리가 왜 선인가’부터 다시 묻자.")
        elif "습관" in h.repair_hint: repairs.append("'필연'은 단지 습관일 수 있습니다.")
        elif "언어 사용" in h.repair_hint: repairs.append("그 단언은 언어의 오용일 수 있습니다.")
        elif "권력-지식" in h.repair_hint: repairs.append("진리는 권력 장치와 얽혀 있습니다.")
    soft = pick(p.lexicon.taboo_softeners)
    add = (" " + soft + " " if soft else " ") + " ".join(repairs)
    return (text + add).strip()

def ensure_concept_anchor(p: Philosopher, text: str) -> str:
    cg = concept_graph_score(p.name, text)
    if cg >= 0.24: return text
    anchors = {
        "칸트": " 보편 타당성을 위해선 가능 조건을 요구해야 합니다.",
        "니체": " 결국 그 '진리'는 가치의 가면이다!",
        "플라톤": " 변하지 않는 기준이 없으면 진리는 흩어집니다.",
        "아리스토텔레스": " 사례와 원인 분석으로 말을 세워야 합니다.",
        "소크라테스": " 너는 그것을 진정으로 아는가?",
        "데카르트": " 나는 생각한다, 고로 존재한다.",
        "마르크스": " 문제는 해석이 아니라 변혁이다.",
        "비트겐슈타인": " 말할 수 없는 것엔 침묵해야 합니다.",
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
    if entropy_boost > 0.0 and len(unique_ops) >= 4:
        k += 1

    chosen: Set[str] = set()
    attempts = 0
    target_k = min(k, len(unique_ops))
    while len(chosen) < target_k and attempts < 20:
        attempts += 1
        r = random.random()
        cum = 0.0
        for opn, pr in zip(unique_ops, probs):
            cum += pr
            if r <= cum:
                chosen.add(opn)
                break

    if len(chosen) < target_k:
        remaining = [op for op in unique_ops if op not in chosen]
        if remaining:
            chosen.update(random.sample(remaining, min(len(remaining), target_k - len(chosen))))

    ordered_chosen = [op for op in unique_ops if op in chosen]
    target = pick_target_concept_from_other(other_claim, fallback=topic)
    for opn in ordered_chosen:
        OPS[opn](p, topic, other_claim, g, tension, target=target)
    return g

def linearize(p: Philosopher, g: ArgGraph, tension: float) -> str:
    reg = tension_to_register(tension)
    if reg == "low":
        lead = pick(p.lexicon.hedges) or "아마도"
    elif reg == "mid":
        lead = "그러나"
    else:
        lead = pick(p.lexicon.intensifiers) or "단호히"

    parts: List[str] = []
    if p.style.interrogation_bias > 0.65:
        parts.append(f"{lead}, 먼저 묻겠습니다.")

    if reg == "high" and p.style.rhetoric_bias >= p.style.justification_bias:
        if g.attack: parts.append(g.attack)
        if g.claim: parts.append(g.claim)
        if g.warrant: parts.append(g.warrant)
        if g.constraint: parts.append(g.constraint)
        if g.synthesis: parts.append(g.synthesis)
    else:
        if g.claim: parts.append(f"{lead}, {g.claim}")
        if g.warrant: parts.append(g.warrant)
        if g.constraint: parts.append(g.constraint)
        if g.attack: parts.append(g.attack)
        if g.synthesis: parts.append(g.synthesis)

    if p.style.poetic_bias > 0.6 and p.lexicon.metaphors:
        parts.insert(0, pick(p.lexicon.metaphors))

    text = " ".join([s.strip() for s in parts if s and s.strip()])
    if random.random() < 0.25:
        core_word = pick(p.lexicon.core)
        if core_word:
            text = f"({core_word}) {text}"

    if p.name == "니체":
        text = text.replace("합니다", "한다").replace("입니다", "이다")
        if reg == "high":
            text = text.replace(".", "!")
    return text.strip()

@dataclass
class Verdict:
    status: str
    note: str
    tension: float
    coherence: float
    novelty: float
    is_warmup: bool = False

class Arbiter:
    def __init__(self, cast_size: int):
        self.t_hist: List[float] = []
        self.c_hist: List[float] = []
        self.n_hist: List[float] = []
        self.last: List[str] = []
        self.p_last: Dict[str, str] = {}
        self.warmup_turns = cast_size * WARMUP_CAST_PASSES
        self.window_size = WINDOW_SIZE
        self.turn_count = 0
        self.warmup_done = False
        self.calibration_snapshot = (0.0, 0.0, 0.0)

    def novelty_check(self, text: str, speaker: str) -> float:
        toks = tokenize(text)
        glob_n = 1.0
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
        t_raw = sum(1 for m in ["붕괴", "폭로", "검열", "배제", "도약", "위선", "지배", "가면", "우상", "독점", "투쟁"] if m in text) + min(2, text.count("!") // 2)
        t = clamp(t_raw / 6.0, 0.0, 1.0)
        c_density = clamp(len(extract_concepts(text)) / 6.0, 0.0, 1.0)
        c = clamp(c_density, 0.0, 1.0)
        n = self.novelty_check(text, speaker)

        self.turn_count += 1
        self.t_hist.append(t)
        self.c_hist.append(c)
        self.n_hist.append(n)
        self.last.append(text)
        self.p_last[speaker] = text

        if not self.warmup_done:
            if self.turn_count < self.warmup_turns:
                return Verdict("CONTINUE", "(Warm-up)", t, c, n, is_warmup=True)
            elif self.turn_count == self.warmup_turns:
                self.calibration_snapshot = (t, c, n)
                self.warmup_done = True
                self.t_hist.clear()
                self.c_hist.clear()
                self.n_hist.clear()
                self.last.clear()
                self.p_last.clear()  # [Method] Total Amnesia for fresh start
                return Verdict("RESET", f"(Calibration Complete) [Snap: T{t:.2f}/C{c:.2f}/N{n:.2f}]", t, c, n, is_warmup=True)

        if len(self.t_hist) >= 5:
            if (sum(1 for x in self.t_hist[-5:] if x > 0.80) >= 4) and (sum(1 for y in self.c_hist[-5:] if y < 0.28) >= 3):
                return Verdict("MELTDOWN", "파국 임박.", t, c, n)

        if len(self.n_hist) >= 6:
            if sum(1 for x in self.n_hist[-6:] if x < 0.25) >= 5:
                return Verdict("DEADLOCK", "교착 상태.", t, c, n)

        if len(self.t_hist) >= 6:
            th = self.t_hist
            ch = self.c_hist
            if th[-6] > 0.70 and th[-1] < 0.35 and ch[-1] > 0.45:
                return Verdict("CONSENSUS", "공명 발생.", t, c, n)

        return Verdict("CONTINUE", "", t, c, n)

    def adapt(self, philos: List[Philosopher], verdict: Verdict):
        if verdict.status == "RESET":
            (t, c, n) = self.calibration_snapshot
            for p in philos:
                p.phase["open"] = clamp(p.phase["open"] + 0.1 * n, 0.0, 1.0)
            return

        if verdict.is_warmup:
            return

        for p in philos:
            if verdict.status == "MELTDOWN":
                p.phase["attack"] = clamp(p.phase["attack"] - 0.15, 0.0, 1.0)
                p.phase["open"] = clamp(p.phase["open"] + 0.10, 0.0, 1.0)
                p.phase["synthesize"] = clamp(p.phase["synthesize"] + 0.10, 0.0, 1.0)
            elif verdict.status == "DEADLOCK":
                p.phase["open"] = clamp(p.phase["open"] + 0.15, 0.0, 1.0)
                p.phase["attack"] = clamp(p.phase["attack"] + 0.05, 0.0, 1.0)
            elif verdict.status == "CONSENSUS":
                p.phase["synthesize"] = clamp(p.phase["synthesize"] + 0.15, 0.0, 1.0)
                p.phase["attack"] = clamp(p.phase["attack"] - 0.10, 0.0, 1.0)
            else:
                for k in p.phase:
                    p.phase[k] = clamp(p.phase[k] * 0.98 + 0.01, 0.0, 1.0)

def build_user_philosopher(user_claim: str) -> Philosopher:
    cs = extract_concepts(user_claim or "")
    if not cs:
        cs = {"진리"}
    vec = {c: 1.0 for c in cs}
    soft_mix = ["그런데", "하지만", "제 생각엔", "요지는", "정리하면,"]
    lex = Lexicon(
        core=["나의 주장", "직관", "사례"],
        evidentials=["제가 보기엔", "경험상", "생각해보면"],
        hedges=["아마", "가능성은", "일단"],
        intensifiers=["확실히", "분명히", "강하게"],
        metaphors=["말은 지도이고, 세계는 지형입니다."],
        taboo_softeners=soft_mix,
    )
    style = StyleProfile(rhetoric_bias=0.55, justification_bias=0.65, interrogation_bias=0.60, poetic_bias=0.20)
    ops = ["define_split", "empirical_classify", "falsification", "elenchus"]
    return Philosopher(name="사용자", era="현대", truth_vector=vec, lexicon=lex, reasoning=ReasoningOps(ops), taboo=[], style=style)

def build_grand_cast() -> List[Philosopher]:
    def L(core, evid, hed, inten, meta, soft): return Lexicon(core, evid, hed, inten, meta, soft)
    def S(rhet, just, inter, poet): return StyleProfile(rhet, just, inter, poet)
    def P(name, era, vec, lex, ops, taboo, style): return Philosopher(name, era, vec, lex, ReasoningOps(ops), taboo, style)

    soft_mix = ["그러나", "하지만", "다만", "그럼에도", "물론,", "인정합니다만,"]
    return [
        P("플라톤","고대",{"형이상":0.90,"보편":0.92,"이성":0.80}, L(["이데아","본질"],["우선"],["아마도"],["단호히"],["그림자를 보며 진리를 말할 순 없습니다."],soft_mix), ["define_split","reductio","condition_censor"],[],S(0.55,0.65,0.55,0.35)),
        P("아리스토텔레스","고대",{"경험":0.88,"이성":0.75,"방법":0.80}, L(["원인","분류"],["관찰하자면"],["대체로"],["정확히"],["대상을 해부대 위에 올려야 합니다."],soft_mix), ["empirical_classify","define_split"],[],S(0.35,0.85,0.35,0.10)),
        P("칸트","근대",{"이성":0.90,"보편":0.88,"도덕":0.82}, L(["가능 조건","보편 타당성"],["따져보면"],["우선"],["결코"],["규칙 없는 판단은 나침반 없는 항해입니다."],soft_mix), ["condition_censor","reductio"], [TABOOS["kant_empirical_jump"]], S(0.30,0.95,0.75,0.05)),
        P("니체","현대",{"권력":0.90,"가치":0.85,"진리":0.55}, L(["망치","우상","힘"],["보라"],["때때로"],["가차없이"],["진리는 약자의 위안이자 강자의 도구다."],soft_mix), ["genealogy_expose","value_invert","reductio"], [TABOOS["nietzsche_universal_morals"]], S(0.90,0.45,0.55,0.75)),
        P("소크라테스","고대",{"방법":0.95,"주체":0.75}, L(["반문","무지"],["그렇다면"],["아마"],["분명히"],["나는 내가 모른다는 것을 안다."],soft_mix), ["elenchus","define_split"],[],S(0.45,0.60,0.95,0.05)),
        P("데카르트","근대",{"주체":0.92,"이성":0.85}, L(["명석판명","의심"],["단호히"],["일단"],["확실히"],["흔들리는 것은 모두 걷어내라."],soft_mix), ["methodic_doubt","define_split"],[],S(0.40,0.85,0.55,0.10)),
        P("흄","근대",{"경험":0.92,"방법":0.80}, L(["습관","인상"],["경험상"],["대개"],["결국"],["우리는 논리보다 믿음을 먼저 배운다."],soft_mix), ["hume_skeptic","empirical_classify"], [TABOOS["hume_necessary_causation"]], S(0.25,0.80,0.55,0.10)),
        P("스피노자","근대",{"존재":0.90,"이성":0.85}, L(["필연","자연"],["필연적으로"],["정확히"],["반드시"],["자연은 변덕이 아니라 질서다."],soft_mix), ["define_split","reductio"],[],S(0.35,0.90,0.45,0.10)),
        P("마르크스","근대",{"사회":0.92,"역사":0.85,"권력":0.85}, L(["계급","생산"],["현실적으로"],["대개"],["단호히"],["중요한 것은 해석이 아니라 변혁이다."],soft_mix), ["marx_ideology","power_knowledge"],[],S(0.70,0.55,0.55,0.20)),
        P("비트겐슈타인","현대",{"언어":0.95,"방법":0.85}, L(["언어게임","규칙"],["보자면"],["아마도"],["정확히"],["말할 수 없는 것엔 침묵하라."],soft_mix), ["language_therapy","define_split"], [TABOOS["witt_metaphysics_assert"]], S(0.25,0.85,0.55,0.10)),
        P("헤겔","근대",{"역사":0.92,"방법":0.85,"진리":0.75}, L(["변증","전개"],["따라가면"],["일단"],["결국"],["진리는 과정이다."],soft_mix), ["dialectic","define_split"],[],S(0.55,0.65,0.45,0.15)),
        P("쇼펜하우어","근대",{"주체":0.75,"존재":0.70,"가치":0.65}, L(["고통","의지"],["솔직히"],["대개"],["단호히"],["삶은 고통의 진자다."],soft_mix), ["pessimism_will","define_split"],[],S(0.55,0.55,0.55,0.25)),
        P("키르케고르","근대",{"주체":0.90,"가치":0.80}, L(["실존","결단"],["먼저"],["어쩌면"],["결국"],["진리는 살아내는 것이다."],soft_mix), ["subjective_truth","existential_choice"],[],S(0.60,0.55,0.70,0.35)),
        P("하이데거","현대",{"존재":0.92,"주체":0.75}, L(["존재","드러남"],["먼저"],["어쩌면"],["결국"],["진리는 숨김과 드러남의 싸움이다."],soft_mix), ["being_unconceal","define_split"],[],S(0.55,0.60,0.55,0.25)),
        P("사르트르","현대",{"자유":0.92,"주체":0.88}, L(["자유","책임"],["단호히"],["때때로"],["결국"],["인간은 자유라는 형벌을 받았다."],soft_mix), ["existential_choice","define_split"],[],S(0.70,0.55,0.55,0.30)),
        P("푸코","현대",{"사회":0.90,"권력":0.90}, L(["규율","장치"],["추적하면"],["대개"],["분명히"],["진리는 장치 속에서 생산된다."],soft_mix), ["power_knowledge","genealogy_expose"], [TABOOS["foucault_truth_neutral"]], S(0.70,0.55,0.55,0.20)),
        P("아렌트","현대",{"사회":0.90,"가치":0.78}, L(["공적세계","행위"],["보자면"],["어쩌면"],["분명히"],["진리는 세계를 함께 드는 것이다."],soft_mix), ["public_world","define_split"],[],S(0.45,0.70,0.55,0.15)),
        P("포퍼","현대",{"방법":0.92,"경험":0.80}, L(["반증","가설"],["먼저"],["일단"],["분명히"],["지식은 반박을 통해 자란다."],soft_mix), ["falsification","empirical_classify"],[],S(0.35,0.80,0.55,0.05)),
        P("롤스","현대",{"도덕":0.90,"사회":0.85}, L(["공정","원초상태"],["가정해보면"],["일단"],["분명히"],["규칙은 약자의 자리에서 견뎌야 한다."],soft_mix), ["original_position","condition_censor"],[],S(0.35,0.85,0.55,0.05)),
        P("라이프니츠","근대",{"이성":0.88,"보편":0.85}, L(["충분이유","조화"],["따져보면"],["아마도"],["필연적으로"],["왜 무가 아니라 유인가?"],soft_mix), ["condition_censor","define_split"],[],S(0.35,0.85,0.55,0.20)),
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
        yield f"🏛️ Grand Agora v27.1 (Production Ready) | 주제: {topic}"
        yield f"참여 ({len(self.philos)}명): {', '.join(p.name for p in self.philos[:5])}..."
        yield "-" * 50
        yield "\n[🎤 개회사 및 캘리브레이션 진행 중...]"

        base_tension = 0.28

        if state_lock:
            async with state_lock:
                if "사용자" not in self.positions:
                    self.positions["사용자"] = f"저는 '{topic}'에 대해, 최소한 반박 가능성(검증/반증)이 있어야 진리에 가깝다고 봅니다."
        else:
            if "사용자" not in self.positions:
                self.positions["사용자"] = f"저는 '{topic}'에 대해, 최소한 반박 가능성(검증/반증)이 있어야 진리에 가깝다고 봅니다."

        opening_buffer: Dict[str, str] = {}
        for p in self.philos:
            if stop_event and stop_event.is_set():
                return

            if p.name == "사용자":
                if state_lock:
                    async with state_lock:
                        s = self.positions["사용자"]
                else:
                    s = self.positions["사용자"]
                opening_buffer[p.name] = s
                self.arb.p_last[p.name] = s
                yield f"[개회사] 사용자: {s[:60]}..."
            else:
                g = build_arggraph(p, topic, "", base_tension, "open")
                s = linearize(p, g, base_tension)
                s = ensure_concept_anchor(p, s)
                opening_buffer[p.name] = s
                self.arb.p_last[p.name] = s
                if random.random() < 0.25:
                    yield f"[개회사] {p.name}: {s[:60]}..."
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

                if me.name == "사용자":
                    if state_lock:
                        async with state_lock:
                            s = self.positions.get("사용자", "").strip()
                    else:
                        s = self.positions.get("사용자", "").strip()

                    if not s:
                        yield "\n🗣️ 사용자: (이번 턴은 발언 없음)"
                        continue

                    verdict = self.arb.judge(s, me.name)
                    hud = f"   📊 [Mode:USER->USER   | Fric:0.00] (T:{verdict.tension:.2f} C:{verdict.coherence:.2f} N:{verdict.novelty:.2f})"
                    yield f"\n🗣️ 사용자:\n\"{s}\""
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
                    yield "\n🛑 [SYSTEM] 엔트로피 임계점 초과. 파국(MELTDOWN)으로 인해 토론 중단."
                    return
                if verdict.status == "CONSENSUS":
                    yield "\n✅ [SYSTEM] 공명(Resonance) 발생. 합의 도달."
                    return

                await asyncio.sleep(0.1)

# ============================================================
# [SECTION 2] The Server & UI
# ============================================================

app = FastAPI()

html = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agora Interactive</title>
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
    <h1>🏛️ Grand Philosophical Agora (Production Ready)</h1>
    <div id="chat-container">
        <div class="message system">SYSTEM: 연결되었습니다. 주제와 초기 입장을 입력하여 토론을 시작하세요.</div>
    </div>
    <div id="input-area">
        <div class="input-row">
            <input type="text" id="topicInput" placeholder="주제: (예: 정의란 무엇인가?)" />
        </div>
        <div class="input-row">
            <input type="text" id="userClaimInput" placeholder="나의 주장 / 개입 (Enter로 전송)" />
            <button id="startBtn" onclick="sendAction('start')">토론 시작</button>
            <button id="stopBtn" onclick="sendAction('stop')" disabled>중단</button>
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
                msg.innerText = "SYSTEM: 서버에 연결되었습니다.";
                chat.appendChild(msg);

                // Reset UI State on Reconnect
                startBtn.disabled = false;
                stopBtn.disabled = true;
                startBtn.innerText = "토론 시작";
            };

            ws.onmessage = function(event) {
                const msg = document.createElement("div");
                msg.className = "message";
                let text = event.data;

                if (text.includes("📊")) msg.className += " hud";
                else if (text.includes("SYSTEM") || text.includes("개회사")) msg.className += " system";
                else if (text.includes("🔄")) msg.className += " reroll";
                else if (text.includes("🗣️")) msg.className += " speaker";

                msg.innerText = text;
                chat.appendChild(msg);
                chat.scrollTop = chat.scrollHeight;

                if (text.includes("토론 중단") || text.includes("합의 도달") || text.includes("토론이 종료") || text.includes("토론을 종료합니다")) {
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    startBtn.innerText = "새 토론 시작";
                }
            };

            ws.onclose = function() {
                const msg = document.createElement("div");
                msg.className = "message system";
                msg.innerText = "SYSTEM: 연결이 끊어졌습니다. 재연결 시도 중...";
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
                if (!topic) { alert("주제를 입력해주세요."); return; }

                startBtn.disabled = true;
                stopBtn.disabled = false;
                startBtn.innerText = "진행 중...";
                chat.innerHTML = "";
                ws.send(JSON.stringify({ type: 'start', topic: topic, user_claim: userClaim }));
                claimInput.value = "";
                claimInput.placeholder = "토론 중 언제든지 개입할 수 있습니다...";
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
    stop_event = asyncio.Event()  # For internal loop breaking
    server_state = "IDLE"  # IDLE, RUNNING

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
            # [Zombie Killer+] Trigger stop & cancel if socket fails
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
                # stop any previous simulation cleanly
                stop_event.set()
                await cancel_task_safely()
                stop_event.clear()

                server_state = "RUNNING"

                topic = payload.get("topic", "진리")
                user_claim = payload.get("user_claim", "") or ""

                cast = build_grand_cast()
                user_p = build_user_philosopher(user_claim)
                cast.insert(0, user_p)
                agora_instance = Agora(cast, seed=None)

                async with state_lock:
                    agora_instance.positions["사용자"] = (
                        user_claim if user_claim else f"저는 '{topic}'에 대해, 최소한 반박 가능성(검증/반증)이 있어야 진리에 가깝다고 봅니다."
                    )

                async def run_sim():
                    nonlocal agora_instance, server_state
                    try:
                        async for line in agora_instance.run_async_generator(topic=topic, rounds=999, state_lock=state_lock, stop_event=stop_event):
                            if stop_event.is_set():
                                break
                            await safe_send(line)
                        if not stop_event.is_set():
                            await safe_send("\n🏁 [SYSTEM] 토론이 종료되었습니다.")
                    except asyncio.CancelledError:
                        pass
                    finally:
                        agora_instance = None
                        server_state = "IDLE"

                simulation_task = asyncio.create_task(run_sim())

            elif action_type == "update":
                # [Server Guard] Strict State Check (+ task done check)
                if not is_running():
                    await safe_send("\n⚠️ [SYSTEM] 실행 중이 아닙니다. 먼저 토론을 시작하세요.")
                    continue

                new_claim = (payload.get("text", "") or "").strip()
                if not new_claim:
                    continue

                async with state_lock:
                    if agora_instance:
                        agora_instance.positions["사용자"] = new_claim
                await safe_send(f"\n✍️ [SYSTEM] 사용자 입장 업데이트 반영: \"{new_claim[:60]}...\"")

            elif action_type == "stop":
                stop_event.set()
                await cancel_task_safely()

                agora_instance = None
                server_state = "IDLE"
                await safe_send("\n🏁 [SYSTEM] 사용자에 의해 토론을 종료합니다.")

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
