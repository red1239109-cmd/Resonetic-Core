# 🧠 Resonetics Engine: Performance Edition (v6.1)

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-teal)](https://fastapi.tiangolo.com/)
[![Status](https://img.shields.io/badge/Status-Production_Ready-success)](https://github.com/)

> **"Logic is not just semantics; it is a topological structure."**

**Resonetics Engine**은 최신 **위상수학적 데이터 분석(TDA)**과 **대규모 언어 모델(SBERT)**을 결합한 고성능 논리 추론 검증 엔진입니다. 단순한 코사인 유사도를 넘어, 문장 간의 **구조적 연결성(Structural Connectivity)**을 수학적으로 증명합니다.

v6.1은 **GPU 가속**, **Rust 기반 직렬화(orjson)**, **하이브리드 TDA 아키텍처**를 통해 실시간 스트리밍 추론이 가능하도록 최적화되었습니다.

---

## 🚀 Key Features (v6.1)

### ⚡ Extreme Performance
* **GPU Accelerated SBERT:** 배치(Batch) 처리를 통해 임베딩 속도를 **10배 이상** 가속화했습니다.
* **Rust Serialization:** `orjson`을 도입하여 JSON 직렬화 오버헤드를 제거했습니다.
* **Hybrid TDA Architecture:** 캐시(Cache) → 비동기 TDA(Async) → 폴백(Fallback) 3단계 전략으로 속도와 정확도를 모두 잡았습니다.

### 🧬 Topological Reasoning (위상수학적 추론)
* **Persistent Homology:** 데이터의 '구멍(Loop)'을 찾아 논리적 비약이나 단절을 감지합니다.
* **Time-Delay Embedding:** 1차원 텍스트 벡터를 고차원 포인트 클라우드로 변환하여 구조를 분석합니다.
* **Confidence Score:** 추론의 신뢰도를 `Method` (TDA/Fallback)와 `Coherence`를 기반으로 수치화합니다.

### 🛡️ Production Ready
* **Memory Safety:** `SizedLRUCache`를 통해 메모리 누수(OOM)를 원천 차단합니다.
* **Observability:** Prometheus 메트릭(`INFERENCE_REQUESTS`, `TDA_CALC_TIME`)이 내장되어 있습니다.
* **Streaming API:** SSE(Server-Sent Events)를 통해 추론 과정을 실시간으로 중계합니다.

---

## 🛠️ Quick Start

### 1. Prerequisites
* Python 3.9+
* CUDA capable GPU (Recommended) or Multi-core CPU

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/red1239109-cmd/resonetics-engine.git](https://github.com/red1239109-cmd/resonetics-engine.git)
cd resonetics-engine

# Install dependencies
pip install torch sentence-transformers fastapi uvicorn ripser gudhi orjson prometheus_client

3. Running the Server
Option A: With GPU (Recommended)

# Workers=1 to avoid VRAM duplication. Internal process pool handles CPU tasks.
uvicorn resonetics_engine_v6_1:app --host 0.0.0.0 --port 8000 --workers 1

Option B: CPU Only

# Scale workers to utilize CPU cores
uvicorn resonetics_engine_v6_1:app --host 0.0.0.0 --port 8000 --workers 4

Usage Example (Curl)

curl -N -X POST http://localhost:8000/infer_stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Is AI dangerous?",
    "premises": [
      "AI systems learn patterns from massive data.",
      "Human data inherently contains historical biases.",
      "Learning from biased data transfers bias to the model.",
      "Biased models can make unfair decisions in society.",
      "Therefore, AI poses a potential danger."
    ]
  }'

  Response (Stream):

  {"premise": "AI systems...", "conclusion": "Human data...", "coherence": 0.92, "method": "TDA", "confidence": 0.92}
{"premise": "Human data...", "conclusion": "Learning from...", "coherence": 0.88, "method": "CACHE", "confidence": 0.88}
...
