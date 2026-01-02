# Data Refinery Engine (DRE)

**Version**: 1.13.4  
**Backend**: Polars (preferred) + Pandas fallback  
**License**: MIT

고품질 특성 선택을 위한 자동 데이터 정제 엔진입니다.  
컬럼별 Relevance + Quality 점수를 기반으로 동적으로 특성을 유지/제거하며, PID 제어기로 목표 유지율을 안정적으로 추종합니다.

### 주요 특징
- Polars/Pandas 자동 호환
- 안전한 병렬 컬럼 스코어링 (thread/process, 메타 샘플만 전달)
- PID 기반 동적 임계값 조정 (anti-windup 포함)
- 포렌식 수준 로깅 (Timeline / Queue / DAG JSONL 스토어, 로테이션 지원)
- ExplainCard로 각 컬럼 결정 근거 제공
- DAGRunner 포함 (DOT 출력 지원)
- CLI 벤치마크 (행 스윕, 컬럼 스윕)

### 설치
```bash
pip install polars pandas numpy  # 필수
git clone https://github.com/[your-username]/data-refinery-engine.git
cd data-refinery-engine

from dre_v1_13_4_single_trunk import DataRefineryEngine, make_df

engine = DataRefineryEngine(target_kpi="income", parallel=True)
data = make_df(100_000, 100)  # 10만 행, 100 컬럼 샘플 데이터
df_clean, lineage, cards = engine.refine(data)

print(f"Kept {len(lineage['kept'])} / Dropped {len(lineage['dropped'])} columns")
print(f"Final threshold: {lineage['threshold']:.3f}")

python dre_v1_13_4_single_trunk.py --mode benchmark --sizes 10000,50000,100000 --parallel
python dre_v1_13_4_single_trunk.py --mode col-sweep --col-sweep 20,50,100,200 --parallel

MIT License - 자세한 내용은 LICENSE 파일을 참조하세요.

작성자: [red1239109-cmd] (2026)
