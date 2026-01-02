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

# 데모 실행
python dre_v1_13_4_single_trunk.py --mode demo

# 벤치마크
python dre_v1_13_4_single_trunk.py --mode benchmark --parallel

# 컬럼 스윕
python dre_v1_13_4_single_trunk.py --mode col-sweep --parallel
로그는 자동으로 runs/ 디렉토리에 생성됩니다 (Git 무시).

MIT License - 자세한 내용은 LICENSE 파일을 참조하세요.

작성자: [red1239109-cmd] (2026)
