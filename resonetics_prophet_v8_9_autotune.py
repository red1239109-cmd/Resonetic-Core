# ========== v8.9 5-min 검증 (단일 셀) ==========
%%capture
!pip install torch --index-url https://download.pytorch.org/whl/cpu
!pip install numpy pandas matplotlib

# ===== 1. 코드 내려받기 =====
!rm -rf Resonetic-Core
!git clone https://github.com/red1239109-cmd/Resonetic-Core.git
%cd Resonetic-Core

# ===== 2. 300-step 실행 (로그 파일 생성) =====
!python resonetics_prophet_v8_9_autotune.py --steps 300

# ===== 3. 즉각 검증 =====
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 로그 읽기
log_file = 'logs/prophet_run.jsonl'
# 파일이 있는지 os.path.exists로 확인하고, 없으면 백업 파일(.1)을 읽도록 처리
df = pd.read_json(log_file, lines=True) if os.path.exists(log_file) else pd.read_json('logs/prophet_run.jsonl.1', lines=True)

# ① 메인 loss
plt.figure()
plt.plot(df.step, df.loss)
plt.title('Main Loss'); plt.xlabel('step'); plt.ylabel('loss')
plt.show()

# ② knob 궤적
plt.figure()
# 데이터 구조에 따라 키 에러 방지를 위해 get 사용 권장
plt.plot(df.step, df.knobs.apply(lambda x: x.get('eps')), label='eps')
plt.plot(df.step, df.knobs.apply(lambda x: x.get('wR')), label='wR')
plt.legend(); plt.title('Knobs'); plt.xlabel('step')
plt.show()

# ③ PID 진동 (u_pid 컬럼 확인 필요 - v8.9 코드 기준 uR/uF일 수 있음. 확인 후 수정)
# 만약 로그에 'u'라는 단일 컬럼이 없다면 이 부분에서 키 에러가 날 수 있습니다.
# v8.9 autotune 코드는 'pid': {'uR': ..., 'uF': ...} 형태로 저장합니다.
if 'pid' in df.columns:
    plt.figure()
    plt.plot(df.step, df.pid.apply(lambda x: x.get('uR')), label='uR (Gap)')
    plt.plot(df.step, df.pid.apply(lambda x: x.get('uF')), label='uF (Flow)')
    plt.legend()
    plt.title('PID Outputs'); plt.xlabel('step'); plt.ylabel('u')
    plt.show()

# ④ 숫자 요약
print('✅ loss 최종값:', df.loss.iloc[-1])
print('✅ eps  최종값:', df.knobs.iloc[-1]['eps'])
print('✅ wR   최종값:', df.knobs.iloc[-1]['wR'])

# PID 진동 횟수 (uR 기준)
if 'pid' in df.columns:
    uR = df.pid.apply(lambda x: x.get('uR'))
    print('✅ uR 진동 횟수(부호변화):', ((uR.shift() * uR) < 0).sum())
