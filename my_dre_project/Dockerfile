# Dockerfile
FROM python:3.12-slim

# 작업 디렉토리
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY dre_v1_6.py .

# 포트 오픈 (대시보드)
EXPOSE 8080

# 실행
CMD ["python", "dre_v1_6.py"]
