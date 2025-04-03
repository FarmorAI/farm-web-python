# 1. 가벼운 Base Image 사용
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 불필요한 파일 제외 (미리 .dockerignore 설정)
COPY requirements.txt ./

# 4. 패키지 설치 최적화
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# 5. 소스 코드 복사 (필요한 파일만 선택적으로 복사)
COPY . .

# Uvicorn으로 FastAPI 실행
CMD ["python", "main.py"]