# 버전 정보 출력
pip freeze > requirements.txt

# 버전 설치
pip install -r requirements.txt

# fastapi 서버 실행
uvicorn main:app --reload --host 127.0.0.1 --port 9090

