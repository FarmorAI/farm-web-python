# 버전 정보 출력
pip freeze > requirements.txt

# 버전 설치
pip install -r requirements.txt

# fastapi 서버 재가동
uvicorn main:app --reload