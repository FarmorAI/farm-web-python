# 가상 환경 설치 (mac)
python3.10 -m venv myenv

# 가상 환경 활성화 (mac)
source myenv/bin/activate

# 가상 환경 활성화 (window)
myenv\Scripts\activate

# 가상 환경 비활성화 (mac, window 동일)
deactivate

# 버전 정보 출력
pip freeze > requirements.txt

# 버전 설치
pip install -r requirements.txt

# fastapi 서버 실행
uvicorn main:app --reload --host 127.0.0.1 --port 9090
python main.py

# 욜로 설치
pip install ultralytics
from ultralytics import YOLO 사용


# 새로운 Conda 가상환경 생성
conda create -n 가상환경이름 python=3.10

# Conda 가상환경 활성화
conda activate 가상환경이름

# zsh 초기화
conda init zsh

# 터미널 다시 시작 
exec zsh