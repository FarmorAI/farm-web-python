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


# 딥러닝 기반 배경 제거 (간단한 예시)
def remove_background_deep_learning(image_path, output_path):
    # 이미지 로드
    img = cv2.imread(image_path)
    
    # 여기서는 외부 라이브러리 사용 예시 (실제로는 설치 필요)
    # pip install rembg
    from rembg import remove
    
    # 딥러닝 모델을 사용해 배경 제거
    result = remove(img)
    
    # 결과 저장
    cv2.imwrite(output_path, result)
    
    return result