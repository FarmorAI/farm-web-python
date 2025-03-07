import cv2
import numpy as np

class ImagePreprocessor:
   def __init__(self, target_size=(224, 224)):
      self.target_size = target_size

   # 이미지 전처리 함수
   def preprocess(self, image: np.ndarray) -> np.ndarray:
      image = np.array(image)

      # 1. 이미지 크기 조정
      image = cv2.resize(image, self.target_size)

      # 3. 샤프닝 필터 정의 (일반적인 샤프닝 커널)
      mask = np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype = np.float32)
      image = cv2.filter2D(image, -1, mask)

      # 4. 이미지 정규화
      image = image.astype(np.float32) / 255.0

      # 5. 배치 차원 추가
      # 모델의 입력은 batch 단위로 처리되기 때문에, 
      # 단일 이미지를 모델에 전달하려면 batch_size=1로 설정해줘야 합니다.
      # 모델이 여러 개의 샘플을 처리하는 방식을 기대하는데, 
      # 단일 이미지 데이터를 제공하면 오류가 발생할 수 있습니다.
      # (224, 224, 3) -> (1, 224, 224, 3) 
      image = np.expand_dims(image, axis=0)
      return image

imagePreprocessor = ImagePreprocessor()