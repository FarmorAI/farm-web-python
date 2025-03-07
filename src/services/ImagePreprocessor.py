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

      # 2. CLAHE 적용 (대비 향상)
      lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # 이미지를 RGB에서 LAB 색 공간으로 변환
      l, a, b = cv2.split(lab)                      # LAB 이미지에서 L(밝기), A(색상 정보), B(색상 정보) 채널을 분리
      clahe = cv2.createCLAHE(
         clipLimit=2.0, tileGridSize=(8, 8)
      )                                             # 이미지 블록을 8x8로 나눠서 각 블록에 대해 히스토그램 평활화 적용
      l = clahe.apply(l)                            # 명도 정보를 부분별로 보정해, 어두운 곳은 더 밝게, 밝은 곳은 더 어둡게 보정
      lab = cv2.merge((l, a, b))                    # CLAHE가 적용된 l 채널과 원래의 a, b 채널을 다시 합쳐서 LAB 이미지
      image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

      # 3. 샤프닝 필터 정의 (일반적인 샤프닝 커널)
      # mask = np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype = np.float32)
      # image = cv2.filter2D(image, -1, mask)

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