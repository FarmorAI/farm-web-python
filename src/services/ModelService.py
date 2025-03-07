import numpy as np
from src.models.AppleModel import AppleModel
from services.ImagePreprocessor import ImagePreprocessor

class ModelService:
   def __init__(self):
      self.model = AppleModel
      self.prepro = ImagePreprocessor()

   def analyze(self, image: np.ndarray) -> dict:
      try:
         # 이미지 전처리
         processed_image = self.prepro.preprocess(image)

         # 예측 수행
         result = self.model.predict(processed_image)
         return result
      
      except Exception as e:
         return {
            "success":False,
            "error": str(e)
         }

# 싱글톤 인스턴스 생성
modelService = ModelService()