import os
import numpy as np
from PIL import Image
from datetime import datetime
from src.models.AppleModel import appleModel
from src.services.ImagePreprocessor import imagePreprocessor
from src.services.YoloPreprocessor import yoloPreprocessor

class ModelService:
   def __init__(self):
      self.model = appleModel
      self.imagePrepro = imagePreprocessor
      self.yoloPrepro = yoloPreprocessor

   def analyze(self, image: np.ndarray) -> dict:
      try:
         yolo_image, yolo_results = self.yoloPrepro.yolo_detect(image) # YOLO로 이미지 분석
         processed_image = self.imagePrepro.preprocess(yolo_image[0])  # 이미지 전처리
         result = self.model.predict(processed_image)                  # 예측 수행

         # YOLO 이미지 저장
         image_path = self.save_image(yolo_results[0].plot())
         result["image_path"] = image_path
         
         return result
      
      except Exception as e:
         return {
            "success":False,
            "error": str(e)
         }
   
   def save_image(self, image: np.ndarray, save_directory: str = "images") -> str:
      try:
         # 디렉토리 생성 (존재하지 않으면)
         if not os.path.exists(save_directory):
            os.makedirs(save_directory)

         # 이미지 이름 생성 (현재 시간 기준)
         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
         image_name = f"image_{timestamp}.png"
         image_path = os.path.join(save_directory, image_name)

         # 이미지를 저장
         img = Image.fromarray(image)
         img.save(image_path)
         return image_path
      
      except Exception as e:
         return {
            "success": False,
            "error": str(e)
         }

# 싱글톤 인스턴스 생성
modelService = ModelService()