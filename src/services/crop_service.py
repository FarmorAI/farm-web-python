import numpy as np
from app.models.tensorflow_model import CropModel

# 모델 인스턴스 생성
crop_model = CropModel()

def analyze_crop(image: np.ndarray):
   result = crop_model.predict(image)
   return result