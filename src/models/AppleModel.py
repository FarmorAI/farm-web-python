import tensorflow as tf
import numpy as np

class AppleModel:
   def __init__(self):
      # 훈련된 keras 모델 불러오기
      model_path='src/models/apple_dl_model_v1.keras'
      self.model = tf.keras.models.load_model(model_path)

   # v1 모델 훈련 진행
   def predict(self, image: np.ndarray) -> dict:
      try:
         prediction = self.model.predict(image)
         
         # 예측 결과를 클래스로 변환 (응답 예시: {'특':0., '상':0., '보통':0.})
         predicted_class ={'특': 0, '상': 1, '보통': 2}
         confidence = {k: round(float(prediction[0][v]), 2) for k, v in predicted_class.items()}

         return {
            "success": True,
            "quality": confidence
         }

      except Exception as e:
         return {
            "success": False,
            "error": str(e)
         }

appleModel = AppleModel()