import tensorflow as tf
import numpy as np

class AppleModel:
   def __init__(self, model_path='../models/apple_dl_model_v1.keras'):
      # 훈련된 keras 모델 불러오기
      self.model_v1 = tf.keras.models.load_model(model_path)

   # v1 모델 훈련 진행
   def predict_v1(self, image: np.ndarray) -> dict:
      try:
         prediction = self.model_v1.predict(image)
         
         # 예측 결과를 클래스로 변환 (예: [특, 상, 보통])
         classes = {0:'특', 1:'상', 2:'보통'}
         predicted_class = classes[np.argmax(prediction[0])]
         confidence = float(np.max(prediction[0]))

         return {
            "success": True,
            "quality": predicted_class,
            "confidence": confidence
         }

      except Exception as e:
         return {
            "success": False,
            "error": str(e)
         }