import tensorflow as tf
import numpy as np

class AppleModel:
   def __init__(self):
      # 훈련된 keras 모델 불러오기
      model_path='D:/project_final/farm-web-python/src/models/apple_dl_model_v1.keras'
      self.model = tf.keras.models.load_model(model_path)

   # v1 모델 훈련 진행
   def predict(self, image: np.ndarray) -> dict:
      try:
         prediction = self.model.predict(image)
         
         # 예측 결과를 클래스로 변환 (예: [특, 상, 보통])
         classes = {0:'특', 1:'상', 2:'보통'}
         predicted_class = ['특', '상', '보통'] # classes[np.argmax(prediction[0])]
         confidence = [round(float(val), 2) for val in prediction[0]]

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
      
appleModel = AppleModel()