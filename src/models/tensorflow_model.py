import tensorflow as tf
import numpy as np
import cv2

class CropModel:
   def __init__(self, model_path="app/static/models/crop_model.h5"):
      self.model = tf.keras.models.load_model(model_path)

   def preprocess(self, image):
      # 이미지 전처리 (크기 조정, 정규화 등)
      image = cv2.resize(image, (224, 224))
      image = image.astype("float32") / 255.0
      image = np.expand_dims(image, axis=0)
      return image

   def predict(self, image):
      processed_image = self.preprocess(image)
      prediction = self.model.predict(processed_image)
      return prediction.tolist()