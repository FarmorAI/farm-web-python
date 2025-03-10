import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from fastapi import UploadFile
from starlette.datastructures import Headers
from src.models.AppleModel import appleModel
from src.preprocessor.ImagePreprocessor import imagePreprocessor
from src.preprocessor.YoloPreprocessor import yoloPreprocessor

class ModelService:
   def __init__(self):
      self.model = appleModel
      self.imagePrepro = imagePreprocessor
      self.yoloPrepro = yoloPreprocessor

   # 이미지 분석 및 결과 반환
   async def analyze(self, image: np.ndarray) -> tuple:
      try:
         yolo_image, yolo_results = self.yoloPrepro.yolo_detect(image) # YOLO로 이미지 분석
         processed_image = self.imagePrepro.preprocess(yolo_image[0])  # 이미지 전처리
         result = self.model.predict(processed_image)                  # 예측 수행
         processed_image = self.convert_to_upload_file(yolo_results[0].plot())
         return processed_image, result
      
      except Exception as e:
         return {
            "success":False,
            "error": str(e)
         }

   # UploadFile로 변환하는 함수
   def convert_to_upload_file(self, image: np.ndarray, filename: str = "default.png") -> UploadFile:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR 이미지를 RGB로 변환
      image = Image.fromarray(image)    # numpy 배열을 PIL 이미지로 변환
      image = image.resize((300, 300))  # 사이즈 조정
      
      # BytesIO 스트림 생성
      image_stream = BytesIO()
      file_type = filename.split('.')[-1].lower()
      image.save(image_stream, format=file_type)  # PNG 형식으로 저장
      image_stream.seek(0)  # 파일 포인터를 처음으로 이동

      # UploadFile 객체 생성 및 반환
      upload_file = UploadFile(filename=filename, file=image_stream)
      upload_file._headers = Headers({"content-type": f"image/{file_type}"})
      return upload_file

# 싱글톤 인스턴스 생성
modelService = ModelService()