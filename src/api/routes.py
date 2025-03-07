from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2
from src.services.ModelService import ModelService

router = APIRouter()

@router.post("/analyze")
async def analyze_api(file: UploadFile = File(...)):
   # 허용된 이미지 형식 검증
   if not file.content_type.startswith("image/"):
      raise HTTPException(status_code=400, detail="Invalid file type")
   
   try:
      # 이미지 읽기
      image_bytes = await file.read()
      image_np = np.frombuffer(image_bytes, np.uint8)
      image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

      if image is None:
         raise HTTPException(status_code=400, detail="Convert file Error")
      
      # 예측 분석
      result = ModelService.analyze(image)
      return result
   
   except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))