from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2
from app.services.crop_service import analyze_crop

router = APIRouter()

@router.post("/analyze")
async def analyze_crop_api(file: UploadFile = File(...)):
   # 이미지 읽기
   image_bytes = await file.read()
   np_arr = np.frombuffer(image_bytes, np.uint8)
   img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

   # 작물 상태 분석
   result = analyze_crop(img)

   return {"status": result}