import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from src.services.ModelService import modelService
from src.services.S3Service import s3_service

router = APIRouter()

@router.post("/analyze")  # return : JSONResponse
async def analyze(file: UploadFile = File(...)):
   
   # 이미지 형식 검증
   if not file.content_type.startswith("image/"):
      raise HTTPException(status_code=400, detail="이미지 타입만 허용")
   
   try:
      # 이미지 읽기
      image_bytes = await file.read()
      image_np = np.frombuffer(image_bytes, np.uint8)
      image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

      # 모델 예측 분석
      processed_image, quality_metrics, apple_count, apples_info = await modelService.analyze(image)
      
      # S3에 파일 업로드
      s3_result = await s3_service.upload_image(processed_image)

      return JSONResponse(content={
            "message": "success",
            "image_url": s3_result["file_url"],
            "quality": quality_metrics["quality"],
            "count": apple_count,
            "apples" : apples_info
      })
   except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
   


# 응답 결과 예시
# {
#     "message": "success",
#     "image_url": "https://farmorai-bucket00.s3.ap-northeast-2.amazonaws.com/images/324f9fb1-9a56-4304-baad-c40d443db886.png",
#     "quality": {
#         "특": 0.36,
#         "상": 0,
#         "보통": 0.64
#     }
#     "count": 3,
#     "apples": [
#     {
#       "color_ratio": { "red": 85.1, "green": 2.4, "brown": 1.2 },
#       "ripeness": 0.85,
#       "grade": "특"
#     },
#     {
#       "color_ratio": { "빨강": 68.7, "초록": 5.5, "갈색": 3.1 },
#       "ripeness": 0.69,
#       "grade": "상"
#     },
#     {
#       "color_ratio": { "빨강": 50.3, "초록": 10.0, "갈색": 5.0 },
#       "ripeness": 0.5,
#       "grade": "보통"
#     }
#   ]
# }