import uvicorn
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, HTTPException

import cv2
import numpy as np
from src.services.ModelService import modelService  # 싱글톤 인스턴스를 import
from src.services.S3Service import S3Service
from src.repository.MySqlRepository import mysql_repository
import scipy.stats as stats

app = FastAPI()
s3_service = S3Service()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# return : JSONResponse
@app.post("/analyze")
async def analyze_api(file: UploadFile = File(...)):
    # 이미지 형식 검증
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 타입만 허용")
    
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # 예측 분석 및 S3에 파일 업로드
        processed_image, quality_metrics = await modelService.analyze(image)
        s3_result = await s3_service.upload_image(processed_image)

        # JSON 전달
        return JSONResponse(content={
            "message": "success",
            "image_url": s3_result["file_url"],
            "quality_metrics": quality_metrics
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 메인페이지 그래프
@app.get("/fit_distribution")
async def fit_distribution():
    df = mysql_repository.get_data()
    
    response_data = []
    for grade in [0, 1, 2]:
        data = df[df["cate3"] == grade]["weight"]
        mu, std = np.mean(data), np.std(data)
        x = np.linspace(mu - 3*std, mu + 3*std, 100)
        y = stats.norm.pdf(x, mu, std)
        
        # ✅ 무게(weight) 기준 상위 25% 값 계산
        threshold_75 = np.percentile(data, 75)
        mean_value = mu  # ✅ 평균값 추가

        response_data.append({
            "grade": grade,
            "mu": mu,
            "std": std,
            "mean": mean_value,  # ✅ 평균값 추가
            "x": x.tolist(),
            "y": y.tolist(),
            "threshold_75": threshold_75  # ✅ 무게 기준 상위 25% 값 추가
        })
    
    return {"distribution": response_data}

# uvicorn 실행 환경 설정
if __name__ == "__main__":
    uvicorn.run("main:app", host= "0.0.0.0", port=9090, reload=True)


# 응답 결과 예시
# {
#     "message": "success",
#     "image_url": "https://farmorai-bucket00.s3.ap-northeast-2.amazonaws.com/images/324f9fb1-9a56-4304-baad-c40d443db886.png",
#     "quality_metrics": {
#         "success": true,
#         "quality": {
#         "특": 0.36,
#         "상": 0,
#         "보통": 0.64
#         }
#     }
# }