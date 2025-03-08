import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import cv2
import numpy as np
from src.services.ModelService import modelService  # 싱글톤 인스턴스를 import

app = FastAPI()

class AIRequest(BaseModel) :
    data : list # spring 에서 받은 예제 데이터

class AIResponse(BaseModel) :
    result : str # 예측 결과

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/analyze")
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
        result = modelService.analyze(image)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# uvicorn 실행 환경 설정
if __name__ == "__main__":
    uvicorn.run("main:app", host= "0.0.0.0", port=9090, reload=True)