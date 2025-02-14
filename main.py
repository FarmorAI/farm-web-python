from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
from ultralytics import YOLO
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
from fastapi.responses import FileResponse
from fastapi.responses import Response


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


model = YOLO("yolov8n.pt")  # 사전 학습된 YOLOv8 모델 로드

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # 파일을 읽고 OpenCV 형식으로 변환
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        img_np = np.array(image)
        print(file) # 스프링에서 받은 파일 정보 출력

        # YOLOv8 모델로 객체 탐지 수행
        results = model(img_np)

        # 객체 탐지 정보 리스트
        detections_info = []
        
        for result in results:
            detections = result.boxes.data.tolist()
            print(f"🔍 탐지된 객체 수: {len(detections)}")

            # 객체 정보 저장
            for det in detections:
                x1, y1, x2, y2, conf, cls = map(int, det[:6])
                class_name = model.names[cls]  # 클래스 이름 가져오기

                detections_info.append({
                    "class": class_name,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

                # 객체 탐지된 이미지에 바운딩 박스 및 텍스트 추가
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_np, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 탐지된 이미지 시각화
            image_with_boxes = result.plot()

            image_result = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
            # 이미지 저장
            # cv2.imwrite(file.filename, image_result)
            
            # 이미지 변환 및 JPEG 인코딩
            result_img = Image.fromarray(image_result[:, :, ::-1])
            img_io = BytesIO()
            result_img.save(img_io, format="JPEG")
            img_io.seek(0)

        return Response(content=img_io.getvalue(), media_type="image/jpeg",status_code=200)  
        # JSON 형태로 객체 탐지 정보 반환
        return {
            "message": "객체 탐지 완료",
            "detections": detections_info
        }

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return Response(content=f"서버 오류: {str(e)}", media_type="text/plain", status_code=500)

if __name__ == "__main__":

    uvicorn.run(app, host= "127.0.0.1", port=8000)