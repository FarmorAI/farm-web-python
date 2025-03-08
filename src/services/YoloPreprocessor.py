import numpy as np
from PIL import Image
from ultralytics import YOLO


class YoloPreprocessor:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    # YOLO 모델로 객체 인식 수행
    def yolo_detect(self, img, target_class_id=47):

        # 이미지가 PIL 이미지인 경우 numpy 배열로 변환
        img_np = np.array(img) if isinstance(img, Image.Image) else img
        results = self.model.predict(img_np)

        extracted_obj = []
        # 결과에서 바운딩 박스 정보 가져오기
        for result in results:
            # 바운딩 박스 정보가 사과(47)인 경우에만 박스 추출
            for box, cls in zip(result.boxes, result.boxes.cls):
                if int(cls) == target_class_id:
                    # 이미지에서 객체 영역 자르기
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # xyxy 형식의 좌표
                    cropped_obj = img_np[y1:y2, x1:x2]
                    extracted_obj.append(cropped_obj)

        return extracted_obj, results

yoloPreprocessor = YoloPreprocessor()