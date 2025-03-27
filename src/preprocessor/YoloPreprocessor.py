import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

class YoloPreprocessor:
    def __init__(self):
        self.model = YOLO('yolov8m.pt')

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
        
        count = len(extracted_obj)

        return extracted_obj, results, count
    
      # 색상 비율 분석
    def analyze_color_ratio(self, cropped_img):
        bgr = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        red_lower1 = np.array([0, 70, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 70, 50])
        red_upper2 = np.array([179, 255, 255])
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        brown_lower = np.array([10, 50, 20])
        brown_upper = np.array([25, 255, 200])

        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, red_lower1, red_upper1),
            cv2.inRange(hsv, red_lower2, red_upper2)
        )
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)

        total_pixels = hsv.shape[0] * hsv.shape[1]
        red_ratio = round((cv2.countNonZero(red_mask) / total_pixels) * 100, 1)
        green_ratio = round((cv2.countNonZero(green_mask) / total_pixels) * 100, 1)
        brown_ratio = round((cv2.countNonZero(brown_mask) / total_pixels) * 100, 1)

        return {
            "red": red_ratio,
            "green": green_ratio,
            "brown": brown_ratio
        }

    # 숙성도 추정 (빨강 비율 기반)
    def estimate_ripeness(self, color_ratio):
        return round(color_ratio["red"] / 100, 2)

    # 숙성도 기반 품질 등급 분류
    def classify_grade(self, ripeness):
        if ripeness >= 0.85:
            return "특"
        elif ripeness >= 0.65:
            return "상"
        else:
            return "보통"

    # 전체 분석 파이프라인
    def analyze_apples(self, img):
        cropped_objs, _, results, count = self.yolo_detect(img)
        apples = []

        for crop in cropped_objs :
            color_ratio = self.analyze_color_ratio(crop)
            ripeness = self.estimate_ripeness(color_ratio)
            grade = self.classify_grade(ripeness)

            apples.append({
                "color_ratio": color_ratio,
                "ripeness": ripeness,
                "grade": grade
            })

        return {
            "count": count,
            "apples": apples,
            "results": results
        }
    
    

yoloPreprocessor = YoloPreprocessor()