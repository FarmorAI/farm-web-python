import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

class YoloPreprocessor:
    def __init__(self):
        self.model = YOLO('yolov8m.pt')

    def yolo_detect(self, img, target_class_id=47):
        img_np = np.array(img) if isinstance(img, Image.Image) else img
        results = self.model.predict(img_np)

        extracted_obj = []
        for result in results:
            for box, cls in zip(result.boxes, result.boxes.cls):
                if int(cls) == target_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_obj = img_np[y1:y2, x1:x2]
                    if cropped_obj.size == 0:
                        continue
                    extracted_obj.append(cropped_obj)

        count = len(extracted_obj)
        return extracted_obj, results, count

    def analyze_color_ratio(self, cropped_img):
        # FastAPI는 BGR로 들어오기 때문에 RGB2BGR 변환은 생략
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

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

        total_pixels = hsv.shape[0] * hsv.shape[1] or 1  # 0 나눗셈 방지
        red_ratio = round((cv2.countNonZero(red_mask) / total_pixels) * 100, 1)
        green_ratio = round((cv2.countNonZero(green_mask) / total_pixels) * 100, 1)
        brown_ratio = round((cv2.countNonZero(brown_mask) / total_pixels) * 100, 1)

        return {
            "red": red_ratio,
            "green": green_ratio,
            "brown": brown_ratio
        }

    def estimate_ripeness(self, color_ratio):
        return round(color_ratio["red"] / 100, 2)

    def classify_grade(self, ripeness):
        if ripeness >= 0.85:
            return "특"
        elif ripeness >= 0.65:
            return "상"
        else:
            return "보통"

    def analyze_apples(self, img):
        cropped_objs, results, count = self.yolo_detect(img)
        apples = []

        for crop in cropped_objs:
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