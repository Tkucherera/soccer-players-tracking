from ultralytics import YOLO
from dataclasses import dataclass
import numpy as np
import logging
from .models import Track, Detection, DetectionOutput, CLASS_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

class Detector:

    def __init__(self, config: dict):
        self.model = YOLO(config["model_path"])
        logging.info(f"Model loaded from {config['model_path']}")
        self.conf_threshold = config.get("confidence_threshold", 0.5)
        self.imgsz = config.get("imgsz", (640, 640))
        self.device = config.get("device", "cpu")

    def run(self, frame: np.ndarray, pitch_mask: np.ndarray) -> DetectionOutput:
        results = self.model.predict(
            frame, 
            conf=self.conf_threshold, 
            imgsz=self.imgsz, 
            device=self.device
        )

        objects, ball = [], None

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            foot_x, foot_y = (x1 + x2) // 2, y2
            if pitch_mask[foot_y, foot_x] == 0:
                continue  # Skip detections outside the pitch area

            class_id = int(box.cls[0])
            det = Detection(
                bbox=tuple(box.xyxy[0].int().tolist()),
                class_id=class_id,
                class_name=CLASS_NAMES.get(class_id, "unknown"),
                confidence=float(box.conf),
            )
            if det.class_name == "ball":
                if ball is None or det.confidence > ball.confidence:    
                    ball = det
            else:
                objects.append(det)
        return DetectionOutput(objects=objects, ball=ball)
    
    @staticmethod
    def footpoint(bbox: tuple) -> tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)
    






    
    
