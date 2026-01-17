"""
People Detection Module

Uses YOLOv8 for detecting and counting people in video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class Detection:
    """Represents a detected person."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[int, int]
    area: int
    
    @property
    def x1(self) -> int:
        return self.bbox[0]
    
    @property
    def y1(self) -> int:
        return self.bbox[1]
    
    @property
    def x2(self) -> int:
        return self.bbox[2]
    
    @property
    def y2(self) -> int:
        return self.bbox[3]
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1


class PeopleDetector:
    """
    Detects and counts people in video frames using YOLOv8.
    """
    
    # COCO class ID for person
    PERSON_CLASS_ID = 0
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",  # nano model for speed
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,  # None for auto, "cpu", "cuda", etc.
    ):
        """
        Initialize the people detector.
        
        Args:
            model_name: YOLOv8 model to use (yolov8n/s/m/l/x)
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None
        self._initialized = False
        
    def initialize(self):
        """Load the YOLO model."""
        if self._initialized:
            return
        
        try:
            from ultralytics import YOLO
            
            print(f"Loading YOLO model: {self.model_name}")
            self._model = YOLO(self.model_name)
            
            if self.device:
                self._model.to(self.device)
            
            self._initialized = True
            print("Model loaded successfully")
            
        except ImportError:
            raise ImportError(
                "ultralytics package not installed. "
                "Run: pip install ultralytics"
            )
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect people in a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of Detection objects for detected people
        """
        if not self._initialized:
            self.initialize()
        
        # Run inference
        results = self._model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
            
            for i, (box, cls, conf) in enumerate(zip(
                boxes.xyxy.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy()
            )):
                # Only detect people (class 0 in COCO)
                if int(cls) != self.PERSON_CLASS_ID:
                    continue
                
                if conf < self.confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    center=(center_x, center_y),
                    area=area,
                )
                detections.append(detection)
        
        return detections
    
    def count_people(self, frame: np.ndarray) -> int:
        """
        Count the number of people in a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Number of detected people
        """
        detections = self.detect(frame)
        return len(detections)
    
    def detect_with_visualization(
        self,
        frame: np.ndarray,
        draw_boxes: bool = True,
        draw_centers: bool = True,
        draw_count: bool = True,
    ) -> Tuple[np.ndarray, List[Detection], int]:
        """
        Detect people and draw visualizations on the frame.
        
        Args:
            frame: BGR image as numpy array
            draw_boxes: Whether to draw bounding boxes
            draw_centers: Whether to draw center points
            draw_count: Whether to draw people count
            
        Returns:
            Tuple of (annotated_frame, detections, count)
        """
        detections = self.detect(frame)
        count = len(detections)
        
        # Create a copy for visualization
        vis_frame = frame.copy()
        
        for det in detections:
            if draw_boxes:
                cv2.rectangle(
                    vis_frame,
                    (det.x1, det.y1),
                    (det.x2, det.y2),
                    (0, 255, 0),  # Green
                    2
                )
                
                # Draw confidence
                label = f"{det.confidence:.2f}"
                cv2.putText(
                    vis_frame,
                    label,
                    (det.x1, det.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            if draw_centers:
                cv2.circle(
                    vis_frame,
                    det.center,
                    5,
                    (0, 0, 255),  # Red
                    -1
                )
        
        if draw_count:
            cv2.putText(
                vis_frame,
                f"People: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),  # Yellow
                2
            )
        
        return vis_frame, detections, count
    
    def get_detection_stats(self, detections: List[Detection]) -> Dict[str, Any]:
        """
        Get statistics about detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Dictionary with detection statistics
        """
        if not detections:
            return {
                "count": 0,
                "avg_confidence": 0.0,
                "avg_area": 0,
                "centers": [],
            }
        
        confidences = [d.confidence for d in detections]
        areas = [d.area for d in detections]
        centers = [d.center for d in detections]
        
        return {
            "count": len(detections),
            "avg_confidence": np.mean(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "avg_area": int(np.mean(areas)),
            "min_area": min(areas),
            "max_area": max(areas),
            "centers": centers,
        }


class PeopleDetectorLite:
    """
    Lightweight people detector using HOG + SVM.
    Faster but less accurate than YOLO.
    No GPU required.
    """
    
    def __init__(
        self,
        win_stride: Tuple[int, int] = (8, 8),
        padding: Tuple[int, int] = (8, 8),
        scale: float = 1.05,
    ):
        """
        Initialize the HOG detector.
        
        Args:
            win_stride: Window stride for detection
            padding: Padding for detection
            scale: Scale factor for multi-scale detection
        """
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect people using HOG + SVM.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of Detection objects
        """
        # Resize for faster detection
        scale_factor = 1.0
        if frame.shape[1] > 800:
            scale_factor = 800 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        
        # Detect people
        boxes, weights = self._hog.detectMultiScale(
            frame,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
        )
        
        detections = []
        
        for (x, y, w, h), weight in zip(boxes, weights):
            # Scale back to original size
            x = int(x / scale_factor)
            y = int(y / scale_factor)
            w = int(w / scale_factor)
            h = int(h / scale_factor)
            
            detection = Detection(
                bbox=(x, y, x + w, y + h),
                confidence=float(weight),
                center=(x + w // 2, y + h // 2),
                area=w * h,
            )
            detections.append(detection)
        
        return detections
    
    def count_people(self, frame: np.ndarray) -> int:
        """Count people in frame."""
        return len(self.detect(frame))


if __name__ == "__main__":
    # Test with a sample image
    print("Testing PeopleDetector...")
    
    # Create a dummy test image
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test HOG detector (doesn't require download)
    print("\nTesting HOG detector...")
    hog_detector = PeopleDetectorLite()
    detections = hog_detector.detect(test_frame)
    print(f"HOG detections: {len(detections)}")
    
    # Test YOLO detector
    print("\nTesting YOLO detector...")
    try:
        yolo_detector = PeopleDetector()
        yolo_detector.initialize()
        detections = yolo_detector.detect(test_frame)
        print(f"YOLO detections: {len(detections)}")
    except Exception as e:
        print(f"YOLO test failed: {e}")
