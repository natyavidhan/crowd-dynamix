"""
Crowd Video Processor

Main processor that combines stream capture, people detection,
and movement analysis into a unified pipeline.
"""

import cv2
import numpy as np
import time
import json
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from queue import Queue

from .stream_capture import VideoStreamCapture
from .people_detector import PeopleDetector, PeopleDetectorLite, Detection
from .movement_analyzer import MovementAnalyzer, CrowdFlowAnalysis, CrowdDensityAnalyzer


@dataclass
class ProcessingResult:
    """Result from processing a single frame."""
    timestamp: float
    frame_number: int
    people_count: int
    detections: List[Detection]
    flow_analysis: CrowdFlowAnalysis
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "people_count": self.people_count,
            "detections": [
                {
                    "bbox": d.bbox,
                    "confidence": d.confidence,
                    "center": d.center,
                }
                for d in self.detections
            ],
            "flow": self.flow_analysis.to_dict(),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class CrowdVideoProcessor:
    """
    Main processor for crowd analysis from video streams.
    
    Combines:
    - Video stream capture from IP webcam
    - People detection using YOLO
    - Movement analysis using optical flow
    """
    
    def __init__(
        self,
        stream_url: str = "http://10.203.120.62:8080/video",
        use_yolo: bool = True,
        yolo_model: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the crowd video processor.
        
        Args:
            stream_url: URL of the IP webcam video stream
            use_yolo: Whether to use YOLO (True) or HOG (False)
            yolo_model: YOLO model to use
            confidence_threshold: Detection confidence threshold
        """
        self.stream_url = stream_url
        
        # Initialize components
        self.stream = VideoStreamCapture(stream_url)
        
        if use_yolo:
            self.detector = PeopleDetector(
                model_name=yolo_model,
                confidence_threshold=confidence_threshold,
            )
        else:
            self.detector = PeopleDetectorLite()
        
        self.movement_analyzer = MovementAnalyzer()
        self.density_analyzer = CrowdDensityAnalyzer()
        
        self._frame_count = 0
        self._start_time = None
        self._running = False
        self._results_queue: Queue = Queue()
        
        # Callbacks
        self._on_result: Optional[Callable[[ProcessingResult], None]] = None
        self._on_frame: Optional[Callable[[np.ndarray, ProcessingResult], None]] = None
    
    def set_result_callback(self, callback: Callable[[ProcessingResult], None]):
        """Set callback for processing results."""
        self._on_result = callback
    
    def set_frame_callback(self, callback: Callable[[np.ndarray, ProcessingResult], None]):
        """Set callback for annotated frames."""
        self._on_frame = callback
    
    def process_frame(self, frame: np.ndarray) -> ProcessingResult:
        """
        Process a single frame.
        
        Args:
            frame: BGR image
            
        Returns:
            ProcessingResult with analysis data
        """
        start_time = time.time()
        
        # Detect people
        detections = self.detector.detect(frame)
        
        # Analyze movement
        timestamp = time.time() - (self._start_time or time.time())
        flow_analysis = self.movement_analyzer.analyze_frame(
            frame,
            detections,
            timestamp=timestamp,
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        self._frame_count += 1
        
        return ProcessingResult(
            timestamp=timestamp,
            frame_number=self._frame_count,
            people_count=len(detections),
            detections=detections,
            flow_analysis=flow_analysis,
            processing_time_ms=processing_time,
        )
    
    def process_and_visualize(
        self,
        frame: np.ndarray,
        show_boxes: bool = True,
        show_flow: bool = True,
        show_density: bool = False,
    ) -> tuple[np.ndarray, ProcessingResult]:
        """
        Process frame and create visualization.
        
        Args:
            frame: BGR image
            show_boxes: Draw bounding boxes
            show_flow: Draw flow vectors
            show_density: Show density heatmap
            
        Returns:
            Tuple of (annotated_frame, result)
        """
        result = self.process_frame(frame)
        
        vis = frame.copy()
        
        # Draw detections
        if show_boxes:
            for det in result.detections:
                cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
                cv2.circle(vis, det.center, 4, (0, 0, 255), -1)
        
        # Draw flow vectors
        if show_flow:
            vis = self.movement_analyzer.visualize_flow(
                vis,
                result.flow_analysis.flow_vectors,
            )
        
        # Overlay density map
        if show_density:
            vis = self.density_analyzer.visualize_density(
                vis,
                result.detections,
                alpha=0.3,
            )
        
        # Draw info overlay
        self._draw_info_overlay(vis, result)
        
        return vis, result
    
    def _draw_info_overlay(self, frame: np.ndarray, result: ProcessingResult):
        """Draw information overlay on frame."""
        # Background rectangle
        cv2.rectangle(frame, (5, 5), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (300, 100), (255, 255, 255), 1)
        
        # Text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 25
        
        cv2.putText(frame, f"People: {result.people_count}", (10, y), font, 0.6, (0, 255, 255), 2)
        y += 20
        
        cv2.putText(
            frame,
            f"Speed: {result.flow_analysis.avg_speed:.1f} px/frame",
            (10, y), font, 0.5, (255, 255, 255), 1
        )
        y += 18
        
        cv2.putText(
            frame,
            f"Direction: {result.flow_analysis.dominant_direction} ({result.flow_analysis.avg_direction:.0f}°)",
            (10, y), font, 0.5, (255, 255, 255), 1
        )
        y += 18
        
        cv2.putText(
            frame,
            f"Processing: {result.processing_time_ms:.1f}ms",
            (10, y), font, 0.5, (200, 200, 200), 1
        )
    
    def run_live(
        self,
        display: bool = True,
        max_frames: Optional[int] = None,
        output_path: Optional[str] = None,
    ):
        """
        Run live processing on the video stream.
        
        Args:
            display: Show live video window
            max_frames: Maximum frames to process (None for infinite)
            output_path: Path to save output video
        """
        print(f"Connecting to stream: {self.stream_url}")
        
        if not self.stream.connect():
            print("Failed to connect to stream")
            return
        
        print("Initializing detector...")
        if hasattr(self.detector, 'initialize'):
            self.detector.initialize()
        
        self._start_time = time.time()
        self._running = True
        
        # Video writer for output
        writer = None
        
        print("Starting live processing... Press 'q' to quit")
        
        try:
            while self._running:
                if max_frames and self._frame_count >= max_frames:
                    break
                
                ret, frame = self.stream.read_frame()
                
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Process and visualize
                vis, result = self.process_and_visualize(frame)
                
                # Initialize video writer
                if output_path and writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, 20, (w, h))
                
                if writer:
                    writer.write(vis)
                
                # Callbacks
                if self._on_result:
                    self._on_result(result)
                
                if self._on_frame:
                    self._on_frame(vis, result)
                
                # Display
                if display:
                    cv2.imshow("Crowd Analysis", vis)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Print periodic stats
                if self._frame_count % 30 == 0:
                    elapsed = time.time() - self._start_time
                    fps = self._frame_count / elapsed if elapsed > 0 else 0
                    print(f"Frame {self._frame_count}: {result.people_count} people, "
                          f"{result.flow_analysis.dominant_direction}, "
                          f"FPS: {fps:.1f}")
        
        finally:
            self._running = False
            self.stream.disconnect()
            
            if writer:
                writer.release()
            
            if display:
                cv2.destroyAllWindows()
            
            print(f"\nProcessed {self._frame_count} frames")
    
    def run_async(self):
        """Start processing in background thread."""
        self._running = True
        thread = threading.Thread(target=self._async_loop, daemon=True)
        thread.start()
        return thread
    
    def _async_loop(self):
        """Background processing loop."""
        self.stream.connect()
        self.stream.start_async()
        self._start_time = time.time()
        
        if hasattr(self.detector, 'initialize'):
            self.detector.initialize()
        
        while self._running:
            frame = self.stream.get_frame_async(timeout=1.0)
            
            if frame is None:
                continue
            
            result = self.process_frame(frame)
            self._results_queue.put(result)
            
            if self._on_result:
                self._on_result(result)
        
        self.stream.stop_async()
        self.stream.disconnect()
    
    def stop(self):
        """Stop processing."""
        self._running = False
    
    def get_result(self, timeout: float = 1.0) -> Optional[ProcessingResult]:
        """Get result from async processing queue."""
        try:
            return self._results_queue.get(timeout=timeout)
        except:
            return None
    
    def reset(self):
        """Reset processor state."""
        self._frame_count = 0
        self._start_time = None
        self.movement_analyzer.reset()


def process_video_file(
    video_path: str,
    output_path: Optional[str] = None,
    use_yolo: bool = True,
) -> List[ProcessingResult]:
    """
    Process a video file and return results.
    
    Args:
        video_path: Path to input video
        output_path: Path for output video (optional)
        use_yolo: Use YOLO detector
        
    Returns:
        List of processing results
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Create processor (without stream)
    if use_yolo:
        detector = PeopleDetector()
        detector.initialize()
    else:
        detector = PeopleDetectorLite()
    
    analyzer = MovementAnalyzer()
    results = []
    
    # Video writer
    writer = None
    if output_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        detections = detector.detect(frame)
        
        # Analyze
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        flow = analyzer.analyze_frame(frame, detections, timestamp)
        
        frame_count += 1
        proc_time = (time.time() - start_time) * 1000 / frame_count
        
        result = ProcessingResult(
            timestamp=timestamp,
            frame_number=frame_count,
            people_count=len(detections),
            detections=detections,
            flow_analysis=flow,
            processing_time_ms=proc_time,
        )
        results.append(result)
        
        # Write annotated frame
        if writer:
            vis = frame.copy()
            for det in detections:
                cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
            cv2.putText(vis, f"People: {len(detections)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            writer.write(vis)
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    if writer:
        writer.release()
    
    print(f"Processed {frame_count} frames total")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crowd Video Processor")
    parser.add_argument("--url", default="http://10.203.120.62:8080/video",
                       help="Video stream URL")
    parser.add_argument("--video", help="Video file path (instead of stream)")
    parser.add_argument("--output", help="Output video path")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't show display window")
    parser.add_argument("--use-hog", action="store_true",
                       help="Use HOG detector instead of YOLO")
    parser.add_argument("--max-frames", type=int,
                       help="Maximum frames to process")
    
    args = parser.parse_args()
    
    if args.video:
        # Process video file
        results = process_video_file(
            args.video,
            output_path=args.output,
            use_yolo=not args.use_hog,
        )
        print(f"\nResults summary:")
        print(f"  Total frames: {len(results)}")
        if results:
            avg_people = sum(r.people_count for r in results) / len(results)
            print(f"  Average people: {avg_people:.1f}")
    else:
        # Live stream processing
        processor = CrowdVideoProcessor(
            stream_url=args.url,
            use_yolo=not args.use_hog,
        )
        
        processor.run_live(
            display=not args.no_display,
            max_frames=args.max_frames,
            output_path=args.output,
        )
