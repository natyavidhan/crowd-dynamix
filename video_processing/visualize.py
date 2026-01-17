"""
Live Visualization for Crowd Analysis

Displays real-time video feed with:
- People detection bounding boxes
- Movement flow vectors
- Crowd density heatmap
- Statistics overlay
"""

import cv2
import numpy as np
import time
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_processing.stream_capture import VideoStreamCapture
from video_processing.people_detector import PeopleDetector, PeopleDetectorLite
from video_processing.movement_analyzer import MovementAnalyzer, CrowdDensityAnalyzer


class LiveVisualizer:
    """Real-time visualization of crowd analysis."""
    
    def __init__(
        self,
        stream_url: str = "http://10.203.120.62:8080/video",
        use_yolo: bool = False,  # Default to HOG for speed
        window_name: str = "Crowd Analysis - Live",
    ):
        self.stream_url = stream_url
        self.window_name = window_name
        
        # Components
        self.stream = VideoStreamCapture(stream_url)
        
        if use_yolo:
            try:
                self.detector = PeopleDetector(confidence_threshold=0.5)
                self.detector.initialize()
                print("Using YOLO detector")
            except:
                print("YOLO not available, falling back to HOG")
                self.detector = PeopleDetectorLite()
        else:
            self.detector = PeopleDetectorLite()
            print("Using HOG detector")
        
        self.movement_analyzer = MovementAnalyzer()
        self.density_analyzer = CrowdDensityAnalyzer(grid_size=(16, 12))
        
        # State
        self.frame_count = 0
        self.start_time = None
        self.show_boxes = True
        self.show_flow = True
        self.show_density = False
        self.show_stats = True
        self.paused = False
        
        # Stats history for smoothing
        self.people_history = []
        self.speed_history = []
        self.history_size = 30
        
    def run(self):
        """Run the live visualization."""
        print(f"Connecting to {self.stream_url}...")
        
        if not self.stream.connect():
            print("Failed to connect to stream!")
            return
        
        print("Connected! Starting visualization...")
        print("\nControls:")
        print("  b - Toggle bounding boxes")
        print("  f - Toggle flow vectors")
        print("  d - Toggle density heatmap")
        print("  s - Toggle statistics")
        print("  p - Pause/Resume")
        print("  r - Reset analyzer")
        print("  q - Quit")
        print()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        self.start_time = time.time()
        
        try:
            while True:
                if not self.paused:
                    ret, frame = self.stream.read_frame()
                    
                    if not ret:
                        print("Failed to read frame, reconnecting...")
                        self.stream.disconnect()
                        time.sleep(1)
                        self.stream.connect()
                        continue
                    
                    # Process frame
                    vis_frame = self.process_and_visualize(frame)
                    
                    # Show frame
                    cv2.imshow(self.window_name, vis_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('b'):
                    self.show_boxes = not self.show_boxes
                    print(f"Bounding boxes: {'ON' if self.show_boxes else 'OFF'}")
                elif key == ord('f'):
                    self.show_flow = not self.show_flow
                    print(f"Flow vectors: {'ON' if self.show_flow else 'OFF'}")
                elif key == ord('d'):
                    self.show_density = not self.show_density
                    print(f"Density heatmap: {'ON' if self.show_density else 'OFF'}")
                elif key == ord('s'):
                    self.show_stats = not self.show_stats
                    print(f"Statistics: {'ON' if self.show_stats else 'OFF'}")
                elif key == ord('p'):
                    self.paused = not self.paused
                    print(f"{'PAUSED' if self.paused else 'RESUMED'}")
                elif key == ord('r'):
                    self.movement_analyzer.reset()
                    self.people_history.clear()
                    self.speed_history.clear()
                    print("Analyzer reset")
        
        finally:
            self.stream.disconnect()
            cv2.destroyAllWindows()
            print("\nVisualization stopped")
    
    def process_and_visualize(self, frame: np.ndarray) -> np.ndarray:
        """Process frame and create visualization."""
        self.frame_count += 1
        process_start = time.time()
        
        # Resize for faster processing (keep aspect ratio)
        scale = 640 / frame.shape[1]
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Detect people
        detections = self.detector.detect(small_frame)
        
        # Scale detections back to original size
        for det in detections:
            det.bbox = tuple(int(v / scale) for v in det.bbox)
            det.center = (int(det.center[0] / scale), int(det.center[1] / scale))
        
        # Analyze movement
        timestamp = time.time() - self.start_time
        flow_analysis = self.movement_analyzer.analyze_frame(frame, detections, timestamp)
        
        process_time = (time.time() - process_start) * 1000
        
        # Update history
        self.people_history.append(len(detections))
        self.speed_history.append(flow_analysis.avg_speed)
        if len(self.people_history) > self.history_size:
            self.people_history.pop(0)
            self.speed_history.pop(0)
        
        # Create visualization
        vis = frame.copy()
        
        # Density heatmap (render first, as background)
        if self.show_density:
            vis = self.density_analyzer.visualize_density(vis, detections, alpha=0.4)
        
        # Bounding boxes
        if self.show_boxes:
            for det in detections:
                # Box
                cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
                
                # Center point
                cv2.circle(vis, det.center, 5, (0, 0, 255), -1)
                
                # Confidence label
                label = f"{det.confidence:.0%}"
                cv2.putText(vis, label, (det.x1, det.y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Flow vectors
        if self.show_flow and flow_analysis.flow_vectors:
            for vec in flow_analysis.flow_vectors:
                # Scale arrow for visibility
                end_x = int(vec.start[0] + vec.dx * 3)
                end_y = int(vec.start[1] + vec.dy * 3)
                
                cv2.arrowedLine(vis, vec.start, (end_x, end_y),
                               (255, 0, 255), 2, tipLength=0.3)
        
        # Statistics overlay
        if self.show_stats:
            self.draw_stats_overlay(vis, detections, flow_analysis, process_time)
        
        return vis
    
    def draw_stats_overlay(self, frame, detections, flow_analysis, process_time):
        """Draw statistics panel on frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (10, 10), (320, 200), (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "CROWD ANALYSIS", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y = 60
        line_height = 25
        
        # People count
        avg_people = np.mean(self.people_history) if self.people_history else 0
        cv2.putText(frame, f"People: {len(detections)} (avg: {avg_people:.1f})",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += line_height
        
        # Movement speed
        avg_speed = np.mean(self.speed_history) if self.speed_history else 0
        cv2.putText(frame, f"Speed: {flow_analysis.avg_speed:.1f} px/f (avg: {avg_speed:.1f})",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += line_height
        
        # Direction
        cv2.putText(frame, f"Direction: {flow_analysis.dominant_direction} ({flow_analysis.avg_direction:.0f} deg)",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += line_height
        
        # Flow vectors
        cv2.putText(frame, f"Movement vectors: {len(flow_analysis.flow_vectors)}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += line_height
        
        # FPS / Processing time
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f} | Process: {process_time:.0f}ms",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += line_height
        
        # Frame count
        cv2.putText(frame, f"Frame: {self.frame_count}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Direction compass (bottom right)
        self.draw_compass(frame, flow_analysis.avg_direction, w - 80, h - 80, 50)
    
    def draw_compass(self, frame, direction, cx, cy, radius):
        """Draw a compass showing crowd direction."""
        # Background circle
        cv2.circle(frame, (cx, cy), radius, (50, 50, 50), -1)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)
        
        # Direction labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "N", (cx - 5, cy - radius - 5), font, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "S", (cx - 5, cy + radius + 15), font, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "E", (cx + radius + 5, cy + 5), font, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "W", (cx - radius - 15, cy + 5), font, 0.4, (255, 255, 255), 1)
        
        # Direction arrow
        # Convert from image coordinates (0=right, 90=down) to compass (0=up)
        compass_angle = (direction - 90) % 360
        angle_rad = np.radians(compass_angle)
        
        end_x = int(cx + radius * 0.7 * np.sin(angle_rad))
        end_y = int(cy - radius * 0.7 * np.cos(angle_rad))
        
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 255, 255), 3, tipLength=0.4)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Crowd Analysis Visualization")
    parser.add_argument("--url", default="http://10.203.120.62:8080/video",
                       help="Video stream URL")
    parser.add_argument("--yolo", action="store_true",
                       help="Use YOLO detector (requires ultralytics)")
    
    args = parser.parse_args()
    
    visualizer = LiveVisualizer(
        stream_url=args.url,
        use_yolo=args.yolo,
    )
    
    visualizer.run()


if __name__ == "__main__":
    main()
