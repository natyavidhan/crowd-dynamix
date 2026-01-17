"""
Movement Analysis Module

Calculates crowd speed and direction using optical flow
and detection tracking between frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import math

from .people_detector import Detection


@dataclass
class MovementVector:
    """Represents movement of a detected person."""
    start: Tuple[int, int]  # Previous position
    end: Tuple[int, int]  # Current position
    dx: float  # Horizontal displacement
    dy: float  # Vertical displacement
    speed: float  # Pixels per frame
    direction: float  # Angle in degrees (0=right, 90=down)
    
    @property
    def magnitude(self) -> float:
        """Movement magnitude (same as speed)."""
        return self.speed


@dataclass 
class CrowdFlowAnalysis:
    """Analysis results for crowd movement."""
    timestamp: float
    people_count: int
    avg_speed: float  # Average speed in pixels/frame
    avg_direction: float  # Average direction in degrees
    dominant_direction: str  # Human-readable direction
    flow_vectors: List[MovementVector]
    density_map: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "people_count": self.people_count,
            "avg_speed": round(self.avg_speed, 2),
            "avg_direction": round(self.avg_direction, 2),
            "dominant_direction": self.dominant_direction,
            "flow_vector_count": len(self.flow_vectors),
        }


class MovementAnalyzer:
    """
    Analyzes crowd movement using optical flow and detection tracking.
    """
    
    def __init__(
        self,
        history_size: int = 10,
        min_movement_threshold: float = 2.0,
        max_movement_threshold: float = 100.0,
    ):
        """
        Initialize the movement analyzer.
        
        Args:
            history_size: Number of frames to keep in history
            min_movement_threshold: Minimum pixels to count as movement
            max_movement_threshold: Maximum pixels (filter out noise)
        """
        self.history_size = history_size
        self.min_movement_threshold = min_movement_threshold
        self.max_movement_threshold = max_movement_threshold
        
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_detections: List[Detection] = []
        self._detection_history: deque = deque(maxlen=history_size)
        self._flow_history: deque = deque(maxlen=history_size)
        
        # Lucas-Kanade optical flow parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Farneback dense optical flow parameters
        self._farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        timestamp: float = 0.0,
    ) -> CrowdFlowAnalysis:
        """
        Analyze crowd movement in a frame.
        
        Args:
            frame: BGR image
            detections: List of person detections
            timestamp: Frame timestamp
            
        Returns:
            CrowdFlowAnalysis with movement data
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flow_vectors = []
        
        if self._prev_gray is not None and len(self._prev_detections) > 0:
            # Track movement of detected people
            flow_vectors = self._track_detections(gray, detections)
        
        # Update history
        self._prev_gray = gray.copy()
        self._prev_detections = detections.copy()
        self._detection_history.append(detections)
        self._flow_history.append(flow_vectors)
        
        # Calculate aggregate statistics
        avg_speed = 0.0
        avg_direction = 0.0
        
        if flow_vectors:
            speeds = [v.speed for v in flow_vectors]
            directions = [v.direction for v in flow_vectors]
            
            avg_speed = np.mean(speeds)
            avg_direction = self._circular_mean(directions)
        
        dominant_direction = self._get_direction_name(avg_direction)
        
        return CrowdFlowAnalysis(
            timestamp=timestamp,
            people_count=len(detections),
            avg_speed=avg_speed,
            avg_direction=avg_direction,
            dominant_direction=dominant_direction,
            flow_vectors=flow_vectors,
        )
    
    def _track_detections(
        self,
        gray: np.ndarray,
        current_detections: List[Detection],
    ) -> List[MovementVector]:
        """
        Track movement of detections between frames.
        
        Uses Hungarian algorithm for matching + optical flow for refinement.
        """
        flow_vectors = []
        
        if not current_detections or not self._prev_detections:
            return flow_vectors
        
        # Simple nearest-neighbor matching based on center distance
        prev_centers = np.array([d.center for d in self._prev_detections], dtype=np.float32)
        curr_centers = np.array([d.center for d in current_detections], dtype=np.float32)
        
        # Use optical flow to refine tracking
        if len(prev_centers) > 0:
            prev_points = prev_centers.reshape(-1, 1, 2)
            
            next_points, status, err = cv2.calcOpticalFlowPyrLK(
                self._prev_gray,
                gray,
                prev_points,
                None,
                **self._lk_params
            )
            
            if next_points is not None:
                for i, (prev_pt, next_pt, st) in enumerate(zip(prev_points, next_points, status)):
                    if st[0] == 0:
                        continue
                    
                    px, py = prev_pt.ravel()
                    nx, ny = next_pt.ravel()
                    
                    dx = nx - px
                    dy = ny - py
                    speed = math.sqrt(dx*dx + dy*dy)
                    
                    # Filter out noise
                    if speed < self.min_movement_threshold:
                        continue
                    if speed > self.max_movement_threshold:
                        continue
                    
                    direction = math.degrees(math.atan2(dy, dx))
                    if direction < 0:
                        direction += 360
                    
                    flow_vectors.append(MovementVector(
                        start=(int(px), int(py)),
                        end=(int(nx), int(ny)),
                        dx=dx,
                        dy=dy,
                        speed=speed,
                        direction=direction,
                    ))
        
        return flow_vectors
    
    def compute_dense_flow(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dense optical flow for the entire frame.
        
        Returns:
            Tuple of (flow_visualization, flow_magnitude)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self._prev_gray is None:
            self._prev_gray = gray
            return np.zeros_like(frame), np.zeros_like(gray)
        
        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray,
            gray,
            None,
            **self._farneback_params
        )
        
        # Convert flow to HSV visualization
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        hsv = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction
        hsv[..., 1] = 255  # Saturation
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
        
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        self._prev_gray = gray
        
        return flow_vis, mag
    
    def visualize_flow(
        self,
        frame: np.ndarray,
        flow_vectors: List[MovementVector],
        arrow_color: Tuple[int, int, int] = (0, 255, 0),
        arrow_scale: float = 2.0,
    ) -> np.ndarray:
        """
        Draw flow vectors on frame.
        
        Args:
            frame: BGR image
            flow_vectors: List of movement vectors
            arrow_color: Color for arrows
            arrow_scale: Scale factor for arrow length
            
        Returns:
            Annotated frame
        """
        vis = frame.copy()
        
        for vec in flow_vectors:
            # Scale the arrow for visibility
            end_x = int(vec.start[0] + vec.dx * arrow_scale)
            end_y = int(vec.start[1] + vec.dy * arrow_scale)
            
            cv2.arrowedLine(
                vis,
                vec.start,
                (end_x, end_y),
                arrow_color,
                2,
                tipLength=0.3
            )
        
        return vis
    
    def _circular_mean(self, angles: List[float]) -> float:
        """
        Calculate circular mean of angles in degrees.
        """
        if not angles:
            return 0.0
        
        sin_sum = sum(math.sin(math.radians(a)) for a in angles)
        cos_sum = sum(math.cos(math.radians(a)) for a in angles)
        
        mean_angle = math.degrees(math.atan2(sin_sum, cos_sum))
        if mean_angle < 0:
            mean_angle += 360
        
        return mean_angle
    
    def _get_direction_name(self, angle: float) -> str:
        """
        Convert angle to human-readable direction.
        
        Args:
            angle: Angle in degrees (0=right, 90=down, etc.)
            
        Returns:
            Direction name (N, NE, E, SE, S, SW, W, NW)
        """
        # Normalize to 0-360
        angle = angle % 360
        
        # Map to compass directions
        # Note: In image coordinates, y increases downward
        directions = [
            (337.5, 360, "E"),
            (0, 22.5, "E"),
            (22.5, 67.5, "SE"),
            (67.5, 112.5, "S"),
            (112.5, 157.5, "SW"),
            (157.5, 202.5, "W"),
            (202.5, 247.5, "NW"),
            (247.5, 292.5, "N"),
            (292.5, 337.5, "NE"),
        ]
        
        for start, end, name in directions:
            if start <= angle < end:
                return name
        
        return "E"
    
    def get_movement_stats(self) -> Dict[str, Any]:
        """
        Get aggregate movement statistics from history.
        """
        if not self._flow_history:
            return {
                "frames_analyzed": 0,
                "total_movements": 0,
            }
        
        all_vectors = []
        for vectors in self._flow_history:
            all_vectors.extend(vectors)
        
        if not all_vectors:
            return {
                "frames_analyzed": len(self._flow_history),
                "total_movements": 0,
                "avg_speed": 0.0,
            }
        
        speeds = [v.speed for v in all_vectors]
        directions = [v.direction for v in all_vectors]
        
        return {
            "frames_analyzed": len(self._flow_history),
            "total_movements": len(all_vectors),
            "avg_speed": np.mean(speeds),
            "max_speed": max(speeds),
            "min_speed": min(speeds),
            "avg_direction": self._circular_mean(directions),
            "speed_std": np.std(speeds),
        }
    
    def reset(self):
        """Reset analyzer state."""
        self._prev_gray = None
        self._prev_detections = []
        self._detection_history.clear()
        self._flow_history.clear()


class CrowdDensityAnalyzer:
    """
    Analyzes crowd density distribution across the frame.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize density analyzer.
        
        Args:
            grid_size: Grid divisions for density map (cols, rows)
        """
        self.grid_size = grid_size
    
    def compute_density_map(
        self,
        frame_shape: Tuple[int, int],
        detections: List[Detection],
    ) -> np.ndarray:
        """
        Compute density map showing crowd distribution.
        
        Args:
            frame_shape: (height, width) of frame
            detections: List of person detections
            
        Returns:
            Density map as numpy array
        """
        height, width = frame_shape[:2]
        density = np.zeros(self.grid_size[::-1], dtype=np.float32)
        
        cell_width = width / self.grid_size[0]
        cell_height = height / self.grid_size[1]
        
        for det in detections:
            cx, cy = det.center
            grid_x = min(int(cx / cell_width), self.grid_size[0] - 1)
            grid_y = min(int(cy / cell_height), self.grid_size[1] - 1)
            density[grid_y, grid_x] += 1
        
        return density
    
    def visualize_density(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay density heatmap on frame.
        """
        density = self.compute_density_map(frame.shape, detections)
        
        # Normalize and resize to frame size
        if density.max() > 0:
            density_norm = (density / density.max() * 255).astype(np.uint8)
        else:
            density_norm = density.astype(np.uint8)
        
        heatmap = cv2.resize(density_norm, (frame.shape[1], frame.shape[0]))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)
        
        return result


if __name__ == "__main__":
    print("Testing MovementAnalyzer...")
    
    analyzer = MovementAnalyzer()
    
    # Create test frames
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate detections
    detections1 = [
        Detection(bbox=(100, 100, 150, 200), confidence=0.9, center=(125, 150), area=5000),
        Detection(bbox=(300, 200, 350, 300), confidence=0.85, center=(325, 250), area=5000),
    ]
    
    detections2 = [
        Detection(bbox=(110, 105, 160, 205), confidence=0.9, center=(135, 155), area=5000),
        Detection(bbox=(310, 190, 360, 290), confidence=0.85, center=(335, 240), area=5000),
    ]
    
    # Analyze
    result1 = analyzer.analyze_frame(frame1, detections1, timestamp=0.0)
    result2 = analyzer.analyze_frame(frame2, detections2, timestamp=0.033)
    
    print(f"Frame 1: {result1.to_dict()}")
    print(f"Frame 2: {result2.to_dict()}")
    print(f"Movement stats: {analyzer.get_movement_stats()}")
