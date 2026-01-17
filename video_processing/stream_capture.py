"""
Video Stream Capture Module

Handles capturing video frames from IP webcam streams.
Supports HTTP MJPEG streams (like IP Webcam Android app).
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Tuple, Generator
from queue import Queue, Empty
import requests


class VideoStreamCapture:
    """
    Captures video frames from an IP webcam stream.
    
    Supports:
    - MJPEG streams over HTTP
    - Direct OpenCV video capture
    - Frame buffering for async processing
    """
    
    def __init__(
        self,
        stream_url: str = "http://10.203.120.62:8080/video",
        buffer_size: int = 10,
        reconnect_delay: float = 2.0,
    ):
        """
        Initialize the video stream capture.
        
        Args:
            stream_url: URL of the IP webcam video stream
            buffer_size: Number of frames to buffer
            reconnect_delay: Seconds to wait before reconnecting on failure
        """
        self.stream_url = stream_url
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: Queue = Queue(maxsize=buffer_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        
    def connect(self) -> bool:
        """
        Establish connection to the video stream.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            print(f"Connecting to stream: {self.stream_url}")
            self._cap = cv2.VideoCapture(self.stream_url)
            
            if not self._cap.isOpened():
                print("Failed to open video stream with OpenCV")
                return False
            
            # Get stream properties
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Stream connected: {width}x{height} @ {fps:.1f} FPS")
            return True
            
        except Exception as e:
            print(f"Error connecting to stream: {e}")
            return False
    
    def disconnect(self):
        """Release the video stream connection."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        print("Stream disconnected")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the stream.
        
        Returns:
            Tuple of (success, frame)
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None
        
        ret, frame = self._cap.read()
        
        if ret:
            self._frame_count += 1
            self._last_frame = frame.copy()
            self._update_fps()
            
        return ret, frame
    
    def _update_fps(self):
        """Update FPS calculation."""
        self._fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self._last_fps_time
        
        if elapsed >= 1.0:
            self._fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._last_fps_time = current_time
    
    def start_async(self):
        """Start asynchronous frame capture in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("Async capture started")
    
    def stop_async(self):
        """Stop asynchronous frame capture."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("Async capture stopped")
    
    def _capture_loop(self):
        """Background thread for continuous frame capture."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self.connect():
                    time.sleep(self.reconnect_delay)
                    continue
            
            ret, frame = self.read_frame()
            
            if not ret:
                print("Frame read failed, reconnecting...")
                self.disconnect()
                time.sleep(self.reconnect_delay)
                continue
            
            # Put frame in queue, drop oldest if full
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    pass
            
            self._frame_queue.put(frame)
    
    def get_frame_async(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get a frame from the async buffer.
        
        Args:
            timeout: Seconds to wait for a frame
            
        Returns:
            Frame if available, None otherwise
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def frames(self, max_frames: Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames from the stream.
        
        Args:
            max_frames: Maximum number of frames to yield (None for infinite)
            
        Yields:
            Video frames as numpy arrays
        """
        count = 0
        while True:
            if max_frames is not None and count >= max_frames:
                break
            
            ret, frame = self.read_frame()
            if not ret:
                break
            
            yield frame
            count += 1
    
    @property
    def fps(self) -> float:
        """Current measured FPS."""
        return self._fps
    
    @property
    def frame_count(self) -> int:
        """Total frames captured."""
        return self._frame_count
    
    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """Most recently captured frame."""
        return self._last_frame
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_async()
        self.disconnect()


def test_stream_connection(url: str = "http://10.203.120.62:8080/video") -> dict:
    """
    Test connection to a video stream and return diagnostic info.
    
    Args:
        url: Stream URL to test
        
    Returns:
        Dictionary with connection test results
    """
    results = {
        "url": url,
        "reachable": False,
        "opencv_opens": False,
        "can_read_frame": False,
        "frame_shape": None,
        "error": None,
    }
    
    # Test HTTP reachability
    try:
        response = requests.head(url, timeout=5)
        results["reachable"] = response.status_code == 200
        results["http_status"] = response.status_code
    except requests.exceptions.RequestException as e:
        results["error"] = f"HTTP error: {e}"
        return results
    
    # Test OpenCV capture
    try:
        cap = cv2.VideoCapture(url)
        results["opencv_opens"] = cap.isOpened()
        
        if cap.isOpened():
            ret, frame = cap.read()
            results["can_read_frame"] = ret
            
            if ret and frame is not None:
                results["frame_shape"] = frame.shape
                results["frame_dtype"] = str(frame.dtype)
        
        cap.release()
        
    except Exception as e:
        results["error"] = f"OpenCV error: {e}"
    
    return results


if __name__ == "__main__":
    # Quick test
    print("Testing video stream connection...")
    results = test_stream_connection()
    
    for key, value in results.items():
        print(f"  {key}: {value}")
