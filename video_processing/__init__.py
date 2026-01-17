"""
Video Processing Module for Crowd Dynamix

This module handles:
- IP webcam video stream capture
- People detection and counting using YOLO
- Crowd movement analysis (speed and direction)
"""

from .stream_capture import VideoStreamCapture
from .people_detector import PeopleDetector
from .movement_analyzer import MovementAnalyzer
from .processor import CrowdVideoProcessor

__all__ = [
    "VideoStreamCapture",
    "PeopleDetector", 
    "MovementAnalyzer",
    "CrowdVideoProcessor",
]
