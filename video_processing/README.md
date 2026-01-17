# Video Processing Module

This module provides real-time crowd analysis from IP webcam video streams. It captures video frames, detects and counts people using machine learning, and analyzes crowd movement (speed and direction).

## Features

- **Stream Capture**: Connect to IP webcam streams (MJPEG over HTTP)
- **People Detection**: Count people using YOLOv8 or HOG+SVM
- **Movement Analysis**: Calculate crowd speed and direction using optical flow
- **Density Mapping**: Visualize crowd density distribution

## Installation

```bash
cd video_processing
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Ultralytics (YOLOv8)
- PyTorch
- Requests

## Quick Start

### 1. Test Stream Connectivity

```bash
python test_processing.py --test connectivity --url http://10.203.120.62:8080/video
```

### 2. Run Full Processing Pipeline

```bash
python processor.py --url http://10.203.120.62:8080/video
```

### 3. Process a Video File

```bash
python processor.py --video input.mp4 --output output.mp4
```

## Usage

### Basic Usage

```python
from video_processing import CrowdVideoProcessor

# Create processor
processor = CrowdVideoProcessor(
    stream_url="http://10.203.120.62:8080/video",
    use_yolo=True,
    confidence_threshold=0.5,
)

# Run live processing with display
processor.run_live(display=True)
```

### Async Processing with Callbacks

```python
def on_result(result):
    print(f"People: {result.people_count}")
    print(f"Direction: {result.flow_analysis.dominant_direction}")
    print(f"Speed: {result.flow_analysis.avg_speed:.1f} px/frame")

processor.set_result_callback(on_result)
processor.run_async()
```

### Process Individual Frames

```python
import cv2
from video_processing import CrowdVideoProcessor

processor = CrowdVideoProcessor()

cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = processor.process_frame(frame)
    print(f"Frame {result.frame_number}: {result.people_count} people")
```

### Using Individual Components

```python
from video_processing import (
    VideoStreamCapture,
    PeopleDetector,
    MovementAnalyzer,
)

# Stream capture
stream = VideoStreamCapture("http://10.203.120.62:8080/video")
stream.connect()

# People detection
detector = PeopleDetector()
detector.initialize()

# Movement analysis
analyzer = MovementAnalyzer()

for frame in stream.frames(max_frames=100):
    detections = detector.detect(frame)
    flow = analyzer.analyze_frame(frame, detections)
    
    print(f"People: {len(detections)}, Direction: {flow.dominant_direction}")
```

## Module Components

### `stream_capture.py`
- `VideoStreamCapture`: Captures frames from IP webcam
- Supports sync and async capture
- Auto-reconnect on failure

### `people_detector.py`
- `PeopleDetector`: YOLOv8-based detection (accurate)
- `PeopleDetectorLite`: HOG+SVM detection (fast, no GPU needed)

### `movement_analyzer.py`
- `MovementAnalyzer`: Optical flow-based movement tracking
- `CrowdDensityAnalyzer`: Density heatmap generation

### `processor.py`
- `CrowdVideoProcessor`: Main processing pipeline
- Combines all components
- Provides visualization and callbacks

## Output Data

Each processed frame returns a `ProcessingResult`:

```python
{
    "timestamp": 1.234,
    "frame_number": 42,
    "people_count": 15,
    "detections": [
        {"bbox": [100, 100, 200, 300], "confidence": 0.92, "center": [150, 200]},
        ...
    ],
    "flow": {
        "people_count": 15,
        "avg_speed": 12.5,
        "avg_direction": 45.0,
        "dominant_direction": "NE",
        "flow_vector_count": 12
    },
    "processing_time_ms": 45.2
}
```

## Direction Reference

| Angle | Direction | Meaning |
|-------|-----------|---------|
| 0°    | E         | Right   |
| 45°   | SE        | Down-Right |
| 90°   | S         | Down    |
| 135°  | SW        | Down-Left |
| 180°  | W         | Left    |
| 225°  | NW        | Up-Left |
| 270°  | N         | Up      |
| 315°  | NE        | Up-Right |

## IP Webcam Setup

This module is designed to work with the "IP Webcam" Android app:

1. Install "IP Webcam" from Play Store
2. Start server in the app
3. Note the IP address shown (e.g., `http://192.168.1.100:8080`)
4. Access video at `http://<IP>:8080/video`

### Alternative Endpoints

- `/video` - MJPEG stream
- `/shot.jpg` - Single JPEG image
- `/photoaf.jpg` - JPEG with autofocus

## Performance Tips

1. **Use YOLOv8n** (nano) for real-time processing
2. **Reduce resolution** for faster processing
3. **Use `PeopleDetectorLite`** on CPU-only systems
4. **Adjust confidence threshold** to balance accuracy vs speed

## Troubleshooting

### Stream won't connect
- Check if the IP webcam is running
- Verify the IP address and port
- Ensure you're on the same network
- Try accessing the URL in a browser first

### Low FPS
- Use a smaller YOLO model (yolov8n)
- Reduce video resolution in IP Webcam settings
- Use HOG detector instead of YOLO

### YOLO not loading
- Install ultralytics: `pip install ultralytics`
- First run downloads the model (~6MB for nano)
