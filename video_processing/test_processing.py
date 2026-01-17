"""
Tests for Video Processing Module

Run these tests to verify video stream connectivity and processing functionality.
"""

import cv2
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_stream_connectivity(url: str = "http://10.203.120.62:8080/video"):
    """
    Test basic connectivity to the IP webcam stream.
    """
    print("=" * 60)
    print("TEST: Stream Connectivity")
    print("=" * 60)
    print(f"URL: {url}")
    
    # Test 1: HTTP Reachability
    print("\n1. Testing HTTP reachability...")
    try:
        import requests
        response = requests.head(url, timeout=5)
        print(f"   HTTP Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
        http_ok = response.status_code == 200
    except Exception as e:
        print(f"   ERROR: {e}")
        http_ok = False
    
    # Test 2: OpenCV Connection
    print("\n2. Testing OpenCV VideoCapture...")
    cap = cv2.VideoCapture(url)
    
    if cap.isOpened():
        print("   Connection: SUCCESS")
        
        # Get stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        
        # Test 3: Read frames
        print("\n3. Testing frame capture...")
        frames_read = 0
        start_time = time.time()
        
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                frames_read += 1
                if i == 0:
                    print(f"   First frame shape: {frame.shape}")
                    print(f"   Frame dtype: {frame.dtype}")
        
        elapsed = time.time() - start_time
        actual_fps = frames_read / elapsed if elapsed > 0 else 0
        
        print(f"   Frames captured: {frames_read}/10")
        print(f"   Actual FPS: {actual_fps:.1f}")
        
        cap.release()
        opencv_ok = frames_read > 0
    else:
        print("   Connection: FAILED")
        opencv_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"HTTP Reachable: {'✓' if http_ok else '✗'}")
    print(f"OpenCV Capture: {'✓' if opencv_ok else '✗'}")
    
    return http_ok and opencv_ok


def test_stream_capture_module(url: str = "http://10.203.120.62:8080/video"):
    """
    Test the VideoStreamCapture module.
    """
    print("\n" + "=" * 60)
    print("TEST: VideoStreamCapture Module")
    print("=" * 60)
    
    from video_processing.stream_capture import VideoStreamCapture, test_stream_connection
    
    # Use built-in test function
    print("\n1. Running diagnostic test...")
    results = test_stream_connection(url)
    
    for key, value in results.items():
        print(f"   {key}: {value}")
    
    # Test context manager
    print("\n2. Testing context manager...")
    try:
        with VideoStreamCapture(url) as stream:
            if stream._cap and stream._cap.isOpened():
                print("   Context manager: SUCCESS")
                
                # Test frame generator
                print("\n3. Testing frame generator...")
                count = 0
                for frame in stream.frames(max_frames=5):
                    count += 1
                    print(f"   Frame {count}: shape={frame.shape}")
                
                print(f"   Total frames from generator: {count}")
            else:
                print("   Context manager: FAILED (not connected)")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    return results.get("can_read_frame", False)


def test_people_detector():
    """
    Test the people detector module.
    """
    print("\n" + "=" * 60)
    print("TEST: People Detector Module")
    print("=" * 60)
    
    from video_processing.people_detector import PeopleDetector, PeopleDetectorLite
    
    # Create test image with some structure
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test HOG detector (always available)
    print("\n1. Testing HOG Detector...")
    try:
        hog = PeopleDetectorLite()
        detections = hog.detect(test_frame)
        print(f"   HOG detector: SUCCESS")
        print(f"   Detections on random image: {len(detections)}")
    except Exception as e:
        print(f"   HOG detector ERROR: {e}")
    
    # Test YOLO detector
    print("\n2. Testing YOLO Detector...")
    try:
        yolo = PeopleDetector(model_name="yolov8n.pt")
        yolo.initialize()
        detections = yolo.detect(test_frame)
        print(f"   YOLO detector: SUCCESS")
        print(f"   Detections on random image: {len(detections)}")
    except ImportError:
        print("   YOLO detector: SKIPPED (ultralytics not installed)")
        print("   Install with: pip install ultralytics")
    except Exception as e:
        print(f"   YOLO detector ERROR: {e}")
    
    return True


def test_movement_analyzer():
    """
    Test the movement analyzer module.
    """
    print("\n" + "=" * 60)
    print("TEST: Movement Analyzer Module")
    print("=" * 60)
    
    from video_processing.movement_analyzer import MovementAnalyzer, CrowdDensityAnalyzer
    from video_processing.people_detector import Detection
    
    analyzer = MovementAnalyzer()
    
    # Create test sequence
    print("\n1. Testing with simulated detections...")
    
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate people moving right
    detections1 = [
        Detection(bbox=(100, 200, 150, 300), confidence=0.9, center=(125, 250), area=5000),
        Detection(bbox=(300, 100, 350, 200), confidence=0.85, center=(325, 150), area=5000),
    ]
    
    detections2 = [
        Detection(bbox=(120, 200, 170, 300), confidence=0.9, center=(145, 250), area=5000),
        Detection(bbox=(320, 100, 370, 200), confidence=0.85, center=(345, 150), area=5000),
    ]
    
    # Analyze frames
    result1 = analyzer.analyze_frame(frame1, detections1, timestamp=0.0)
    result2 = analyzer.analyze_frame(frame2, detections2, timestamp=0.033)
    
    print(f"   Frame 1 result: {result1.to_dict()}")
    print(f"   Frame 2 result: {result2.to_dict()}")
    
    stats = analyzer.get_movement_stats()
    print(f"   Movement stats: {stats}")
    
    # Test density analyzer
    print("\n2. Testing density analyzer...")
    density_analyzer = CrowdDensityAnalyzer()
    density = density_analyzer.compute_density_map((480, 640), detections1)
    print(f"   Density map shape: {density.shape}")
    print(f"   Density map:\n{density}")
    
    return True


def test_full_pipeline(url: str = "http://10.203.120.62:8080/video", num_frames: int = 30):
    """
    Test the full processing pipeline with live stream.
    """
    print("\n" + "=" * 60)
    print("TEST: Full Processing Pipeline")
    print("=" * 60)
    
    from video_processing.processor import CrowdVideoProcessor
    
    processor = CrowdVideoProcessor(
        stream_url=url,
        use_yolo=True,
        confidence_threshold=0.5,
    )
    
    print(f"\nProcessing {num_frames} frames from stream...")
    
    results = []
    
    def on_result(result):
        results.append(result)
        if len(results) % 10 == 0:
            print(f"   Frame {result.frame_number}: {result.people_count} people, "
                  f"direction: {result.flow_analysis.dominant_direction}")
    
    processor.set_result_callback(on_result)
    
    try:
        processor.run_live(
            display=True,  # Set to False for headless testing
            max_frames=num_frames,
        )
        
        print(f"\nProcessed {len(results)} frames")
        
        if results:
            avg_people = sum(r.people_count for r in results) / len(results)
            avg_time = sum(r.processing_time_ms for r in results) / len(results)
            print(f"Average people: {avg_people:.1f}")
            print(f"Average processing time: {avg_time:.1f}ms")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_with_sample_video():
    """
    Test with a local video file (if available).
    """
    print("\n" + "=" * 60)
    print("TEST: Sample Video File Processing")
    print("=" * 60)
    
    # Try to find a sample video
    sample_paths = [
        "sample.mp4",
        "test_video.mp4",
        "../sample.mp4",
    ]
    
    video_path = None
    for path in sample_paths:
        if os.path.exists(path):
            video_path = path
            break
    
    if not video_path:
        print("No sample video found. Creating synthetic test...")
        
        # Create synthetic video
        print("\nCreating synthetic 100-frame test video...")
        frames = []
        
        for i in range(100):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw moving rectangle (simulating person)
            x = 50 + i * 5
            y = 200
            cv2.rectangle(frame, (x, y), (x + 50, y + 100), (255, 255, 255), -1)
            frames.append(frame)
        
        # Process synthetic frames
        from video_processing.people_detector import PeopleDetectorLite
        from video_processing.movement_analyzer import MovementAnalyzer
        
        detector = PeopleDetectorLite()
        analyzer = MovementAnalyzer()
        
        print("Processing synthetic frames with HOG detector...")
        for i, frame in enumerate(frames[:20]):
            detections = detector.detect(frame)
            flow = analyzer.analyze_frame(frame, detections, timestamp=i/30.0)
            if i % 5 == 0:
                print(f"   Frame {i}: {len(detections)} detections, "
                      f"direction: {flow.dominant_direction}")
        
        print("Synthetic test completed")
        return True
    
    else:
        from video_processing.processor import process_video_file
        
        print(f"Processing video: {video_path}")
        results = process_video_file(video_path, use_yolo=False)
        print(f"Processed {len(results)} frames")
        return len(results) > 0


def run_all_tests(url: str = "http://10.203.120.62:8080/video"):
    """
    Run all tests.
    """
    print("\n" + "=" * 60)
    print("CROWD VIDEO PROCESSING - TEST SUITE")
    print("=" * 60)
    print(f"Stream URL: {url}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Basic connectivity
    print("\n[1/5] Testing stream connectivity...")
    try:
        results["connectivity"] = test_stream_connectivity(url)
    except Exception as e:
        print(f"Test failed: {e}")
        results["connectivity"] = False
    
    # Test 2: Stream capture module
    print("\n[2/5] Testing stream capture module...")
    try:
        results["stream_capture"] = test_stream_capture_module(url)
    except Exception as e:
        print(f"Test failed: {e}")
        results["stream_capture"] = False
    
    # Test 3: People detector
    print("\n[3/5] Testing people detector...")
    try:
        results["people_detector"] = test_people_detector()
    except Exception as e:
        print(f"Test failed: {e}")
        results["people_detector"] = False
    
    # Test 4: Movement analyzer
    print("\n[4/5] Testing movement analyzer...")
    try:
        results["movement_analyzer"] = test_movement_analyzer()
    except Exception as e:
        print(f"Test failed: {e}")
        results["movement_analyzer"] = False
    
    # Test 5: Sample video
    print("\n[5/5] Testing with sample video...")
    try:
        results["sample_video"] = test_with_sample_video()
    except Exception as e:
        print(f"Test failed: {e}")
        results["sample_video"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    return all(results.values())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Video Processing Module")
    parser.add_argument("--url", default="http://10.203.120.62:8080/video",
                       help="Video stream URL")
    parser.add_argument("--test", choices=[
        "connectivity", "stream", "detector", "movement", "video", "full", "all"
    ], default="all", help="Which test to run")
    parser.add_argument("--frames", type=int, default=30,
                       help="Number of frames for full pipeline test")
    
    args = parser.parse_args()
    
    if args.test == "connectivity":
        test_stream_connectivity(args.url)
    elif args.test == "stream":
        test_stream_capture_module(args.url)
    elif args.test == "detector":
        test_people_detector()
    elif args.test == "movement":
        test_movement_analyzer()
    elif args.test == "video":
        test_with_sample_video()
    elif args.test == "full":
        test_full_pipeline(args.url, args.frames)
    else:
        run_all_tests(args.url)
