"""Quick connectivity test for IP webcam stream."""

import cv2
import requests
import time

URL = "http://10.203.120.62:8080/video"

print("=" * 50)
print("IP WEBCAM STREAM CONNECTIVITY TEST")
print("=" * 50)
print(f"URL: {URL}")
print()

# Test HTTP
print("1. Testing HTTP reachability...")
try:
    response = requests.head(URL, timeout=5)
    print(f"   HTTP Status: {response.status_code}")
    print(f"   Content-Type: {response.headers.get('Content-Type', 'N/A')}")
except requests.exceptions.ConnectTimeout:
    print("   ERROR: Connection timed out (device unreachable)")
except requests.exceptions.ConnectionError as e:
    print(f"   ERROR: Connection failed - {e}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Test OpenCV
print()
print("2. Testing OpenCV VideoCapture...")
cap = cv2.VideoCapture(URL)

if cap.isOpened():
    print("   Connection: SUCCESS")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    
    print()
    print("3. Reading test frames...")
    frames_read = 0
    start = time.time()
    
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            frames_read += 1
            if i == 0:
                print(f"   First frame shape: {frame.shape}")
                print(f"   Frame dtype: {frame.dtype}")
    
    elapsed = time.time() - start
    actual_fps = frames_read / elapsed if elapsed > 0 else 0
    
    print(f"   Frames captured: {frames_read}/10")
    print(f"   Actual FPS: {actual_fps:.1f}")
    
    cap.release()
else:
    print("   Connection: FAILED")
    print("   Make sure IP Webcam is running and accessible")

print()
print("=" * 50)
print("TEST COMPLETE")
print("=" * 50)
