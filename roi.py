import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load YOLO model (replace with your trained model path)
model = YOLO("my_model.pt")   # e.g., "best.pt"

# Initialize video capture (0 = Mac webcam, or use video path)
cap = cv2.VideoCapture(0)

# Optional: Path to a sound file for alert
ALERT_SOUND = "/System/Library/Sounds/Ping.aiff"  # built-in macOS sound

def play_alert():
    # Play sound using macOS system player
    os.system(f"afplay '{ALERT_SOUND}' &")

def is_inside_zone(center, polygon):
    """Check if a point lies inside the polygon region"""
    return cv2.pointPolygonTest(polygon, center, False) >= 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Define blind spot zones (tweak to match your camera angle)
    left_zone = np.array([
        [0, h],
        [int(w * 0.35), int(h * 0.65)],
        [int(w * 0.35), h]
    ], np.int32)

    right_zone = np.array([
        [w, h],
        [int(w * 0.65), int(h * 0.65)],
        [int(w * 0.65), h]
    ], np.int32)

    # Draw blind spot regions
    cv2.polylines(frame, [left_zone], True, (0, 255, 255), 2)
    cv2.polylines(frame, [right_zone], True, (0, 255, 255), 2)

    # Run YOLO prediction
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    alert_triggered = False

    for (x1, y1, x2, y2, conf, cls) in detections:
        cx = int((x1 + x2)
