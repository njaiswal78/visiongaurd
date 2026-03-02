#!/usr/bin/env python3
"""
Interactive ROI setup: capture a frame, draw a rectangle to define the movement-tracking zone.
Saves to roi_zone.json. Movement alerts will only trigger when activity is detected inside this zone.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from camera_vision import build_rtsp_url, capture_frame
from roi_config import save_roi

RTSP_URL = os.environ.get("RTSP_URL", "rtsp://192.168.1.100:554/stream1")
RTSP_USERNAME = os.environ.get("RTSP_USERNAME", "")
RTSP_PASSWORD = os.environ.get("RTSP_PASSWORD", "")


def main():
    rtsp_url = build_rtsp_url(RTSP_URL, RTSP_USERNAME, RTSP_PASSWORD)
    print("Capturing frame from camera...")
    frame_bytes = capture_frame(rtsp_url)
    if not frame_bytes:
        print("Error: Could not capture frame. Check RTSP URL and credentials.")
        sys.exit(1)

    import cv2
    import numpy as np

    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("Error: Could not decode image.")
        sys.exit(1)

    h, w = img.shape[:2]
    roi = [0, 0, 0, 0]  # x1, y1, x2, y2 in pixels
    drawing = False
    start_pt = None

    def on_mouse(event, x, y, flags, param):
        nonlocal roi, drawing, start_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
            roi = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            roi[2], roi[3] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi[2], roi[3] = x, y
            # Ensure x1 < x2, y1 < y2
            roi[0], roi[2] = min(roi[0], roi[2]), max(roi[0], roi[2])
            roi[1], roi[3] = min(roi[1], roi[3]), max(roi[1], roi[3])

    cv2.namedWindow("Draw movement zone (drag rectangle, press S to save, Q to quit)")
    cv2.setMouseCallback("Draw movement zone (drag rectangle, press S to save, Q to quit)", on_mouse)

    print()
    print("Instructions:")
    print("  • Click and drag to draw a rectangle around the area to monitor")
    print("  • Press S to save and exit")
    print("  • Press Q to quit without saving")
    print()

    while True:
        display = img.copy()
        if roi[2] > roi[0] and roi[3] > roi[1]:
            cv2.rectangle(display, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 2)
            cv2.putText(
                display, "Movement tracking zone",
                (roi[0], roi[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        cv2.imshow("Draw movement zone (drag rectangle, press S to save, Q to quit)", display)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q") or key == ord("Q"):
            print("Quit without saving.")
            cv2.destroyAllWindows()
            sys.exit(0)
        if key == ord("s") or key == ord("S"):
            if roi[2] <= roi[0] or roi[3] <= roi[1]:
                print("Draw a valid rectangle first (click and drag).")
                continue
            # Convert to normalized 0-1
            x1 = roi[0] / w
            y1 = roi[1] / h
            x2 = roi[2] / w
            y2 = roi[3] / h
            save_roi(x1, y1, x2, y2)
            print(f"Saved ROI: x1={x1:.3f}, y1={y1:.3f}, x2={x2:.3f}, y2={y2:.3f}")
            print("Movement alerts will now only trigger when activity is detected in this zone.")
            cv2.destroyAllWindows()
            break

    print("Done.")


if __name__ == "__main__":
    main()
