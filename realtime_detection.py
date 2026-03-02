#!/usr/bin/env python3
"""
Real-time object and face detection on RTSP camera feed.
Live OpenCV window with bounding boxes. Face detection every frame (local),
DETR object detection every N seconds (Cloudflare API).
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from camera_vision import build_rtsp_url
from roi_config import load_roi
from vision_detection import (
    USE_YOLO,
    detect_objects,
    detect_vehicle_plates,
    draw_detections_on_frame,
    format_detections,
    get_faces_for_display,
)

MAX_DIM = 960  # Higher res for better detection


def capture_frame_from_cap(cap: cv2.VideoCapture) -> tuple[np.ndarray | None, bytes | None]:
    """Read frame from open VideoCapture. Returns (BGR array, JPEG bytes for API)."""
    ret, frame = cap.read()
    if not ret or frame is None:
        return None, None
    h, w = frame.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return frame, jpeg.tobytes()


def run_detr_loop(
    frame_bytes_ref: list,
    frame_lock: threading.Lock,
    objects_ref: list,
    plates_ref: list,
    account_id: str,
    auth_token: str,
    gateway_id: str | None,
    interval_sec: float,
) -> None:
    """Background thread: run DETR + plate OCR on latest frame every interval_sec."""
    while True:
        time.sleep(interval_sec)
        with frame_lock:
            fb = frame_bytes_ref[0] if frame_bytes_ref else None
        if fb is None:
            continue
        try:
            objs = detect_objects(fb, account_id, auth_token, gateway_id)
            objects_ref.clear()
            objects_ref.extend(objs)
            plates = detect_vehicle_plates(fb, objs)
            plates_ref.clear()
            plates_ref.extend(plates)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time detection on RTSP camera")
    parser.add_argument(
        "--rtsp",
        default=os.environ.get("RTSP_URL", "rtsp://192.168.1.100:554/stream1"),
        help="RTSP stream URL (or set RTSP_URL in .env)",
    )
    parser.add_argument("-u", "--username", default=None, help="RTSP username")
    parser.add_argument("-p", "--password", default=None, help="RTSP password")
    parser.add_argument(
        "--detr-interval",
        type=float,
        default=2.0,
        help="Seconds between DETR API calls (default: 2)",
    )
    parser.add_argument(
        "--no-detr",
        action="store_true",
        help="Disable DETR (face detection only, no API)",
    )
    args = parser.parse_args()

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    auth_token = os.environ.get("CLOUDFLARE_AUTH_TOKEN")
    gateway_id = os.environ.get("CLOUDFLARE_AI_GATEWAY_ID") or None
    username = args.username or os.environ.get("RTSP_USERNAME", "")
    password = args.password or os.environ.get("RTSP_PASSWORD", "")

    needs_cloudflare = not args.no_detr and not USE_YOLO
    if needs_cloudflare and (not account_id or not auth_token):
        print("Error: Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN in .env")
        print("Or set USE_YOLO=1 for local YOLOv8, or use --no-detr for face detection only.")
        sys.exit(1)

    rtsp_url = build_rtsp_url(args.rtsp, username, password)

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MS"):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MS, 10000)
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MS"):
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MS, 5000)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream. Check URL and credentials.")
        sys.exit(1)

    # Shared state for DETR thread
    frame_bytes_ref: list = []
    frame_lock = threading.Lock()
    objects_ref: list = []
    plates_ref: list = []

    if not args.no_detr:
        t = threading.Thread(
            target=run_detr_loop,
            args=(frame_bytes_ref, frame_lock, objects_ref, plates_ref, account_id, auth_token, gateway_id, args.detr_interval),
            daemon=True,
        )
        t.start()

    print("=" * 50)
    print("Real-time Detection")
    print("=" * 50)
    print(f"Stream: {args.rtsp}")
    print(f"Face detection: every frame | DETR: every {args.detr_interval}s" if not args.no_detr else "Face detection only")
    print("Press 'q' or Ctrl+C to quit")
    print("=" * 50)

    fps_start = time.perf_counter()
    fps_frames = 0
    fps_display = 0.0

    try:
        while True:
            frame, frame_bytes = capture_frame_from_cap(cap)
            if frame is None:
                time.sleep(0.1)
                continue

            # Update latest frame for DETR thread
            with frame_lock:
                frame_bytes_ref.clear()
                frame_bytes_ref.append(frame_bytes)

            # Face detection (fast, every frame)
            faces, visitor_names = get_faces_for_display(frame_bytes)

            # Get latest DETR results (may be from previous run)
            objs = list(objects_ref)
            plates = list(plates_ref) if not args.no_detr else []

            # Draw overlays (including ROI zone if configured)
            draw_detections_on_frame(frame, objs, faces, roi=load_roi())

            # FPS
            fps_frames += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                fps_display = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.perf_counter()

            # Status overlay
            status = f"FPS: {fps_display:.1f} | Faces: {len(faces)} | Objects: {len(objs)}"
            cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            det_txt = format_detections(objs, faces, visitor_names, plates)
            if det_txt != "No detections":
                cv2.putText(frame, det_txt[:60], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Real-time Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
