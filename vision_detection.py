#!/usr/bin/env python3
"""
Object detection: YOLOv8 (local) or DETR (Cloudflare), plus face detection (OpenCV).
OCR for vehicle license plates (EasyOCR).
Set USE_YOLO=1 to use local YOLOv8 (no API cost, GPU recommended).
"""

import os
import warnings

# Suppress PyTorch pin_memory warning on Apple MPS
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

import re
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# USE_YOLO=1 → local YOLOv8 (ultralytics). Else → Cloudflare DETR.
USE_YOLO = os.environ.get("USE_YOLO", "").strip().lower() in ("1", "true", "yes")
# USE_PLATE_OCR=1 → run EasyOCR to detect vehicle license plates.
USE_PLATE_OCR = os.environ.get("USE_PLATE_OCR", "1").strip().lower() in ("1", "true", "yes")
VEHICLE_LABELS = {"car", "truck", "motorcycle", "bus", "bicycle"}

DETR_MODEL = "@cf/facebook/detr-resnet-50"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
GATEWAY_BASE = "https://gateway.ai.cloudflare.com/v1"
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")  # n=nano, s=small, m=medium, l=large, x=extra

# OpenCV Haar cascade for face detection (bundled with opencv-python)
FACE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

_yolo_model = None


def _get_yolo_model():
    """Lazy-load YOLOv8 model (downloads on first use)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL)
    return _yolo_model


def _detect_objects_yolo(image_bytes: bytes, min_score: float = 0.3) -> list[dict]:
    """Detect objects using local YOLOv8. Returns list of {label, score, box}."""
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []
        model = _get_yolo_model()
        results = model(img, verbose=False)[0]
        out = []
        if results.boxes is None:
            return []
        names = results.names or {}
        for i, box in enumerate(results.boxes):
            conf = float(box.conf[0]) if box.conf is not None else 0
            if conf < min_score:
                continue
            xyxy = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            cls_id = int(box.cls[0]) if box.cls is not None else 0
            label = names.get(cls_id, "unknown")
            out.append({
                "label": label,
                "score": conf,
                "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            })
        return out
    except Exception:
        return []


def _detr_url(account_id: str, gateway_id: str | None = None) -> str:
    if gateway_id:
        return f"{GATEWAY_BASE}/{account_id}/{gateway_id}/workers-ai/{DETR_MODEL}"
    return f"{API_BASE}/{account_id}/ai/run/{DETR_MODEL}"


def _headers(auth_token: str, gateway_id: str | None = None) -> dict:
    h = {"Authorization": f"Bearer {auth_token}"}
    if gateway_id:
        gt = os.environ.get("CLOUDFLARE_AI_GATEWAY_AUTH_TOKEN") or auth_token
        h["cf-aig-authorization"] = f"Bearer {gt}"
    return h


def _detect_objects_detr(
    image_bytes: bytes,
    account_id: str,
    auth_token: str,
    gateway_id: str | None = None,
    min_score: float = 0.3,
) -> list[dict]:
    """Detect objects using Cloudflare DETR. Returns list of {label, score, box}."""
    url = _detr_url(account_id, gateway_id)
    headers = _headers(auth_token, gateway_id)
    payload = {"image": list(image_bytes)}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True):
            return []
        raw = data.get("result")
        if not isinstance(raw, list):
            return []
        out = []
        for item in raw:
            if isinstance(item, dict) and item.get("score", 0) >= min_score:
                out.append({
                    "label": item.get("label", "unknown"),
                    "score": item.get("score", 0),
                    "box": item.get("box", {}),
                })
        return out
    except Exception:
        return []


def detect_objects(
    image_bytes: bytes,
    account_id: str = "",
    auth_token: str = "",
    gateway_id: str | None = None,
    min_score: float = 0.3,
) -> list[dict]:
    """
    Detect objects. Uses YOLOv8 (local) if USE_YOLO=1, else Cloudflare DETR.
    Returns list of {label, score, box} with box = {xmin, ymin, xmax, ymax}.
    """
    if USE_YOLO:
        return _detect_objects_yolo(image_bytes, min_score)
    return _detect_objects_detr(image_bytes, account_id, auth_token, gateway_id, min_score)


def detect_faces(image_bytes: bytes) -> list[dict]:
    """
    Detect faces using OpenCV Haar cascade. Returns list of {x, y, w, h}.
    Stricter params (minNeighbors=8, minSize=50) to reduce false positives.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    if cascade.empty():
        return []
    faces = cascade.detectMultiScale(img, scaleFactor=1.15, minNeighbors=8, minSize=(50, 50))
    return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]


def draw_roi_on_frame(img: np.ndarray, roi: tuple[float, float, float, float]) -> None:
    """Draw ROI zone rectangle on image (numpy BGR) in-place. roi: (x1,y1,x2,y2) normalized 0-1."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi
    xa, ya = int(x1 * w), int(y1 * h)
    xb, yb = int(x2 * w), int(y2 * h)
    cv2.rectangle(img, (xa, ya), (xb, yb), (0, 255, 255), 2)
    cv2.putText(img, "Movement zone", (xa, ya - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_detections_on_frame(
    img: np.ndarray,
    objects: list[dict],
    faces: list[dict],
    roi: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Draw bounding boxes on image (numpy BGR) in-place. Optionally draw ROI zone.
    DETR returns normalized coords (0-1); we convert to pixels."""
    if roi is not None:
        draw_roi_on_frame(img, roi)
    h, w = img.shape[:2]
    for obj in objects:
        box = obj.get("box", {})
        xmin = float(box.get("xmin", 0))
        ymin = float(box.get("ymin", 0))
        xmax = float(box.get("xmax", 0))
        ymax = float(box.get("ymax", 0))
        # DETR/Cloudflare returns normalized 0-1
        if 0 < xmax <= 1 and 0 < ymax <= 1:
            xmin, xmax = int(xmin * w), int(xmax * w)
            ymin, ymax = int(ymin * h), int(ymax * h)
        else:
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        if xmax > xmin and ymax > ymin:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            label = obj.get("label", "?")
            cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for f in faces:
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]
        label = f.get("name", "face")
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return img


def draw_detections(
    image_bytes: bytes,
    objects: list[dict],
    faces: list[dict],
    roi: tuple[float, float, float, float] | None = None,
) -> bytes:
    """Draw bounding boxes on image and return as JPEG bytes. Optionally draw ROI zone."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    draw_detections_on_frame(img, objects, faces, roi=roi)
    _, jpeg = cv2.imencode(".jpg", img)
    return jpeg.tobytes()


def detect_movement(
    frame_bytes: bytes,
    prev_frame_gray: np.ndarray | None,
    threshold: float = 25.0,
    diff_pixel: int = 25,
    blur_size: int = 21,
    roi: tuple[float, float, float, float] | None = None,
) -> tuple[bool, np.ndarray | None]:
    """
    Compare current frame with previous. Returns (has_movement, current_gray_frame).
    prev_frame_gray can be None on first call.
    threshold: lower = more sensitive (default 25). Use 6-10 for small movements.
    diff_pixel: pixel diff threshold (default 25). Lower = picks up smaller changes.
    blur_size: smaller = more sensitive to fine movement (default 21). Use 11-15 for small movements.
    roi: (x1,y1,x2,y2) normalized 0-1. If set, only movement inside this zone is tracked.
    """
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, prev_frame_gray
    k = max(3, blur_size | 1)  # odd only
    img = cv2.GaussianBlur(img, (k, k), 0)
    if prev_frame_gray is None:
        return False, img
    diff = cv2.absdiff(prev_frame_gray, img)
    _, thresh = cv2.threshold(diff, diff_pixel, 255, cv2.THRESH_BINARY)
    if roi is not None:
        h, w = thresh.shape[:2]
        x1, y1, x2, y2 = roi
        mask = np.zeros_like(thresh)
        xa, ya = int(x1 * w), int(y1 * h)
        xb, yb = int(x2 * w), int(y2 * h)
        mask[ya:yb, xa:xb] = 255
        thresh = cv2.bitwise_and(thresh, mask)
        if np.sum(mask) > 0:
            mean_diff = np.sum(thresh) / np.sum(mask > 0)
        else:
            mean_diff = 0
    else:
        mean_diff = np.mean(thresh)
    return mean_diff > threshold, img


# Plate-like: alphanumeric 5-15 chars, at least 2 letters and 2 digits (MH12AB1234, DL01CA1234)
_PLATE_RE = re.compile(r"^[A-Z0-9]{5,15}$", re.IGNORECASE)


def _normalize_plate(text: str) -> str:
    """Normalize plate text: remove spaces, uppercase."""
    return re.sub(r"[\s\-\.]", "", text.strip().upper())


def _looks_like_plate(text: str) -> bool:
    """Check if text could be a license plate."""
    clean = re.sub(r"[\s\-\.]", "", text.strip().upper())
    if len(clean) < 5 or len(clean) > 15:
        return False
    letters = sum(1 for c in clean if c.isalpha())
    digits = sum(1 for c in clean if c.isdigit())
    return letters >= 2 and digits >= 2 and _PLATE_RE.match(clean)


_ocr_reader = None


def detect_vehicle_plates(image_bytes: bytes, objects: list[dict] | None = None) -> list[str]:
    """
    Run OCR on image and return text that looks like license plates.
    If objects provided, only runs when vehicles (car, truck, etc.) are detected.
    Requires: pip install easyocr. Set USE_PLATE_OCR=0 to disable.
    """
    if not USE_PLATE_OCR:
        return []
    if objects:
        labels = {o.get("label", "").lower() for o in objects}
        if not (labels & VEHICLE_LABELS):
            return []
    try:
        import easyocr
    except ImportError:
        return []

    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    try:
        results = _ocr_reader.readtext(img)
    except Exception:
        return []

    plates = []
    seen = set()
    for (_box, text, conf) in results:
        if conf < 0.3:
            continue
        clean = _normalize_plate(text)
        if not clean or clean in seen:
            continue
        if _looks_like_plate(text):
            plates.append(clean)
            seen.add(clean)
    return plates


def format_detections(
    objects: list[dict],
    faces: list[dict],
    visitor_names: list[str] | None = None,
    plates: list[str] | None = None,
) -> str:
    """Format detection results as human-readable text."""
    parts = []
    if objects:
        from collections import Counter
        counts = Counter(o["label"] for o in objects)
        parts.append("Objects: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if faces:
        if visitor_names:
            parts.append(f"👤 Visitor: {', '.join(visitor_names)}")
        else:
            parts.append(f"Faces: {len(faces)} detected")
    if plates:
        parts.append(f"🚗 Plate: {', '.join(plates)}")
    return "; ".join(parts) if parts else "No detections"


def get_faces_for_display(image_bytes: bytes, known_faces_dir: str | Path | None = None) -> tuple[list[dict], list[str]]:
    """
    Get faces for drawing and visitor names. Uses face_recognition if available, else OpenCV.
    Returns (faces_list for draw_detections, list of recognized names).
    """
    recognized = recognize_faces_with_boxes(image_bytes, known_faces_dir)
    if recognized:
        names = [r["name"] for r in recognized if r["name"] != "unknown"]
        return recognized, names
    faces = detect_faces(image_bytes)
    return faces, []


def recognize_faces(image_bytes: bytes, known_faces_dir: str | Path | None = None) -> list[str]:
    """
    Optional: Recognize faces using face_recognition library.
    Requires: pip install face_recognition
    Place known face images in known_faces_dir (e.g. ./known_faces/john.jpg).
    Returns list of matched names.
    """
    boxes = recognize_faces_with_boxes(image_bytes, known_faces_dir)
    return [b["name"] for b in boxes if b["name"] != "unknown"]


def recognize_faces_with_boxes(
    image_bytes: bytes, known_faces_dir: str | Path | None = None
) -> list[dict]:
    """
    Visitor ID: Recognize faces and return name + bounding box for each.
    Returns list of {name, x, y, w, h}. name="unknown" if no match.
    Requires: pip install face_recognition, known_faces/*.jpg
    """
    try:
        import face_recognition
    except ImportError:
        return []

    known_dir = Path(known_faces_dir) if known_faces_dir else Path(__file__).parent / "known_faces"
    if not known_dir.exists():
        return []

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    known_encodings = []
    known_names = []
    for f in known_dir.glob("*.jpg"):
        kimg = face_recognition.load_image_file(f)
        enc = face_recognition.face_encodings(kimg)
        if enc:
            known_encodings.append(enc[0])
            known_names.append(f.stem)

    if not known_encodings:
        return []

    face_locs = face_recognition.face_locations(rgb)  # (top, right, bottom, left)
    face_encs = face_recognition.face_encodings(rgb, face_locs)
    out = []
    for loc, enc in zip(face_locs, face_encs):
        top, right, bottom, left = loc
        x, y, w, h = left, top, right - left, bottom - top
        name = "unknown"
        if enc is not None and known_encodings:
            dists = face_recognition.face_distance(known_encodings, enc)
            if len(dists) > 0 and min(dists) < 0.6:
                idx = int(np.argmin(dists))
                name = known_names[idx]
        out.append({"name": name, "x": x, "y": y, "w": w, "h": h})
    return out
