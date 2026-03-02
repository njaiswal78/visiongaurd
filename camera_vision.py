#!/usr/bin/env python3
"""
RTSP Camera Vision - Uses Cloudflare Workers AI to analyze camera feed and alert in terminal.
Captures frames from RTSP stream, sends to Llama 3.2 Vision model, prints what it sees.
"""

import argparse
import base64
import os
import subprocess
import sys
import time
from urllib.parse import urlparse, urlunparse

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

from config import ANALYZE_SYSTEM, CHAT_SYSTEM, CHAT_TEXT_SYSTEM, HISTORICAL_SYSTEM

# Set DEBUG=1 or LOG_CALLS=1 in .env to log API calls
def _log(msg: str) -> None:
    if os.environ.get("DEBUG") or os.environ.get("LOG_CALLS"):
        print(f"  [LOG] {msg}")

# Cloudflare Workers AI models
MODEL_ID = "@cf/meta/llama-3.2-11b-vision-instruct"
TEXT_MODEL_ID = "@cf/meta/llama-3.1-8b-instruct"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
GATEWAY_BASE = "https://gateway.ai.cloudflare.com/v1"


def _vision_url(account_id: str, gateway_id: str | None = None, model_id: str = MODEL_ID) -> str:
    """Return API URL: AI Gateway if gateway_id set, else direct Workers AI."""
    if gateway_id:
        return f"{GATEWAY_BASE}/{account_id}/{gateway_id}/workers-ai/{model_id}"
    return f"{API_BASE}/{account_id}/ai/run/{model_id}"


def _vision_headers(auth_token: str, gateway_id: str | None = None) -> dict:
    """Authorization headers. When using gateway, add cf-aig-authorization if Authenticated Gateway is enabled."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    if gateway_id:
        # AI Gateway with Authenticated Gateway enabled requires cf-aig-authorization
        gateway_token = os.environ.get("CLOUDFLARE_AI_GATEWAY_AUTH_TOKEN") or auth_token
        headers["cf-aig-authorization"] = f"Bearer {gateway_token}"
    return headers


def build_rtsp_url(base_url: str, username: str, password: str) -> str:
    """Build RTSP URL with authentication credentials."""
    parsed = urlparse(base_url)
    if username and password:
        netloc = f"{username}:{password}@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        parsed = parsed._replace(netloc=netloc)
    return urlunparse(parsed)


def _capture_frame_ffmpeg(rtsp_url: str, timeout: int, max_dim: int) -> bytes | None:
    """Fallback: capture a single frame using ffmpeg CLI. More reliable for RTSP on macOS."""
    try:
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-analyzeduration", "1M",
            "-probesize", "1M",
            "-i", rtsp_url,
            "-vframes", "1",
            "-f", "mjpeg",
            "pipe:1",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        jpeg = result.stdout
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        _, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return out.tobytes()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def capture_frame(rtsp_url: str, timeout: int = 10, max_dim: int = 960) -> bytes | None:
    """Capture a single frame from RTSP stream, return as JPEG bytes. max_dim=960 for better detection.
    Uses ffmpeg CLI first (avoids OpenCV FFmpeg warning on macOS), falls back to OpenCV."""
    _log(f"capture_frame(rtsp=***, timeout={timeout}, max_dim={max_dim})")

    # Prefer ffmpeg for RTSP - avoids "VIDEOIO(FFMPEG): can't capture by name" warning on macOS
    result = _capture_frame_ffmpeg(rtsp_url, timeout, max_dim)
    if result is not None:
        return result

    # Fallback to OpenCV
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    timeout_ms = timeout * 1000
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MS"):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MS, timeout_ms)
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MS"):
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MS, timeout_ms)

    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpeg.tobytes()


def analyze_frame(
    image_bytes: bytes,
    account_id: str,
    auth_token: str,
    gateway_id: str | None = None,
) -> str | None:
    """Send frame to Cloudflare Workers AI vision model and return description."""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_image}"

    url = _vision_url(account_id, gateway_id)
    _log(f"POST {url} (analyze_frame)")
    headers = _vision_headers(auth_token, gateway_id)
    payload = {
        "messages": [
            {
                "role": "system",
                "content": ANALYZE_SYSTEM,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is image ko detail se analyze karo. Pehle detect karo: kitne log, koi vehicle, janwar. Phir short varnan do. Koi unusual cheez ho to pehle batao."},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        "max_tokens": 256,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        _log(f"  -> {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True):
            errs = data.get("errors", [])
            return f"[API Error: {errs or 'Unknown'}]"
        return data.get("result", {}).get("response")
    except requests.RequestException as e:
        return f"[API Error: {e}]"
    except (KeyError, TypeError) as e:
        return f"[Parse Error: {e}]"


def chat_frame(
    image_bytes: bytes,
    user_message: str,
    account_id: str,
    auth_token: str,
    gateway_id: str | None = None,
    memory_context: str | None = None,
    detection_context: str | None = None,
) -> str | None:
    """Send frame + user message to vision model, return AI response.
    detection_context: DETR results (e.g. 'Objects: 2 person, 1 car') - LLM must use ONLY these, no inventing."""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_image}"

    system = CHAT_SYSTEM
    if detection_context:
        system += f"\n\nDetection data (ground truth - use ONLY this, do NOT invent): {detection_context}"
    if memory_context:
        system += f"\n\nRecent observations (for context):\n{memory_context}"

    url = _vision_url(account_id, gateway_id)
    _log(f"POST {url} (chat_frame: {(user_message[:47] + '...') if len(user_message) > 50 else user_message})")
    headers = _vision_headers(auth_token, gateway_id)
    payload = {
        "messages": [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        "max_tokens": 512,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        _log(f"  -> {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True):
            errs = data.get("errors", [])
            return f"[API Error: {errs or 'Unknown'}]"
        return data.get("result", {}).get("response")
    except requests.RequestException as e:
        return f"[API Error: {e}]"
    except (KeyError, TypeError) as e:
        return f"[Parse Error: {e}]"


def chat_text_only(
    user_message: str,
    account_id: str,
    auth_token: str,
    gateway_id: str | None = None,
) -> str | None:
    """Text-only chat (no image). For general conversation when user is just chatting."""
    url = _vision_url(account_id, gateway_id, TEXT_MODEL_ID)
    _log(f"POST {url} (chat_text_only)")
    headers = _vision_headers(auth_token, gateway_id)
    payload = {
        "messages": [
            {"role": "system", "content": CHAT_TEXT_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 256,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        _log(f"  -> {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True):
            return None
        result = data.get("result", {})
        return result.get("response") or result.get("result")
    except requests.RequestException:
        return None
    except (KeyError, TypeError):
        return None


def answer_historical(
    observations_text: str,
    user_message: str,
    account_id: str,
    auth_token: str,
    gateway_id: str | None = None,
) -> str | None:
    """Answer question from stored observations (no image). Uses text-only model."""
    url = _vision_url(account_id, gateway_id, TEXT_MODEL_ID)
    _log(f"POST {url} (answer_historical)")
    headers = _vision_headers(auth_token, gateway_id)
    payload = {
        "messages": [
            {
                "role": "system",
                "content": HISTORICAL_SYSTEM,
            },
            {
                "role": "user",
                "content": f"Past observations:\n\n{observations_text}\n\nUser question: {user_message}",
            },
        ],
        "max_tokens": 512,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        _log(f"  -> {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True):
            errs = data.get("errors", [])
            return f"[API Error: {errs or 'Unknown'}]"
        result = data.get("result", {})
        return result.get("response") or result.get("result")
    except requests.RequestException as e:
        return f"[API Error: {e}]"
    except (KeyError, TypeError) as e:
        return f"[Parse Error: {e}]"


def agree_to_license(
    account_id: str,
    auth_token: str,
    verbose: bool = False,
    gateway_id: str | None = None,
) -> bool:
    """Agree to Meta license (required once before using Llama 3.2 Vision)."""
    url = _vision_url(account_id, gateway_id)
    _log(f"POST {url} (agree_to_license)")
    headers = {**_vision_headers(auth_token, gateway_id), "Content-Type": "application/json"}
    try:
        resp = requests.post(url, headers=headers, json={"prompt": "agree"}, timeout=15)
        _log(f"  -> {resp.status_code}")
        body = resp.json() if resp.text else {}
        errors = body.get("errors", [])
        # Cloudflare returns 403 with "Thank you for agreeing" when agreement succeeds
        if resp.status_code == 403 and errors:
            msg = str(errors[0].get("message", ""))
            if "Thank you for agreeing" in msg or "You may now use the model" in msg:
                return True
        if verbose or resp.status_code not in (200, 403):
            print(f"  API status: {resp.status_code}")
            if errors:
                print(f"  Errors: {errors}")
            elif "result" in body:
                print(f"  Response: {body.get('result')}")
        return resp.status_code == 200
    except requests.RequestException as e:
        if verbose:
            print(f"  Request error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="RTSP camera vision with Cloudflare AI")
    parser.add_argument(
        "--rtsp",
        default=os.environ.get("RTSP_URL", "rtsp://192.168.1.100:554/stream1"),
        help="RTSP stream URL (or set RTSP_URL in .env)",
    )
    parser.add_argument("--username", "-u", default=None, help="RTSP username")
    parser.add_argument("--password", "-p", default=None, help="RTSP password")
    parser.add_argument("--interval", "-i", type=int, default=10, help="Seconds between captures")
    parser.add_argument("--agree", action="store_true", help="Agree to Meta license (run once)")
    args = parser.parse_args()

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    auth_token = os.environ.get("CLOUDFLARE_AUTH_TOKEN")
    gateway_id = os.environ.get("CLOUDFLARE_AI_GATEWAY_ID") or None
    username = args.username or os.environ.get("RTSP_USERNAME", "")
    password = args.password or os.environ.get("RTSP_PASSWORD", "")

    if not account_id or not auth_token:
        print("Error: Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN in .env or environment")
        sys.exit(1)

    if args.agree:
        print("Agreeing to Meta license...")
        if agree_to_license(account_id, auth_token, verbose=True, gateway_id=gateway_id):
            print("License agreed. You can now run without --agree.")
        else:
            print("Failed to agree. Check credentials.")
        sys.exit(0)

    rtsp_url = build_rtsp_url(args.rtsp, username, password)

    print("=" * 60)
    print("RTSP Camera Vision - Cloudflare Workers AI")
    print("=" * 60)
    print(f"Stream: {args.rtsp}")
    print(f"Interval: {args.interval}s | Model: {MODEL_ID}")
    if gateway_id:
        print(f"Gateway: {gateway_id}")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    while True:
        try:
            print(f"\n[{time.strftime('%H:%M:%S')}] Capturing frame...")
            frame_bytes = capture_frame(rtsp_url)

            if not frame_bytes:
                print("  ⚠ Could not capture frame. Check RTSP URL and credentials.")
                time.sleep(args.interval)
                continue

            print("  Analyzing with Cloudflare AI...")
            description = analyze_frame(frame_bytes, account_id, auth_token, gateway_id)

            if description:
                print("\n  📹 CAMERA SEES:")
                print("  " + "-" * 40)
                for line in description.strip().split("\n"):
                    print(f"  {line}")
                print("  " + "-" * 40)
            else:
                print("  ⚠ No response from AI")

        except KeyboardInterrupt:
            print("\n\nStopped.")
            break
        except Exception as e:
            print(f"  ⚠ Error: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
