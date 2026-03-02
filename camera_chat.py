#!/usr/bin/env python3
"""
Chat with your RTSP camera - Ask questions about what the camera sees.
Each message captures a fresh frame and sends it to the AI with your question.
"""

import argparse
import os
import sys
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

load_dotenv()

from camera_vision import build_rtsp_url, capture_frame, chat_frame
from vision_detection import detect_objects, detect_vehicle_plates, format_detections, get_faces_for_display

# Commands
CMD_QUIT = ("/quit", "/q", "/exit")
CMD_HELP = ("/help", "/h")
CMD_REFRESH = ("/refresh", "/r")


def main():
    parser = argparse.ArgumentParser(description="Chat with your RTSP camera")
    parser.add_argument(
        "--rtsp",
        default=os.environ.get("RTSP_URL", "rtsp://192.168.1.100:554/stream1"),
        help="RTSP stream URL",
    )
    parser.add_argument("--username", "-u", default=None, help="RTSP username")
    parser.add_argument("--password", "-p", default=None, help="RTSP password")
    args = parser.parse_args()

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    auth_token = os.environ.get("CLOUDFLARE_AUTH_TOKEN")
    gateway_id = os.environ.get("CLOUDFLARE_AI_GATEWAY_ID") or None
    username = args.username or os.environ.get("RTSP_USERNAME", "")
    password = args.password or os.environ.get("RTSP_PASSWORD", "")

    if not account_id or not auth_token:
        print("Error: Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN in .env")
        sys.exit(1)

    rtsp_url = build_rtsp_url(args.rtsp, username, password)

    print()
    print("=" * 60)
    print("  📹 Chat with Camera - RTSP + Cloudflare AI")
    print("=" * 60)
    print(f"  Stream: {args.rtsp}")
    print("  Ask anything about what the camera sees.")
    print()
    print("  Commands: /quit, /help, /refresh (new frame)")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in CMD_QUIT:
            print("Bye!")
            break

        if user_input.lower() in CMD_HELP:
            print()
            print("  Type a question about what the camera sees, e.g.:")
            print("    • What do you see?")
            print("    • Is anyone in the room?")
            print("    • Describe the scene.")
            print("    • Are the lights on?")
            print()
            print("  Commands: /quit, /help, /refresh")
            print()
            continue

        if user_input.lower() in CMD_REFRESH:
            user_input = "What do you see in this image? Describe briefly."

        print("  Capturing frame...")
        frame_bytes = capture_frame(rtsp_url)

        if not frame_bytes:
            print("  ⚠ Could not capture frame. Check RTSP URL and credentials.")
            continue

        print("  Detecting...")
        objs = detect_objects(frame_bytes, account_id, auth_token, gateway_id)
        faces, visitor_names = get_faces_for_display(frame_bytes)
        plates = detect_vehicle_plates(frame_bytes, objs)
        det_txt = format_detections(objs, faces, visitor_names, plates)
        print("  Thinking...")
        response = chat_frame(
            frame_bytes, user_input, account_id, auth_token, gateway_id,
            detection_context=det_txt,
        )

        if det_txt != "No detections":
            print("  🔍 " + det_txt)
        if response:
            print()
            print("  📹 Camera:")
            print("  " + "-" * 40)
            for line in response.strip().split("\n"):
                print(f"  {line}")
            print("  " + "-" * 40)
        else:
            print("  ⚠ No response from AI")
        print()


if __name__ == "__main__":
    main()
