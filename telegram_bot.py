#!/usr/bin/env python3
"""
Telegram bot for RTSP camera - Ask questions about what your camera sees via Telegram.
Real-time monitoring: alerts with image when activity/movement detected.
"""

import argparse
import os
import sys
import threading
import time

import requests
from dotenv import load_dotenv

load_dotenv()

from camera_vision import (
    answer_historical,
    build_rtsp_url,
    capture_frame,
    chat_frame,
    chat_text_only,
)
from storage import (
    add_observation,
    format_observations_for_prompt,
    get_observations,
    get_latest_with_image,
    get_recent_observations,
    init_db,
)
from config import BOT_NAME
from roi_config import load_roi
from vision_detection import (
    USE_YOLO,
    detect_movement,
    detect_objects,
    detect_vehicle_plates,
    draw_detections,
    format_detections,
    get_faces_for_display,
)
from audio_utils import (
    capture_audio_chunk,
    check_rtsp_has_audio,
    detect_loud_sound,
    save_clip_to_ogg,
)

TELEGRAM_API = "https://api.telegram.org/bot"


def _log(msg: str) -> None:
    if os.environ.get("DEBUG") or os.environ.get("LOG_CALLS"):
        print(f"  [LOG] {msg}")


def send_photo(bot_token: str, chat_id: int, image_bytes: bytes, caption: str | None = None, parse_mode: str | None = None) -> bool:
    """Send a photo to a Telegram chat. Caption truncated if over 1024 chars."""
    url = f"{TELEGRAM_API}{bot_token}/sendPhoto"
    _log(f"POST api.telegram.org/.../sendPhoto (chat_id={chat_id})")
    if caption and len(caption) > 1024:
        caption = caption[:1021] + "..."
    try:
        files = {"photo": ("image.jpg", image_bytes, "image/jpeg")}
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        if parse_mode:
            data["parse_mode"] = parse_mode
        resp = requests.post(url, data=data, files=files, timeout=15)
        _log(f"  -> {resp.status_code}")
        return resp.status_code == 200
    except requests.RequestException:
        return False


def send_voice(bot_token: str, chat_id: int, audio_bytes: bytes, caption: str | None = None, parse_mode: str | None = None) -> bool:
    """Send voice/audio message to Telegram. OGG format preferred."""
    url = f"{TELEGRAM_API}{bot_token}/sendVoice"
    _log(f"POST api.telegram.org/.../sendVoice (chat_id={chat_id})")
    if caption and len(caption) > 1024:
        caption = caption[:1021] + "..."
    try:
        files = {"voice": ("clip.ogg", audio_bytes, "audio/ogg")}
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        if parse_mode:
            data["parse_mode"] = parse_mode
        resp = requests.post(url, data=data, files=files, timeout=15)
        _log(f"  -> {resp.status_code}")
        return resp.status_code == 200
    except requests.RequestException:
        return False


def send_message(bot_token: str, chat_id: int, text: str, parse_mode: str | None = None) -> bool:
    """Send a text message to a Telegram chat. Truncates if over 4096 chars."""
    url = f"{TELEGRAM_API}{bot_token}/sendMessage"
    if len(text) > 4096:
        text = text[:4093] + "..."
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def wants_photo(text: str) -> bool:
    """Check if user is asking for a frame/photo/image."""
    t = text.lower().strip()
    keywords = ("frame", "photo", "image", "picture", "snapshot", "send me", "show me")
    return any(k in t for k in keywords) or t in ("/photo", "/frame", "/pic")


def wants_detection(text: str) -> bool:
    """Check if user is asking for object/face detection."""
    t = text.lower()
    return any(k in t for k in ("detect", "objects", "faces", "what objects", "face detection"))


def wants_scene_description(text: str) -> bool:
    """True if user wants a general 'what do you see' description."""
    t = text.lower().strip()
    return any(k in t for k in ("kya dikh", "what you see", "what do you see", "kya ho rha", "describe", "batao kya"))

def wants_to_see_stored(text: str) -> bool:
    """True if user wants to see a stored image (e.g. last alert)."""
    t = text.lower().strip()
    keywords = ("show me", "dikha", "dikhao", "last alert", "last movement", "last image", "stored image")
    return t in ("/show", "/last") or any(k in t for k in keywords)


def needs_vision(text: str) -> bool:
    """True if user needs live frame/image. False = just chatting, send text only."""
    t = text.lower().strip()
    if t in ("/photo", "/frame", "/pic", "/detect"):
        return True
    if wants_photo(t) or wants_detection(t):
        return True
    vision_keywords = (
        "dikh", "dikh raha", "kya dikh", "koi hai", "dekho", "check karo", "check kar",
        "view", "scene", "describe", "capture", "snapshot", "kya dikh raha",
        "what do you see", "what see", "show me", "describe the",
    )
    return any(k in t for k in vision_keywords)


def is_historical_question(text: str) -> bool:
    """Check if user is asking about past observations."""
    t = text.lower()
    keywords = (
        "history", "historical", "past", "earlier", "before", "yesterday",
        "last hour", "last 30", "last hour", "did you see", "was there",
        "summary", "what happened", "recall", "remember", "saw earlier",
    )
    return any(k in t for k in keywords)


def run_movement_alert_loop(
    bot_token: str,
    rtsp_url: str,
    account_id: str,
    auth_token: str,
    gateway_id: str | None,
    alert_chat_ids: set,
    check_interval_sec: int = 10,
    cooldown_sec: int = 60,
    movement_threshold: float = 25.0,
    movement_diff_pixel: int = 25,
    movement_blur_size: int = 21,
) -> None:
    """Background: real-time monitoring: detect movement/activity, send alert with image to Telegram."""
    prev_gray = None
    last_alert_time = 0
    roi = load_roi()
    while True:
        time.sleep(check_interval_sec)
        if not alert_chat_ids:
            continue
        try:
            frame_bytes = capture_frame(rtsp_url)
            if not frame_bytes:
                continue
            has_movement, prev_gray = detect_movement(
                frame_bytes,
                prev_gray,
                threshold=movement_threshold,
                diff_pixel=movement_diff_pixel,
                blur_size=movement_blur_size,
                roi=roi,
            )
            if not has_movement:
                continue
            if time.time() - last_alert_time < cooldown_sec:
                continue
            objs = detect_objects(frame_bytes, account_id, auth_token, gateway_id)
            faces, visitor_names = get_faces_for_display(frame_bytes)
            plates = detect_vehicle_plates(frame_bytes, objs)
            det_txt = format_detections(objs, faces, visitor_names, plates)
            annotated = draw_detections(frame_bytes, objs, faces, roi=roi)
            desc = chat_frame(
                frame_bytes,
                "Detection: " + det_txt + ". Is data ke hisaab se ek sentence mein batao kya ho raha hai.",
                account_id,
                auth_token,
                gateway_id,
                detection_context=det_txt,
            )
            caption = f"⚠️ <b>Movement detect ho gayi!</b>\n\n🔍 {det_txt}"
            if desc and not desc.startswith("["):
                caption += f"\n\n📹 {desc}"
            add_observation(det_txt, image_data=annotated)
            for cid in list(alert_chat_ids):
                send_photo(bot_token, cid, annotated, caption=caption, parse_mode="HTML")
            last_alert_time = time.time()
        except Exception as e:
            print(f"  [Movement] Error: {e}")


def run_sound_alert_loop(
    bot_token: str,
    rtsp_url: str,
    alert_chat_ids: set,
    check_interval_sec: int = 15,
    cooldown_sec: int = 90,
    clip_duration_sec: float = 5.0,
) -> None:
    """Background: detect loud sounds (cry, scream), send short audio clip to Telegram."""
    last_alert_time = 0
    while True:
        time.sleep(check_interval_sec)
        if not alert_chat_ids:
            continue
        try:
            audio_bytes = capture_audio_chunk(rtsp_url, duration_sec=clip_duration_sec)
            if not audio_bytes:
                continue
            has_loud, max_rms = detect_loud_sound(audio_bytes)
            if not has_loud:
                continue
            if time.time() - last_alert_time < cooldown_sec:
                continue
            ogg_bytes = save_clip_to_ogg(audio_bytes)
            if not ogg_bytes:
                continue
            caption = f"🔊 <b>Aawaz detect hui!</b> (RMS: {max_rms:.2f})\n\nShort clip."
            for cid in list(alert_chat_ids):
                send_voice(bot_token, cid, ogg_bytes, caption=caption, parse_mode="HTML")
            last_alert_time = time.time()
        except Exception as e:
            print(f"  [Sound] Error: {e}")


def get_updates(bot_token: str, offset: int | None = None, timeout: int = 30) -> list:
    """Long-poll for new messages. Returns list of updates."""
    url = f"{TELEGRAM_API}{bot_token}/getUpdates"
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    try:
        resp = requests.get(url, params=params, timeout=timeout + 5)
        data = resp.json()
        if data.get("ok"):
            return data.get("result", [])
    except requests.RequestException:
        pass
    return []


def main():
    parser = argparse.ArgumentParser(description="Telegram bot for RTSP camera chat")
    parser.add_argument(
        "--rtsp",
        default=os.environ.get("RTSP_URL", "rtsp://192.168.1.100:554/stream1"),
        help="RTSP stream URL",
    )
    parser.add_argument("--username", "-u", default=None, help="RTSP username")
    parser.add_argument("--password", "-p", default=None, help="RTSP password")
    parser.add_argument(
        "--check-interval",
        type=int,
        default=int(os.environ.get("MOVEMENT_CHECK_SECONDS", "2")),
        help="Seconds between activity checks (default: 2 for real-time)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=int(os.environ.get("ALERT_COOLDOWN_SECONDS", "10")),
        help="Seconds between alerts (default: 10 for frequent notifications)",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable sound detection (if RTSP has no audio)",
    )
    parser.add_argument(
        "--sound-interval",
        type=int,
        default=int(os.environ.get("SOUND_CHECK_SECONDS", "15")),
        help="Seconds between sound checks (default: 15)",
    )
    args = parser.parse_args()

    # Movement sensitivity: lower threshold/diff = more sensitive (detects small movements)
    movement_threshold = float(os.environ.get("MOVEMENT_THRESHOLD", "8.0"))
    movement_diff_pixel = int(os.environ.get("MOVEMENT_DIFF_PIXEL", "15"))
    movement_blur_size = int(os.environ.get("MOVEMENT_BLUR_SIZE", "13"))

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    auth_token = os.environ.get("CLOUDFLARE_AUTH_TOKEN")
    gateway_id = os.environ.get("CLOUDFLARE_AI_GATEWAY_ID") or None
    allowed_user_id = os.environ.get("TELEGRAM_USER_ID")  # Optional: restrict by numeric ID
    allowed_username = os.environ.get("TELEGRAM_USERNAME")  # Optional: restrict by username (e.g. njaiswal78)

    username = args.username or os.environ.get("RTSP_USERNAME", "")
    password = args.password or os.environ.get("RTSP_PASSWORD", "")

    if not bot_token:
        print("Error: Set TELEGRAM_BOT_TOKEN in .env (get from @BotFather)")
        sys.exit(1)
    if not account_id or not auth_token:
        print("Error: Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN in .env")
        sys.exit(1)

    rtsp_url = build_rtsp_url(args.rtsp, username, password)

    init_db()
    alert_chat_ids: set[int] = set()

    # Start real-time monitoring: activity detected → alert with image
    movement_thread = threading.Thread(
        target=run_movement_alert_loop,
        args=(
            bot_token,
            rtsp_url,
            account_id,
            auth_token,
            gateway_id,
            alert_chat_ids,
            args.check_interval,
            args.cooldown,
            movement_threshold,
            movement_diff_pixel,
            movement_blur_size,
        ),
        daemon=True,
    )
    movement_thread.start()

    # Start sound monitoring: loud sound (cry, scream) → send short clip
    sound_enabled = False
    if not args.no_sound:
        if check_rtsp_has_audio(rtsp_url):
            sound_thread = threading.Thread(
                target=run_sound_alert_loop,
                args=(bot_token, rtsp_url, alert_chat_ids, args.sound_interval, 90, 5.0),
                daemon=True,
            )
            sound_thread.start()
            sound_enabled = True
        else:
            print("  (RTSP has no audio track – sound detection disabled. Use --no-sound to hide.)")

    print()
    print("=" * 60)
    print(f"  📹 {BOT_NAME}")
    print("=" * 60)
    print("  Bot ready hai. Koi bhi message bhejo.")
    if allowed_username or allowed_user_id:
        parts = []
        if allowed_username:
            parts.append(f"@{allowed_username}")
        if allowed_user_id:
            parts.append(f"ID:{allowed_user_id}")
        print(f"  (Restricted to: {', '.join(parts)})")
    else:
        print("  (No TELEGRAM_USERNAME/TELEGRAM_USER_ID - anyone can use the bot)")
    print(f"  Real-time monitoring: har {args.check_interval}s check | Activity pe image ke saath alert")
    if sound_enabled:
        print(f"  Sound monitoring: har {args.sound_interval}s | loud sound (cry/scream) pe clip bhejega")
    print("  Commands: /start, /help, /photo, /detect, /history, /show")
    print(f"  Detection: {'YOLOv8 (local)' if USE_YOLO else 'Cloudflare DETR'}")
    print("=" * 60)
    print()

    offset = None
    while True:
        try:
            updates = get_updates(bot_token, offset=offset)
            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message")
                if not msg:
                    continue

                chat_id = msg["chat"]["id"]
                from_user = msg["from"]
                user_id = str(from_user["id"])
                user_username = (from_user.get("username") or "").lower()
                text = (msg.get("text") or "").strip()

                # Access check: allow if username OR user ID matches (either can grant access)
                username_ok = (
                    allowed_username
                    and user_username
                    and user_username == allowed_username.strip().lower()
                )
                user_id_ok = allowed_user_id and user_id == str(allowed_user_id).strip()
                if (allowed_username or allowed_user_id) and not (username_ok or user_id_ok):
                    hint = ""
                    if not user_username and allowed_username:
                        hint = " Username set nahi hai. TELEGRAM_USER_ID use karo (@userinfobot se)."
                    send_message(
                        bot_token,
                        chat_id,
                        f"⛔ Ye bot private hai. .env mein TELEGRAM_USERNAME ya TELEGRAM_USER_ID set karo.{hint}",
                    )
                    continue

                alert_chat_ids.add(chat_id)

                if not text:
                    continue

                # Handle commands
                if text == "/start":
                    send_message(
                        bot_token,
                        chat_id,
                        f"📹 <b>{BOT_NAME}</b>\n\n"
                        "Baat karo ya live check karo.\n\n"
                        "• <b>Chat:</b> Hi, thanks – text reply\n"
                        "• <b>Live view:</b> Kya dikh raha? /photo – image ke saath\n"
                        "• /detect – objects & faces | /history – pichla ghanta\n\n"
                        "Activity pe image ke saath alert.",
                        parse_mode="HTML",
                    )
                    continue

                if text == "/help":
                    send_message(
                        bot_token,
                        chat_id,
                        f"📹 <b>{BOT_NAME}</b>\n\n"
                        "<b>Chat:</b> General baat – text only\n"
                        "<b>Live:</b> Kya dikh raha? /photo – image + details\n"
                        "<b>Detection:</b> /detect | /who – face recognition\n"
                        "<b>History:</b> /history – text only\n"
                        "<b>Stored:</b> /show – last alert image\n\n"
                        "Image sirf jab zarurat ho.",
                        parse_mode="HTML",
                    )
                    continue

                if text == "/who":
                    send_message(bot_token, chat_id, "👤 Visitor ID check kar rahi hoon...")
                    frame_bytes = capture_frame(rtsp_url)
                    if not frame_bytes:
                        send_message(bot_token, chat_id, "⚠️ Frame capture nahi ho paya.")
                        continue
                    faces, visitor_names = get_faces_for_display(frame_bytes)
                    caption = f"👤 Visitor: {', '.join(visitor_names)}" if visitor_names else "Koi pehchana face nahi. known_faces/ mein add karo (e.g. rahul.jpg), phir: pip install face_recognition"
                    if faces and not visitor_names:
                        caption += f"\n\n🔍 {len(faces)} face detect (unrecognized)"
                    annotated = draw_detections(frame_bytes, [], faces, roi=load_roi())
                    send_photo(bot_token, chat_id, annotated, caption=caption)
                    continue

                if text == "/detect":
                    send_message(bot_token, chat_id, "🔍 Objects, Visitor ID aur plate check kar rahi hoon...")
                    frame_bytes = capture_frame(rtsp_url)
                    if not frame_bytes:
                        send_message(bot_token, chat_id, "⚠️ Frame capture nahi ho paya.")
                        continue
                    objs = detect_objects(frame_bytes, account_id, auth_token, gateway_id)
                    faces, visitor_names = get_faces_for_display(frame_bytes)
                    plates = detect_vehicle_plates(frame_bytes, objs)
                    det_txt = format_detections(objs, faces, visitor_names, plates)
                    annotated = draw_detections(frame_bytes, objs, faces, roi=load_roi())
                    desc = chat_frame(
                        frame_bytes,
                        "Detection: " + det_txt + ". Is data ke hisaab se ek sentence mein batao kya dikh raha hai.",
                        account_id,
                        auth_token,
                        gateway_id,
                        detection_context=det_txt,
                    )
                    caption = f"🔍 {det_txt}"
                    if desc and not desc.startswith("["):
                        caption += f"\n\n📹 {desc}"
                    send_photo(bot_token, chat_id, annotated, caption=caption)
                    continue

                if text in ("/show", "/last") or wants_to_see_stored(text):
                    obs = get_latest_with_image()
                    if not obs:
                        send_message(bot_token, chat_id, "📷 Abhi koi stored image nahi. Activity detect hone par save hoga.")
                    else:
                        ts, desc, img_bytes = obs
                        from datetime import datetime
                        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                        caption = f"📷 [{dt}] {desc}"
                        send_photo(bot_token, chat_id, img_bytes, caption=caption)
                    continue

                if text == "/history":
                    send_message(bot_token, chat_id, "🔍 Memory mein dhundh raha hoon...")
                    obs = get_recent_observations(minutes=60)
                    if not obs:
                        send_message(bot_token, chat_id, "📜 Pichle ghante abhi koi observation nahi.")
                    else:
                        text_block = format_observations_for_prompt(obs)
                        if len(text_block) > 3500:
                            text_block = text_block[:3497] + "..."
                        send_message(bot_token, chat_id, f"📜 Pichla ghanta\n\n{text_block}")
                    continue

                # Detection request
                if wants_detection(text):
                    send_message(bot_token, chat_id, "🔍 Detect kar rahi hoon...")
                    frame_bytes = capture_frame(rtsp_url)
                    if not frame_bytes:
                        send_message(bot_token, chat_id, "⚠️ Frame capture nahi ho paya.")
                        continue
                    objs = detect_objects(frame_bytes, account_id, auth_token, gateway_id)
                    faces, visitor_names = get_faces_for_display(frame_bytes)
                    plates = detect_vehicle_plates(frame_bytes, objs)
                    det_txt = format_detections(objs, faces, visitor_names, plates)
                    annotated = draw_detections(frame_bytes, objs, faces, roi=load_roi())
                    desc = chat_frame(
                        frame_bytes,
                        "Detection: " + det_txt + ". Is data ke hisaab se ek sentence mein batao kya dikh raha hai.",
                        account_id,
                        auth_token,
                        gateway_id,
                        detection_context=det_txt,
                    )
                    caption = f"🔍 {det_txt}"
                    if desc and not desc.startswith("["):
                        caption += f"\n\n📹 {desc}"
                    send_photo(bot_token, chat_id, annotated, caption=caption)
                    continue

                # Historical question - answer from memory (text only)
                if is_historical_question(text):
                    send_message(bot_token, chat_id, "🔍 Memory mein dhundh raha hoon...")
                    obs = get_observations(limit=100)
                    obs_text = format_observations_for_prompt(obs)
                    if not obs or obs_text == "No observations recorded yet.":
                        send_message(bot_token, chat_id, "📜 Abhi koi observation store nahi. Activity pe record hoga.")
                    else:
                        response = answer_historical(obs_text, text, account_id, auth_token, gateway_id)
                        if response:
                            send_message(bot_token, chat_id, f"📜 {response}")
                        else:
                            send_message(bot_token, chat_id, "⚠️ AI se response nahi aaya.")
                    continue

                # Live chat: run detection first, pass to vision model for accurate description
                if needs_vision(text):
                    send_message(bot_token, chat_id, "📷 Capture kar rahi hoon...")
                    frame_bytes = capture_frame(rtsp_url)
                    if not frame_bytes:
                        send_message(bot_token, chat_id, "⚠️ Frame capture nahi ho paya. RTSP camera check karo.")
                        continue
                    objs = detect_objects(frame_bytes, account_id, auth_token, gateway_id)
                    faces, visitor_names = get_faces_for_display(frame_bytes)
                    plates = detect_vehicle_plates(frame_bytes, objs)
                    det_txt = format_detections(objs, faces, visitor_names, plates)
                    annotated = draw_detections(frame_bytes, objs, faces, roi=load_roi())
                    recent = get_recent_observations(minutes=30)
                    memory = format_observations_for_prompt(recent) if recent else None
                    prompt = (
                        "Is image mein humne detect kiya: " + det_txt + ". "
                        "Is detection data ke hisaab se 2-3 sentences mein batao kya dikh raha hai. SIRF detection data use karo."
                        if (wants_photo(text) or wants_scene_description(text))
                        else text
                    )
                    response = chat_frame(
                        frame_bytes,
                        prompt,
                        account_id,
                        auth_token,
                        gateway_id,
                        memory_context=memory,
                        detection_context=det_txt,
                    )
                    if det_txt != "No detections":
                        add_observation(det_txt)
                    caption = f"🔍 {det_txt}"
                    if response and not response.startswith("["):
                        caption += f"\n\n📹 {response}"
                    send_photo(bot_token, chat_id, annotated, caption=caption)
                else:
                    # Just chatting - text only, no frame
                    response = chat_text_only(text, account_id, auth_token, gateway_id)
                    if response:
                        send_message(bot_token, chat_id, response)
                    else:
                        send_message(bot_token, chat_id, "⚠️ Samajh nahi aaya. Kya dikh raha hai poochho ya /photo bhejo.")

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
