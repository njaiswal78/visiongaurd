#!/bin/bash
# Uninstall the RTSP AI Vision Telegram bot LaunchAgent.

PLIST_NAME="com.rtsp-ai-vision.telegram-bot"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
PLIST_DEST="$LAUNCH_AGENTS/${PLIST_NAME}.plist"

if [[ -f "$PLIST_DEST" ]]; then
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
    rm -f "$PLIST_DEST"
    echo "Service uninstalled."
else
    echo "Service not installed."
fi
