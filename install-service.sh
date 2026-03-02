#!/bin/bash
# Install RTSP AI Vision Telegram bot as a macOS LaunchAgent (background service).
# Runs when you log in, keeps running, restarts if it crashes.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PLIST_NAME="com.rtsp-ai-vision.telegram-bot"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
PLIST_DEST="$LAUNCH_AGENTS/${PLIST_NAME}.plist"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

# Use actual python3 path (avoids "env" showing in Activity Monitor)
PYTHON_PATH="$(which python3)"
if [[ -z "$PYTHON_PATH" ]]; then
    echo "Error: python3 not found in PATH"
    exit 1
fi

# Replace placeholders in plist (use | as sed delimiter since paths contain /)
sed -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__LOG_DIR__|$LOG_DIR|g" \
    -e "s|__PYTHON_PATH__|$PYTHON_PATH|g" \
    "$PROJECT_DIR/${PLIST_NAME}.plist" > "$PLIST_DEST"


echo "Installed plist to $PLIST_DEST"
echo "Logs: $LOG_DIR/telegram-bot.log"

# Load the service (starts it now)
launchctl load "$PLIST_DEST"
echo "Service started. It will run at login and restart if it crashes."

echo ""
echo "Commands:"
echo "  Stop:   launchctl unload $PLIST_DEST"
echo "  Start:  launchctl load $PLIST_DEST"
echo "  Status: launchctl list | grep $PLIST_NAME"
echo "  Logs:   tail -f $LOG_DIR/telegram-bot.log"
