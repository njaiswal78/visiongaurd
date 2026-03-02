#!/usr/bin/env python3
"""Launcher: replaces itself with telegram_bot using a readable process name for Activity Monitor."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.execv(sys.executable, ["RTSP Camera Bot", "telegram_bot.py"] + sys.argv[1:])
