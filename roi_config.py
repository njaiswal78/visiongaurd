#!/usr/bin/env python3
"""ROI (Region of Interest) config for movement tracking. Zone in normalized 0-1 coords."""

import json
from pathlib import Path

ROI_FILE = Path(__file__).parent / "roi_zone.json"


def load_roi() -> tuple[float, float, float, float] | None:
    """
    Load ROI zone. Returns (x1, y1, x2, y2) normalized 0-1, or None for full frame.
    x1,y1 = top-left, x2,y2 = bottom-right.
    """
    if not ROI_FILE.exists():
        return None
    try:
        data = json.loads(ROI_FILE.read_text())
        x1 = float(data.get("x1", 0))
        y1 = float(data.get("y1", 0))
        x2 = float(data.get("x2", 1))
        y2 = float(data.get("y2", 1))
        if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
            return (x1, y1, x2, y2)
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return None


def save_roi(x1: float, y1: float, x2: float, y2: float) -> None:
    """Save ROI zone (normalized 0-1)."""
    ROI_FILE.write_text(json.dumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2}, indent=2))


def roi_to_pixel_rect(roi: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert normalized ROI to pixel rect (x, y, w, h)."""
    x1, y1, x2, y2 = roi
    x = int(x1 * width)
    y = int(y1 * height)
    w = int((x2 - x1) * width)
    h = int((y2 - y1) * height)
    return (x, y, w, h)
