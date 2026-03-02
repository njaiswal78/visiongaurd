#!/usr/bin/env python3
"""SQLite storage for camera observations - enables memory and historical queries."""

import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "observations.db"


def init_db() -> None:
    """Create the observations table if it doesn't exist. Add image_data column if missing."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                description TEXT NOT NULL,
                image_data BLOB,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(timestamp)"
        )
        # Migrate existing table: add image_data if missing
        cols = [r[1] for r in conn.execute("PRAGMA table_info(observations)").fetchall()]
        if "image_data" not in cols:
            conn.execute("ALTER TABLE observations ADD COLUMN image_data BLOB")


def add_observation(description: str, timestamp: float | None = None, image_data: bytes | None = None) -> int:
    """Store an observation. Optionally include image. Returns row id."""
    ts = timestamp or time.time()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO observations (timestamp, description, image_data) VALUES (?, ?, ?)",
            (ts, description, image_data),
        )
        return cur.lastrowid or 0


def get_latest_with_image() -> tuple[float, str, bytes] | None:
    """Get most recent observation that has an image. Returns (timestamp, description, image_bytes) or None."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT timestamp, description, image_data FROM observations WHERE image_data IS NOT NULL ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if row and row["image_data"]:
            return (row["timestamp"], row["description"], bytes(row["image_data"]))
    return None


def get_recent_with_images(limit: int = 10) -> list[tuple[float, str, bytes]]:
    """Get recent observations that have images. Returns list of (timestamp, description, image_bytes)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT timestamp, description, image_data FROM observations WHERE image_data IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(r["timestamp"], r["description"], bytes(r["image_data"])) for r in rows if r["image_data"]]


def get_observations(
    since: float | None = None,
    until: float | None = None,
    limit: int = 200,
) -> list[tuple[float, str]]:
    """Get observations as list of (timestamp, description)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT timestamp, description FROM observations WHERE 1=1"
        params: list = []
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since)
        if until is not None:
            query += " AND timestamp <= ?"
            params.append(until)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [(r["timestamp"], r["description"]) for r in rows]


def get_recent_observations(minutes: int = 5) -> list[tuple[float, str]]:
    """Get observations from the last N minutes."""
    since = time.time() - (minutes * 60)
    return get_observations(since=since, limit=50)


def format_observations_for_prompt(obs: list[tuple[float, str]]) -> str:
    """Format observations as text for AI context."""
    if not obs:
        return "No observations recorded yet."
    lines = []
    for ts, desc in reversed(obs):  # Chronological order
        from datetime import datetime
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        lines.append(f"- [{dt}] {desc}")
    return "\n".join(lines)
