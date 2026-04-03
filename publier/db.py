"""SQLite posting log — tracks what's been posted, prevents duplicates."""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def get_db_path() -> Path:
    import os
    return Path(os.getenv("QC_DB_PATH", "quiet_company.db"))


@contextmanager
def get_db():
    """Thread-safe SQLite connection matching app.py pattern."""
    conn = sqlite3.connect(str(get_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
    finally:
        conn.close()


def init_publier_db():
    """Create publier_posts table. Safe to call every run."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS publier_posts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                calendar_row    INTEGER NOT NULL,
                platform        TEXT    NOT NULL,
                caption         TEXT,
                image_path      TEXT,
                status          TEXT    NOT NULL DEFAULT 'pending',
                platform_post_id TEXT,
                error           TEXT,
                retries         INTEGER NOT NULL DEFAULT 0,
                scheduled_at    TEXT,
                posted_at       TEXT,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(calendar_row, platform)
            );
        """)
        conn.commit()
    logger.info("Publier database ready")


def is_posted(calendar_row: int, platform: str) -> bool:
    """Check if a post has already been successfully published."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT status FROM publier_posts WHERE calendar_row = ? AND platform = ?",
            (calendar_row, platform),
        ).fetchone()
        return row is not None and row["status"] == "posted"


def log_post(
    calendar_row: int,
    platform: str,
    status: str,
    caption: str = "",
    image_path: str = "",
    platform_post_id: str = "",
    error: str = "",
    scheduled_at: str = "",
):
    """Insert or update a posting record."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO publier_posts
                (calendar_row, platform, caption, image_path, status,
                 platform_post_id, error, scheduled_at, posted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(calendar_row, platform) DO UPDATE SET
                status = excluded.status,
                platform_post_id = excluded.platform_post_id,
                error = excluded.error,
                retries = retries + 1,
                posted_at = excluded.posted_at
            """,
            (
                calendar_row,
                platform,
                caption[:500],
                image_path,
                status,
                platform_post_id,
                error,
                scheduled_at,
                datetime.utcnow().isoformat() if status == "posted" else "",
            ),
        )
        conn.commit()


def get_pending_retries(max_retries: int = 3) -> list[dict]:
    """Get failed posts eligible for retry."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT calendar_row, platform, retries
               FROM publier_posts
               WHERE status = 'failed' AND retries < ?
               ORDER BY created_at""",
            (max_retries,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_posts() -> list[dict]:
    """Get full posting log for status display."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, calendar_row, platform, status, retries,
                      platform_post_id, error, scheduled_at, posted_at, created_at
               FROM publier_posts
               ORDER BY created_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
