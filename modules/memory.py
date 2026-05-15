"""
SQLite-backed persistent memory: persons, conversations, summaries.
Can be imported and used independently of the robot.
"""

import sqlite3
import os
import logging
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("DB_PATH", "reachy_memory.db"))
SUMMARY_THRESHOLD = 50  # auto-summarize after this many messages per person


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent read-write safety
    return conn


def init_db() -> None:
    """Create all tables if they do not exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS persons (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                notes       TEXT
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                role        TEXT    NOT NULL CHECK(role IN ('user','assistant')),
                content     TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                summary     TEXT    NOT NULL,
                up_to_msg   INTEGER NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)
    logger.info("Database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Person helpers
# ---------------------------------------------------------------------------

def get_or_create_person(name: str) -> int:
    """Return person_id, creating the record if needed."""
    with get_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO persons (name) VALUES (?)", (name,))
        row = conn.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()
        return row["id"]


def list_persons() -> list[dict]:
    with get_connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM persons ORDER BY name").fetchall()]


def update_person_notes(name: str, notes: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE persons SET notes=? WHERE name=?", (notes, name))


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def save_message(person_id: int, role: str, content: str) -> int:
    """Persist a single message and return its id."""
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (person_id, role, content) VALUES (?,?,?)",
            (person_id, role, content),
        )
        msg_id = cur.lastrowid

    _maybe_summarize(person_id)
    return msg_id


def load_recent_messages(person_id: int, limit: int = 10) -> list[dict]:
    """Return the last *limit* messages for a person as {role, content} dicts."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM conversations
            WHERE person_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (person_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def load_latest_summary(person_id: int) -> Optional[str]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT summary FROM summaries WHERE person_id=? ORDER BY id DESC LIMIT 1",
            (person_id,),
        ).fetchone()
    return row["summary"] if row else None


def _count_unsummarized(person_id: int) -> tuple[int, int]:
    """Return (total_messages, last_summarized_up_to_msg)."""
    with get_connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) as c FROM conversations WHERE person_id=?", (person_id,)
        ).fetchone()["c"]
        last = conn.execute(
            "SELECT up_to_msg FROM summaries WHERE person_id=? ORDER BY id DESC LIMIT 1",
            (person_id,),
        ).fetchone()
    last_up_to = last["up_to_msg"] if last else 0
    return total, last_up_to


def _maybe_summarize(person_id: int) -> None:
    """Trigger GPT summarization when unsummarized messages exceed the threshold."""
    total, last_up_to = _count_unsummarized(person_id)
    unsummarized = total - last_up_to
    if unsummarized < SUMMARY_THRESHOLD:
        return

    logger.info("Auto-summarizing %d messages for person_id=%d", unsummarized, person_id)
    try:
        _create_summary(person_id, total)
    except Exception:
        logger.exception("Auto-summarization failed for person_id=%d — continuing without summary", person_id)


def _create_summary(person_id: int, up_to_msg: int) -> None:
    """Ask GPT to summarize all conversations for a person and store the result."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE person_id=? ORDER BY id",
            (person_id,),
        ).fetchall()

    messages = [{"role": r["role"], "content": r["content"]} for r in rows]
    if not messages:
        logger.warning("_create_summary: no messages found for person_id=%d — skipping", person_id)
        return

    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    if not transcript.strip():
        logger.warning("_create_summary: empty transcript for person_id=%d — skipping", person_id)
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set — cannot create summary")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a memory assistant. Summarize the following conversation "
                    "history concisely, capturing key facts, topics discussed, and any "
                    "important details about the person. Keep it under 200 words."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        max_tokens=300,
    )
    summary_text = response.choices[0].message.content.strip()

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO summaries (person_id, summary, up_to_msg) VALUES (?,?,?)",
            (person_id, summary_text, up_to_msg),
        )
    logger.info("Summary saved for person_id=%d", person_id)


def build_memory_context(person_id: int) -> str:
    """
    Return a string that can be injected into the system prompt to give the
    model context about who it is talking to.
    """
    parts: list[str] = []

    summary = load_latest_summary(person_id)
    if summary:
        parts.append(f"Summary of past conversations:\n{summary}")

    with get_connection() as conn:
        person = conn.execute(
            "SELECT name, notes FROM persons WHERE id=?", (person_id,)
        ).fetchone()

    if person:
        parts.append(f"Du sprichst mit: {person['name']}.")
        if person["notes"]:
            parts.append(f"Notizen zu dieser Person: {person['notes']}")

    return "\n\n".join(parts)
