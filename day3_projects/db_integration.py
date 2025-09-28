"""Show how to combine SQLite queries with Ollama explanations."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import requests

DB_PATH = Path(__file__).resolve().parent / "training_data.db"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    description TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL
);
"""

SEED_DATA = [
    ("AI Basics", "Introduction to AI concepts", 60),
    ("Using Ollama", "Hands-on session with local models", 90),
    ("Prompt Crafting", "Practice writing effective prompts", 45),
    ("Web Development", "Hands-on session with local models", 90),
]


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create the table and insert seed rows if empty."""
    connection.execute(CREATE_TABLE_SQL)
    connection.commit()

    count = connection.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
    if count == 0:
        connection.executemany(
            "INSERT INTO lessons (topic, description, duration_minutes) VALUES (?, ?, ?)",
            SEED_DATA,
        )
        connection.commit()


def fetch_lessons(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all lessons as dictionaries for easy printing."""
    cursor = connection.execute(
        "SELECT id, topic, description, duration_minutes FROM lessons"
    )
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def ask_ollama_about_lesson(lesson: dict[str, Any]) -> str:
    """Request a friendly explanation of the lesson using Ollama."""
    prompt = (
        "You are a helpful course assistant. Explain the lesson below in simple terms "
        "for beginners. Add one tip for learners at the end.\n\n"
        f"Lesson data: {lesson}\n"
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=45,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - simple demo
        return (
            "Could not reach Ollama. Please ensure the server is running.\n"
            f"Details: {exc}"
        )

    data: dict[str, Any] = response.json()
    return data.get("response", "Model returned no text.")


def main() -> None:
    """Create the database, fetch lessons, and ask for AI explanations."""
    print(f"Connecting to SQLite database at {DB_PATH}")
    connection = sqlite3.connect(DB_PATH)

    try:
        initialize_database(connection)
        lessons = fetch_lessons(connection)
    finally:
        connection.close()

    if not lessons:
        print("No lessons found in the database.")
        return

    for lesson in lessons:
        print("-" * 60)
        print(f"Lesson {lesson['id']}: {lesson['topic']}")
        print(f"Duration: {lesson['duration_minutes']} minutes")
        print(f"Description: {lesson['description']}")

        explanation = ask_ollama_about_lesson(lesson)
        print("\nAI explanation:")
        print(explanation)
        print("-" * 60 + "\n")

    print("All lessons processed. Share these explanations with learners.")


if __name__ == "__main__":
    main()
