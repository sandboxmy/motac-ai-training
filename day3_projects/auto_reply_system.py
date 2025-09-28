"""Mock email auto-reply system powered by an Ollama model."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


@dataclass
class EmailMessage:
    """Tiny container representing an incoming email."""

    sender: str
    subject: str
    body: str


def load_inbox() -> list[EmailMessage]:
    """Return a small list of demo emails."""
    return [
        EmailMessage(
            sender="ceo@example.com",
            subject="Product update request",
            body="Can you share a friendly summary of our AI training progress?",
        ),
        EmailMessage(
            sender="support@example.com",
            subject="Customer question",
            body="A customer asked how to restart the AI assistant when it stops responding.",
        ),
    ]


def call_ollama(email: EmailMessage) -> str:
    """Ask the model to draft a professional auto-response."""
    prompt = (
        "You are an assistant writing a polite email reply.\n"
        f"Sender: {email.sender}\n"
        f"Subject: {email.subject}\n"
        f"Message: {email.body}\n\n"
        "Write a short, friendly response with 2 paragraphs and a clear next step."
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=45,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - simple demo
        return f"(Could not reach Ollama. Please start it first.)\nDetails: {exc}"

    data: dict[str, Any] = response.json()
    return data.get("response", "(Model returned no text.)")


def print_draft(email: EmailMessage, draft: str) -> None:
    """Display the drafted reply for review."""
    separator = "-" * 60
    print(separator)
    print(f"To: {email.sender}")
    print(f"Subject: Re: {email.subject}")
    print(separator)
    print(draft)
    print(separator + "\n")


def main() -> None:
    """Generate AI-powered auto-replies for every message in the inbox."""
    inbox = load_inbox()
    if not inbox:
        print("Inbox is empty. Nothing to reply to today!")
        return

    print("Generating drafts...\n")
    for email in inbox:
        draft = call_ollama(email)
        print_draft(email, draft)

    print("All drafts ready. Review them before sending to customers.")


if __name__ == "__main__":
    main()
