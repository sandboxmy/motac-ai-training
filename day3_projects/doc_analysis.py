"""Analyze a document using Ollama to summarize and extract keywords."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

# Paths and configuration values grouped at the top for quick changes.
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_document.txt"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"


def read_document() -> str:
    """Load the text file we want the AI to analyze."""
    return DATA_PATH.read_text(encoding="utf-8")


def call_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the AI's response."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=45,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - simple demo
        return "Could not reach the Ollama server. Please start it first.\n" + str(exc)

    data: dict[str, Any] = response.json()
    return data.get("response", "Model did not return text.")


def main() -> None:
    """Use the model to summarize and extract keywords from the document."""
    print(f"Reading document from {DATA_PATH}")
    # Step 1: load the document from disk.
    document_text = read_document()

    prompt = (
        "You are a helpful assistant. Read the document below and provide:\n"
        "1. A short summary (2-3 sentences).\n"
        "2. A bullet list of 3 important keywords.\n\n"
        f"Document:\n{document_text}\n"
    )

    print("Sending request to Ollama...")
    # Step 2: ask the model for a summary and keywords.
    result = call_ollama(prompt)

    print("\nAI analysis:")
    # Step 3: show the AI-generated insights to the participant.
    print(result)


if __name__ == "__main__":
    main()
