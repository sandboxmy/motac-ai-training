"""Tiny Flask chatbot that forwards messages to an Ollama model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from flask import Flask, jsonify, request


app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # Change to a model you have available, e.g., "llama2".


@dataclass
class ChatMessage:
    """Simple structure to hold a message from the user."""

    text: str


def ask_ollama(message: ChatMessage) -> str:
    """Send the user's text to Ollama and return the answer."""
    payload = {"model": MODEL_NAME, "prompt": message.text, "stream": False}

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=45)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - simple demo
        return (
            "I could not reach the Ollama server. "
            "Please confirm it is running with `ollama serve`."
            f" (Technical details: {exc})"
        )

    data: dict[str, Any] = response.json()
    return data.get("response", "I did not receive a reply from the AI model.")


@app.post("/chat")
def chat_endpoint():
    """Accept JSON {"message": "..."} and return the AI reply."""
    body = request.get_json(silent=True) or {}
    text = str(body.get("message", "")).strip()

    if not text:
        return jsonify({"error": "Please provide a message so I can help."}), 400

    answer = ask_ollama(ChatMessage(text=text))
    return jsonify({"reply": answer})


if __name__ == "__main__":
    # Reminder for training participants.
    print("Starting Flask development server on http://127.0.0.1:5000")
    print("Send a POST request to /chat with JSON: { 'message': 'Hello AI' }")
    app.run(debug=True)
